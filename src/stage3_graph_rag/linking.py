from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import pandas as pd

# Node types that name something a question can ask about; capabilities, architectures and
# concepts ("reasoning", "base", "release") are common words and are not linked
NAMED_TYPES = {
    "AI_MODEL", "FOUNDATION_MODEL", "COMPANY", "ORG", "PERSON", "PRODUCT", "PLATFORM", "HARDWARE",
    "BENCHMARK", "DATASET", "LICENSE", "STANDARD", "LAW", "GPE", "TECHNOLOGY",
}
# Name fragments left by Stage 1 extraction (e.g. "Ultra" from "Nemotron 3 Ultra") and generic product words
GENERIC_NAMES = {
    "agent", "agents", "agreement", "api", "app", "base", "chat", "cloud", "code", "data", "flash", "hub",
    "instruct", "ip", "lightning", "lite", "max", "mini", "model", "models", "nano", "omni", "open", "plan",
    "plus", "pro", "readme", "release", "sdk", "search", "studio", "super", "ui", "ultra", "vision",
}
# Same character replacements as the Stage 1 and WP1 matching (dashes and quotes)
MATCH_MAP = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-", "−": "-", "‘": "'", "’": "'", "“": '"', "”": '"'})
SEPARATOR = r"[\s\-_/]?"


@dataclass(frozen=True)
class EntityMatch:
    entity_id: str
    name: str
    entity_type: str
    span: tuple[int, int]
    surface: str
    method: str  # node_name | family_version | family_lexicon

    def as_dict(self) -> dict[str, Any]:
        return {"entity_id": self.entity_id, "name": self.name, "entity_type": self.entity_type,
                "surface": self.surface, "method": self.method}


def _list(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    text = str(value).strip()
    if text.startswith("["):
        try:
            return [str(v) for v in json.loads(text)]
        except json.JSONDecodeError:
            return []
    return [part.strip() for part in text.split(",") if part.strip()]


def name_pattern(name: str) -> re.Pattern | None:
    """A name with free separators (`Llama 3.2` = `Llama-3.2` = `Llama3.2`) that does not match inside a longer
    name or version (`Qwen3` does not match `Qwen3.8`). Names of two characters must match their case (`Yi`, `R1`)."""
    tokens = re.findall(r"[^\s\-_/]+", name.translate(MATCH_MAP))
    if not tokens or name.strip().lower() in GENERIC_NAMES or len("".join(tokens)) < 2:
        return None
    if re.fullmatch(r"[vV]?\d+(?:\.\d+)*[bBkKmM]?", name.strip()):  # a bare version or size ("v2", "70B")
        return None
    body = SEPARATOR.join(re.escape(token) for token in tokens)
    flags = 0 if len("".join(tokens)) <= 2 else re.IGNORECASE
    return re.compile(rf"(?<![\w.]){body}(?![\w+]|\.\d)", flags)


class EntityLinker:
    """Link the entities named in a question to knowledge graph nodes.

    A node is found by its name, its aliases and, for an open-model family, its version names (`Gemma 3` -> Gemma).
    The longest match wins, so `Llama 3.1 70B` is not also linked as `Llama`. A family mentioned only through a name
    the graph does not hold (`Qwen3.8`) is found with the WP1 lexicon patterns. A lower-case match is kept only for
    names written in lower case (`gpt-oss`), so a common word is not read as an entity."""

    def __init__(self, nodes: pd.DataFrame, family_patterns: dict[str, str] | None = None,
                 family_exceptions: dict[str, str] | None = None) -> None:
        self.nodes = nodes.set_index("entity_id")
        self._patterns: list[tuple[str, str, re.Pattern, str]] = []
        for row in nodes.itertuples(index=False):
            if row.entity_type not in NAMED_TYPES:
                continue
            names = {part.strip() for part in str(row.canonical_name).split(" / ")} | set(_list(row.aliases))
            for name in sorted(names, key=len, reverse=True):
                pattern = name_pattern(name)
                if pattern is not None:
                    self._patterns.append((row.entity_id, name, pattern, "node_name"))
            if row.is_open_model_family or not pd.isna(row.model_versions):
                for version in _list(row.model_versions):
                    pattern = name_pattern(version)
                    if pattern is not None:
                        self._patterns.append((row.entity_id, version, pattern, "family_version"))
        family_ids = nodes[nodes["is_open_model_family"]].set_index("canonical_name")["entity_id"]
        self._families = [(family_ids[label], label, re.compile(rx), "family_lexicon")
                          for label, rx in (family_patterns or {}).items() if label in family_ids]
        self._exceptions = {family_ids[label]: re.compile(rx)
                            for label, rx in (family_exceptions or {}).items() if label in family_ids}
        self._by_entity: dict[str, list[re.Pattern]] = defaultdict(list)
        for entity_id, _, pattern, _ in self._patterns + self._families:
            self._by_entity[entity_id].append(pattern)

    def mentions(self, entity_id: str, text: str) -> bool:
        """Whether `text` names the entity (any of its names, versions or family pattern)."""
        view = text.translate(MATCH_MAP)
        return any(pattern.search(view) for pattern in self._by_entity.get(entity_id, []))

    def _candidates(self, text: str) -> list[EntityMatch]:
        found = []
        for entity_id, name, pattern, method in self._patterns:
            for hit in pattern.finditer(text):
                surface = hit.group(0)
                if surface.islower() and not name.islower():
                    continue
                found.append(EntityMatch(entity_id, str(self.nodes.at[entity_id, "canonical_name"]),
                                         str(self.nodes.at[entity_id, "entity_type"]), hit.span(), surface, method))
        for entity_id, name, pattern, method in self._families:
            view = self._exceptions[entity_id].sub(lambda m: " " * len(m.group(0)), text) \
                if entity_id in self._exceptions else text
            for hit in pattern.finditer(view):
                found.append(EntityMatch(entity_id, name, "FOUNDATION_MODEL", hit.span(), hit.group(0), method))
        return found

    def link(self, question: str, max_entities: int = 5) -> list[EntityMatch]:
        """Entities of `question`, longest match first; one entity per stretch of text (ties keep the node
        found in more documents)."""
        text = question.translate(MATCH_MAP)
        priority = {"node_name": 0, "family_version": 1, "family_lexicon": 2}
        candidates = sorted(
            self._candidates(text),
            key=lambda m: (-(m.span[1] - m.span[0]), priority[m.method],
                           -int(self.nodes.at[m.entity_id, "document_count"]), m.entity_id))
        taken: list[tuple[int, int]] = []
        linked: list[EntityMatch] = []
        for match in candidates:
            start, end = match.span
            if any(start < t_end and t_start < end for t_start, t_end in taken):
                continue
            if any(m.entity_id == match.entity_id for m in linked):
                taken.append(match.span)
                continue
            taken.append(match.span)
            linked.append(match)
        return sorted(linked, key=lambda m: m.span[0])[:max_entities]
