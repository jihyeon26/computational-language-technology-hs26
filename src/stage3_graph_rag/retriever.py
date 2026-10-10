from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def _normalize(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value).lower()))


def _aliases(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass
    return [part.strip() for part in text.split("|") if part.strip()]


class GraphEvidenceRetriever:
    """Resolve query entities, expand typed graph edges, and return source passages."""

    def __init__(
        self,
        nodes: pd.DataFrame,
        assertions: pd.DataFrame,
        text_units: pd.DataFrame,
        documents: pd.DataFrame,
    ) -> None:
        self.nodes = nodes.copy()
        self.assertions = assertions.copy()
        self.text_units = text_units.copy()
        self.documents = documents.copy()
        self._validate()
        self._labels = self.nodes.set_index("entity_id")["canonical_name"].to_dict()
        self._types = self.nodes.set_index("entity_id")["entity_type"].to_dict()
        self._search_names = self._build_search_names()

    @classmethod
    def from_directories(
        cls, wp3_dir: Path | str, wp2_dir: Path | str
    ) -> "GraphEvidenceRetriever":
        wp3_path = Path(wp3_dir)
        wp2_path = Path(wp2_dir)
        return cls(
            nodes=pd.read_parquet(wp3_path / "nodes.parquet"),
            assertions=pd.read_parquet(wp3_path / "typed_edges_assertions.parquet"),
            text_units=pd.read_parquet(wp2_path / "text_units.parquet"),
            documents=pd.read_parquet(wp2_path / "documents.parquet"),
        )

    def _validate(self) -> None:
        requirements = {
            "nodes": (self.nodes, {"entity_id", "canonical_name", "entity_type"}),
            "assertions": (
                self.assertions,
                {
                    "relation_id", "subject_id", "predicate", "object_id",
                    "document_id", "text_unit_id",
                },
            ),
            "text_units": (self.text_units, {"text_unit_id", "document_id"}),
            "documents": (self.documents, {"document_id"}),
        }
        errors = []
        for name, (frame, required) in requirements.items():
            missing = sorted(required - set(frame.columns))
            if missing:
                errors.append(f"{name}: missing {missing}")
        if errors:
            raise ValueError("; ".join(errors))
        if self.nodes["entity_id"].duplicated().any():
            raise ValueError("nodes: entity_id must be unique")

    def _build_search_names(self) -> dict[str, list[tuple[str, str]]]:
        result: dict[str, list[tuple[str, str]]] = {}
        for row in self.nodes.itertuples(index=False):
            names = [(str(row.canonical_name), "canonical_name")]
            if hasattr(row, "aliases"):
                names.extend((alias, "alias") for alias in _aliases(row.aliases))
            normalized = []
            seen = set()
            for name, source in names:
                key = _normalize(name)
                if key and key not in seen:
                    normalized.append((key, source))
                    seen.add(key)
            result[str(row.entity_id)] = normalized
        return result

    def resolve_entities(self, query: str, max_entities: int = 5) -> list[dict[str, Any]]:
        normalized_query = _normalize(query)
        query_tokens = set(normalized_query.split())
        candidates = []
        for entity_id, names in self._search_names.items():
            best_score = 0.0
            best_method = ""
            best_name = ""
            for name, source in names:
                name_tokens = set(name.split())
                if name == normalized_query:
                    score, method = 1.0, f"exact_{source}"
                elif f" {name} " in f" {normalized_query} ":
                    score = 0.92 + min(len(name_tokens), 8) * 0.005
                    method = f"phrase_{source}"
                else:
                    overlap = len(query_tokens & name_tokens)
                    score = 0.0 if not name_tokens else 0.70 * overlap / len(name_tokens)
                    method = f"token_{source}"
                if score > best_score:
                    best_score, best_method, best_name = score, method, name
            if best_score >= 0.70:
                candidates.append(
                    {
                        "entity_id": entity_id,
                        "canonical_name": self._labels[entity_id],
                        "entity_type": self._types[entity_id],
                        "match_score": round(best_score, 4),
                        "match_method": best_method,
                        "matched_name": best_name,
                    }
                )
        candidates.sort(
            key=lambda row: (-row["match_score"], -len(row["matched_name"]), row["entity_id"])
        )
        if candidates and candidates[0]["match_score"] >= 0.90:
            candidates = [row for row in candidates if row["match_score"] >= 0.90]
        selected = []
        selected_names: list[str] = []
        for candidate in candidates:
            name = candidate["matched_name"]
            if any(name != prior and name in prior for prior in selected_names):
                continue
            selected.append(candidate)
            selected_names.append(name)
            if len(selected) >= max_entities:
                break
        return selected

    def retrieve(
        self,
        query: str,
        *,
        entity_ids: Iterable[str] | None = None,
        max_hops: int = 1,
        predicates: Iterable[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        top_k: int = 10,
    ) -> dict[str, Any]:
        if max_hops < 1 or max_hops > 3:
            raise ValueError("max_hops must be between 1 and 3")
        if top_k < 1:
            raise ValueError("top_k must be positive")

        matched = self.resolve_entities(query)
        seeds = list(dict.fromkeys(str(value) for value in (entity_ids or [])))
        if not seeds:
            seeds = [row["entity_id"] for row in matched]
        known_ids = set(self.nodes["entity_id"].astype(str))
        unknown = sorted(set(seeds) - known_ids)
        if unknown:
            raise ValueError(f"Unknown entity_ids: {unknown}")
        if not seeds:
            return {
                "query": query,
                "matched_entities": [],
                "evidence": [],
                "diagnostics": {"reason": "no_entity_match", "returned_evidence": 0},
            }

        edges = self.assertions.copy()
        if predicates:
            allowed = {str(value) for value in predicates}
            edges = edges[edges["predicate"].isin(allowed)].copy()
        dates = pd.to_datetime(edges.get("published_at"), errors="coerce")
        if start_date:
            edges = edges[dates >= pd.Timestamp(start_date)].copy()
            dates = pd.to_datetime(edges.get("published_at"), errors="coerce")
        if end_date:
            edges = edges[dates <= pd.Timestamp(end_date)].copy()

        frontier = set(seeds)
        visited = set(seeds)
        selected_frames = []
        for hop in range(1, max_hops + 1):
            mask = edges["subject_id"].isin(frontier) | edges["object_id"].isin(frontier)
            found = edges[mask].copy()
            if found.empty:
                break
            found["hop"] = hop
            selected_frames.append(found)
            neighbors = set(found["subject_id"]) | set(found["object_id"])
            frontier = neighbors - visited
            visited.update(neighbors)
            if not frontier:
                break

        if not selected_frames:
            return {
                "query": query,
                "matched_entities": matched,
                "evidence": [],
                "diagnostics": {
                    "reason": "no_matching_assertions",
                    "seed_entity_ids": seeds,
                    "returned_evidence": 0,
                },
            }

        selected = (
            pd.concat(selected_frames, ignore_index=True)
            .sort_values(["hop", "relation_id"])
            .drop_duplicates("relation_id", keep="first")
        )
        selected["subject_label"] = selected["subject_id"].map(self._labels)
        selected["object_label"] = selected["object_id"].map(self._labels)
        selected["subject_type"] = selected["subject_id"].map(self._types)
        selected["object_type"] = selected["object_id"].map(self._types)

        text_columns = ["text_unit_id", "document_id"]
        if "text_unit_text" in self.text_units.columns:
            text_columns.append("text_unit_text")
        text_lookup = self.text_units[text_columns].drop_duplicates("text_unit_id")
        if "text_unit_text" not in text_lookup:
            text_lookup["text_unit_text"] = pd.NA
        text_lookup = text_lookup.rename(columns={"document_id": "text_document_id"})
        selected = selected.merge(text_lookup, on="text_unit_id", how="left")

        document_columns = ["document_id"] + [
            column
            for column in ["title", "url", "domain", "published_at"]
            if column in self.documents.columns
        ]
        document_lookup = self.documents[document_columns].drop_duplicates("document_id")
        document_lookup["document_found"] = True
        document_lookup = document_lookup.rename(
            columns={"published_at": "document_published_at"}
        )
        selected = selected.merge(document_lookup, on="document_id", how="left")
        raw_rule_score = (
            selected["rule_score"]
            if "rule_score" in selected
            else pd.Series(0.0, index=selected.index)
        )
        rule_score = pd.to_numeric(raw_rule_score, errors="coerce").fillna(0.0)
        selected["retrieval_score"] = 2.0 / selected["hop"] + rule_score.clip(0, 1)
        selected["_sort_date"] = pd.to_datetime(selected.get("published_at"), errors="coerce")
        selected = selected.sort_values(
            ["retrieval_score", "hop", "_sort_date", "relation_id"],
            ascending=[False, True, False, True],
        ).head(top_k)

        evidence = []
        for row in selected.to_dict(orient="records"):
            passage = row.get("text_unit_text")
            if pd.isna(passage) or not str(passage).strip():
                passage = row.get("evidence_text", "")
            evidence.append(
                {
                    "relation_id": row["relation_id"],
                    "hop": int(row["hop"]),
                    "subject": {
                        "entity_id": row["subject_id"],
                        "name": row.get("subject_label"),
                        "type": row.get("subject_type"),
                    },
                    "predicate": row["predicate"],
                    "object": {
                        "entity_id": row["object_id"],
                        "name": row.get("object_label"),
                        "type": row.get("object_type"),
                    },
                    "passage": passage,
                    "text_unit_id": row["text_unit_id"],
                    "document_id": row["document_id"],
                    "title": row.get("title"),
                    "url": row.get("url"),
                    "domain": row.get("domain"),
                    "published_at": row.get("document_published_at", row.get("published_at")),
                    "rule_score": (
                        None if pd.isna(row.get("rule_score")) else float(row.get("rule_score"))
                    ),
                    "retrieval_score": float(row["retrieval_score"]),
                }
            )

        missing_text = int(selected["text_unit_text"].isna().sum())
        missing_documents = int(selected["document_found"].isna().sum())
        return {
            "query": query,
            "matched_entities": matched,
            "evidence": evidence,
            "diagnostics": {
                "seed_entity_ids": seeds,
                "candidate_assertions": int(len(pd.concat(selected_frames, ignore_index=True))),
                "returned_evidence": len(evidence),
                "missing_text_units": missing_text,
                "missing_documents": missing_documents,
                "max_hops": max_hops,
                "predicates": sorted(set(predicates)) if predicates else None,
                "start_date": start_date,
                "end_date": end_date,
            },
        }
