from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Callable

import pandas as pd

from .linking import EntityLinker

RRF_K = 60  # same constant as the hybrid search of anatoolbox
POOL_SIZE = 100  # text hits re-ranked by the entity ranking

# Question wording -> graph predicates it asks for
INTENTS: dict[str, re.Pattern] = {
    "developed by": re.compile(r"\b(?:who|which (?:company|lab|team|organi[sz]ation))\b.*\b(?:made|make|makes|built|"
                               r"build|develop\w*|creat\w*|train\w*|behind)\b|\bdevelop(?:er|ed by)\b", re.I),
    "released by": re.compile(r"\b(?:who|which company)\b.*\b(?:releas\w*|launch\w*|publish\w*|open[- ]sourc\w*)\b", re.I),
    "announced by": re.compile(r"\b(?:who|which company)\b.*\bannounc\w*\b", re.I),
    "member of family": re.compile(r"\b(?:famil\w*|based on|built on|derived from|version of|base model|variant)\b", re.I),
    "based on": re.compile(r"\b(?:based on|built on|derived from|fine[- ]tuned (?:from|on)|base model)\b", re.I),
    "distilled from": re.compile(r"\b(?:distill\w*|teacher)\b", re.I),
    "works for": re.compile(r"\b(?:work\w* (?:for|at)|who (?:leads|led|heads|runs)|ceo|chief|employ\w*|joined)\b", re.I),
    "founded by": re.compile(r"\b(?:found\w*|co-?found\w*|start\w* the company)\b", re.I),
    "headquartered in": re.compile(r"\b(?:headquarter\w*|based in|where is|where are|which country)\b", re.I),
    "available on": re.compile(r"\b(?:available|access\w*|download\w*|deploy\w*|platform|where can|hosted)\b", re.I),
    "runs on": re.compile(r"\b(?:run\w* on|hardware|gpu\w*|chip\w*|device\w*|laptop\w*|phone\w*)\b", re.I),
    "integrated into": re.compile(r"\b(?:integrat\w*|built into|inside)\b", re.I),
    "supported by": re.compile(r"\b(?:support\w*|partner\w*|backed)\b", re.I),
    "licensed under": re.compile(r"\blicen[cs]\w*\b", re.I),
    "invested in": re.compile(r"\b(?:invest\w*|fund\w*|stake)\b", re.I),
    "acquired by": re.compile(r"\b(?:acqui\w*|bought|buy\w*|owns?)\b", re.I),
    "subsidiary of": re.compile(r"\b(?:subsidiar\w*|owns?|parent)\b", re.I),
    "outperforms": re.compile(r"\b(?:outperform\w*|beat\w*|better than|stronger than|ahead of)\b", re.I),
    "used by": re.compile(r"\b(?:who uses|used by|adopt\w*|customers?)\b", re.I),
    "manufactured by": re.compile(r"\b(?:manufactur\w*|who makes the chips?)\b", re.I),
}
# Competition is not answered from the graph: the relation could not be checked sentence by sentence (KG v2)
COMPETITION = re.compile(r"\b(?:compet\w*|rival\w*)\b", re.I)
_WORD = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_STOP = set("a an and are as at be by can could did do does for from has have how in is it its of on or that the "
            "their them they this to was were what when where which who why will with".split())


def question_intents(question: str) -> list[str]:
    return [predicate for predicate, pattern in INTENTS.items() if pattern.search(question)]


def node_mentions(mentions: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    """Stage 1 entity mentions (`entity_id`, `doc_id`, `sentence_id`, `date`) mapped to the knowledge graph nodes
    through `merged_ids`; mentions of entities that are not graph nodes are dropped."""
    to_node = {}
    for row in nodes.itertuples(index=False):
        for merged in str(row.merged_ids).split("|"):
            to_node[merged.strip()] = row.entity_id
    frame = mentions.assign(node_id=mentions["entity_id"].map(to_node)).dropna(subset=["node_id"])
    return pd.DataFrame({
        "entity_id": frame["node_id"],
        "document_id": frame["doc_id"],
        "text_unit_id": frame["doc_id"] + ":" + frame["sentence_id"].astype(str),
        "published_at": frame["date"].astype(str).str[:10],
    }).drop_duplicates().reset_index(drop=True)


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


def _terms(text: str) -> set[str]:
    return {word for word in _WORD.findall(str(text).lower()) if word not in _STOP}


class GraphTextRetriever:
    """Text retrieval of WP1 with two graph rankings fused into it by reciprocal rank fusion.

    1. **text**: the WP1 search, unchanged (`search(question, **args)`, e.g. hybrid search of the chunks).
    2. **entity**: the first `POOL_SIZE` text hits re-ranked by the number of question entities they mention: in
       the chunk (graph names, aliases and versions) or in the article (graph mention index).
    3. **relation**: the chunks holding the evidence sentence of a verified graph relation of these entities, when
       the relation links two entities of the question or has the predicate the question asks for.

    The relations are returned as graph facts with their article, date and chunk; `facts_in_context` keeps those
    whose chunk is among the passages given to the model, so every fact in a prompt comes with its source sentence.

    Without a linked entity, the result is the text ranking. Competition questions get no relation ranking.
    Lexicon facts (`background`) are returned for inspection but are not evidence: they have no source article."""

    def __init__(self, linker: EntityLinker, assertions: pd.DataFrame, mentions: pd.DataFrame,
                 chunk_records: list[dict[str, Any]], search: Callable[..., dict[str, Any]],
                 background: pd.DataFrame | None = None, max_facts: int = 5) -> None:
        self.linker = linker
        self.assertions = assertions.reset_index(drop=True)
        self.search = search
        self.background = background
        self.max_facts = max_facts
        self._docs_of = mentions.groupby("entity_id")["document_id"].agg(set).to_dict()
        for row in self.assertions.itertuples(index=False):
            self._docs_of.setdefault(row.subject_id, set()).add(row.document_id)
            self._docs_of.setdefault(row.object_id, set()).add(row.document_id)
        self._chunks_of: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._text_of: dict[str, str] = {}
        for record in chunk_records:
            self._chunks_of[str(record["source_id"])].append(record)
            self._text_of[str(record["id"])] = str(record.get("text") or record.get("chunk_text") or "")
        self._labels = linker.nodes["canonical_name"].to_dict()
        self._types = linker.nodes["entity_type"].to_dict()

    # --- rankings ---------------------------------------------------------

    def entity_coverage(self, passage: dict[str, Any], seeds: list[str]) -> int:
        """Number of seeds the passage mentions: in its chunk, or else in its article."""
        in_chunk = sum(self.linker.mentions(seed, self._text_of.get(str(passage["id"]), "")) for seed in seeds)
        in_article = sum(str(passage.get("source_id")) in self._docs_of.get(seed, ()) for seed in seeds)
        return max(in_chunk, in_article)

    def entity_ranking(self, pool: list[dict[str, Any]], seeds: list[str], size: int) -> list[dict[str, Any]]:
        coverage = [(self.entity_coverage(passage, seeds), position, passage) for position, passage in enumerate(pool)]
        return [passage for count, _, passage in sorted(coverage, key=lambda item: (-item[0], item[1]))
                if count > 0][:size]

    def chunk_of(self, document_id: str, sentence: str) -> dict[str, Any] | None:
        """The chunk of `document_id` that contains `sentence` (the first one, as chunks overlap)."""
        key = _normalized(sentence)
        for probe in (key, key[:80], key[-80:]):
            for record in self._chunks_of.get(str(document_id), []):
                if probe and probe in _normalized(record["chunk_text"]):
                    return record
        return None

    def relation_candidates(self, question: str, seeds: list[str]) -> pd.DataFrame:
        if not seeds or COMPETITION.search(question):
            return self.assertions.iloc[0:0]
        seed_set = set(seeds)
        edges = self.assertions[self.assertions["subject_id"].isin(seed_set)
                                | self.assertions["object_id"].isin(seed_set)].copy()
        if edges.empty:
            return edges
        intents = set(question_intents(question))
        q_terms = _terms(question)
        edges["both_seeds"] = edges["subject_id"].isin(seed_set) & edges["object_id"].isin(seed_set)
        edges["intent_match"] = edges["predicate"].isin(intents)
        edges = edges[edges["both_seeds"] | edges["intent_match"]].copy()
        if edges.empty:
            return edges
        overlap = edges["evidence_text"].map(lambda s: len(q_terms & _terms(s)) / max(len(q_terms), 1)).astype(float)
        support = pd.to_numeric(edges.get("support_score"), errors="coerce").fillna(0.5)
        edges["relation_score"] = (2.0 * edges["both_seeds"] + 1.0 * edges["intent_match"] + support + overlap)
        return edges.sort_values(["relation_score", "published_at"], ascending=[False, True])

    @staticmethod
    def _fuse(rankings: list[list[dict[str, Any]]], size: int) -> list[dict[str, Any]]:
        totals: dict[str, float] = defaultdict(float)
        first: dict[str, dict[str, Any]] = {}
        for ranking in rankings:
            for position, passage in enumerate(ranking):
                totals[passage["id"]] += 1.0 / (RRF_K + position + 1)
                first.setdefault(passage["id"], passage)
        order = sorted(totals, key=lambda pid: -totals[pid])[:size]
        return [{**first[pid], "score": round(totals[pid], 6), "rank": rank}
                for rank, pid in enumerate(order, start=1)]

    # --- entry point --------------------------------------------------------

    def retrieve(self, question: str, *, size: int = 30, **search_args: Any) -> dict[str, Any]:
        text = self.search(question, size=max(size, POOL_SIZE), **search_args)
        matches = self.linker.link(question)
        seeds = list(dict.fromkeys(m.entity_id for m in matches))
        rankings = {"text": text["passages"][:size], "entity": [], "relation": []}
        facts: list[dict[str, Any]] = []
        if seeds:
            rankings["entity"] = self.entity_ranking(text["passages"], seeds, size)
        relations = self.relation_candidates(question, seeds)
        seen_chunks: set[str] = set()
        for row in relations.to_dict(orient="records"):
            record = self.chunk_of(row["document_id"], row["evidence_text"])
            if record is None:
                continue
            if record["id"] not in seen_chunks:
                if len(rankings["relation"]) >= size:
                    break
                seen_chunks.add(record["id"])
                rankings["relation"].append({k: v for k, v in record.items() if k not in ("text", "chunk_text")})
            facts.append({
                "subject": self._labels.get(row["subject_id"], row["subject_id"]),
                "relation": row["predicate"],
                "object": self._labels.get(row["object_id"], row["object_id"]),
                "subject_type": self._types.get(row["subject_id"]),
                "object_type": self._types.get(row["object_id"]),
                "source_ids": [row["document_id"]],
                "date": str(row["published_at"])[:10],
                "evidence": row["evidence_text"],
                "support_score": None if pd.isna(row.get("support_score")) else round(float(row["support_score"]), 4),
                "chunk_id": record["id"],
                "relation_id": row["relation_id"],
            })

        route = "text" if not seeds else ("entity+relation" if rankings["relation"] else "entity")
        fused = self._fuse([r for r in rankings.values() if r], size)
        background = []
        if self.background is not None and seeds:
            rows = self.background[self.background["subject_id"].isin(seeds) | self.background["object_id"].isin(seeds)]
            background = [{"subject": self._labels.get(r.subject_id), "relation": r.predicate,
                           "object": self._labels.get(r.object_id), "knowledge_source": r.knowledge_source}
                          for r in rows.itertuples(index=False)]
        return {
            "query": question,
            "corpus": text.get("corpus"),
            "route": route,
            "entities": [m.as_dict() for m in matches],
            "intents": question_intents(question),
            "passages": fused,
            "facts": facts,
            "background": background,
            "rankings": {name: [p["id"] for p in ranking] for name, ranking in rankings.items()},
        }

    def facts_in_context(self, result: dict[str, Any], context_ids: set[str]) -> list[dict[str, Any]]:
        """The graph facts of `result` whose evidence chunk is in the context, best first, each triple once."""
        kept, seen = [], set()
        for fact in result["facts"]:
            triple = (fact["subject"], fact["relation"], fact["object"])
            if fact["chunk_id"] in context_ids and triple not in seen:
                seen.add(triple)
                kept.append(fact)
        return kept[: self.max_facts]
