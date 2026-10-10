from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import networkx as nx
import numpy as np
import pandas as pd

from .contract import ContractError, find_optional_json, load_contract_tables, read_contract


PIPELINE_VERSION = "stage1-wp3-contract-driven-v0.3"


@dataclass(frozen=True)
class WP3PipelineConfig:
    input_dir: Path
    output_dir: Path
    contract_path: Path


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{digest}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _scalar(value: Any, default: Any = "") -> Any:
    safe = _json_safe(value)
    if safe is None:
        return default
    if isinstance(safe, (list, dict)):
        return json.dumps(safe, ensure_ascii=False)
    return safe


def _ensure_columns(frame: pd.DataFrame, defaults: dict[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    for column, default in defaults.items():
        if column not in result.columns:
            result[column] = default
    return result


def prepare_entities(frame: pd.DataFrame) -> pd.DataFrame:
    entities = _ensure_columns(
        frame,
        {
            "mention_count": 0,
            "document_count": 0,
            "first_seen": pd.NA,
            "last_seen": pd.NA,
        },
    )
    for column in ["entity_id", "canonical_name", "entity_type"]:
        entities[column] = entities[column].astype("string").str.strip()
    for column in ["mention_count", "document_count"]:
        entities[column] = pd.to_numeric(entities[column], errors="coerce").fillna(0).astype(int)
    return entities


def prepare_relations(frame: pd.DataFrame) -> pd.DataFrame:
    relations = _ensure_columns(
        frame,
        {
            "relation_id": pd.NA,
            "document_id": pd.NA,
            "text_unit_id": pd.NA,
            "evidence_text": pd.NA,
            "published_at": pd.NA,
            "assertion_status": "asserted",
            "extraction_method": "upstream_unspecified",
            "rule_score": pd.NA,
        },
    ).reset_index(drop=True)
    for column in ["subject_id", "predicate", "object_id", "assertion_status"]:
        relations[column] = relations[column].astype("string").str.strip()
    missing_id = relations["relation_id"].isna() | relations["relation_id"].astype("string").str.strip().eq("")
    for index in relations.index[missing_id]:
        row = relations.loc[index]
        key_parts = [
            str(row.get(column, ""))
            for column in ["subject_id", "predicate", "object_id", "document_id", "text_unit_id"]
        ]
        key = "|".join([*key_parts, str(index)])
        relations.at[index, "relation_id"] = _stable_id("relation", key)
    relations["relation_id"] = relations["relation_id"].astype("string")
    relations["rule_score"] = pd.to_numeric(relations["rule_score"], errors="coerce")
    return relations


def normalize_relation_schema(schema: dict[str, Any] | None) -> dict[str, Any] | None:
    """Convert the WP2 schema layout into the canonical WP3 signature layout."""
    if schema is None:
        return None
    if schema.get("relation_signatures"):
        return schema

    relation_types = (
        schema.get("edges", {}).get("relation", {}).get("types", {})
    )
    if not relation_types:
        return schema

    normalized = dict(schema)
    normalized["relation_signatures"] = {
        predicate: [
            [head_type, tail_type]
            for head_type in specification.get("head_types", [])
            for tail_type in specification.get("tail_types", [])
        ]
        for predicate, specification in relation_types.items()
    }
    return normalized


def annotate_final_relation_types(
    entities: pd.DataFrame,
    relations: pd.DataFrame,
    schema: dict[str, Any] | None,
) -> pd.DataFrame:
    """Re-evaluate relation signatures using canonical node types, not mention types."""
    result = relations.copy()
    type_by_id = entities.set_index("entity_id")["entity_type"].to_dict()
    result["subject_type_final"] = result["subject_id"].map(type_by_id).astype("string")
    result["object_type_final"] = result["object_id"].map(type_by_id).astype("string")

    signatures = {
        predicate: {tuple(pair) for pair in pairs}
        for predicate, pairs in (schema or {}).get("relation_signatures", {}).items()
    }
    if not signatures:
        result["final_type_valid"] = pd.NA
        return result

    result["final_type_valid"] = [
        (subject_type, object_type) in signatures.get(predicate, set())
        for subject_type, predicate, object_type in result[
            ["subject_type_final", "predicate", "object_type_final"]
        ].itertuples(index=False, name=None)
    ]
    return result


def correct_sentence_local_offsets(
    mentions: pd.DataFrame,
    text_units: pd.DataFrame,
) -> pd.DataFrame:
    """Add corrected sentence-local spans while preserving upstream article offsets."""
    required_mentions = {
        "entity_id", "mention_text", "text_unit_id", "article_start_char", "article_end_char"
    }
    required_text_units = {"text_unit_id", "text", "sentence_start"}
    missing_mentions = required_mentions - set(mentions.columns)
    missing_text_units = required_text_units - set(text_units.columns)
    if missing_mentions or missing_text_units:
        raise ValueError(
            "Offset correction requires canonical mention and text-unit columns; "
            f"missing mentions={sorted(missing_mentions)}, "
            f"missing text_units={sorted(missing_text_units)}"
        )
    if not text_units["text_unit_id"].is_unique:
        raise ValueError("text_unit_id must be unique before correcting mention offsets.")

    sentence_lookup = text_units[
        ["text_unit_id", "text", "sentence_start"]
    ].rename(columns={"text": "sentence_text"})
    result = mentions.merge(
        sentence_lookup,
        on="text_unit_id",
        how="left",
        validate="many_to_one",
    )
    result["sentence_local_start_original"] = (
        pd.to_numeric(result["article_start_char"], errors="coerce")
        - pd.to_numeric(result["sentence_start"], errors="coerce")
    ).astype("Int64")
    result["sentence_local_end_original"] = (
        pd.to_numeric(result["article_end_char"], errors="coerce")
        - pd.to_numeric(result["sentence_start"], errors="coerce")
    ).astype("Int64")

    def reconcile(row: pd.Series) -> tuple[Any, Any, str]:
        sentence = row["sentence_text"]
        mention = row["mention_text"]
        start = row["sentence_local_start_original"]
        end = row["sentence_local_end_original"]
        if not isinstance(sentence, str) or not isinstance(mention, str) or not mention:
            return pd.NA, pd.NA, "unresolved"
        if pd.notna(start) and pd.notna(end):
            start_int, end_int = int(start), int(end)
            if (
                0 <= start_int <= end_int <= len(sentence)
                and sentence[start_int:end_int].casefold() == mention.casefold()
            ):
                return start_int, end_int, "already_valid"

        sentence_folded = sentence.casefold()
        mention_folded = mention.casefold()
        positions: list[int] = []
        position = sentence_folded.find(mention_folded)
        while position >= 0:
            positions.append(position)
            position = sentence_folded.find(mention_folded, position + 1)
        if not positions:
            return pd.NA, pd.NA, "unresolved"

        anchor = int(start) if pd.notna(start) else 0
        corrected_start = min(positions, key=lambda candidate: abs(candidate - anchor))
        status = "corrected_unique" if len(positions) == 1 else "corrected_nearest"
        return corrected_start, corrected_start + len(mention), status

    corrected = result.apply(reconcile, axis=1, result_type="expand")
    corrected.columns = [
        "sentence_local_start", "sentence_local_end", "offset_status"
    ]
    result = pd.concat([result, corrected], axis=1)
    result["offset_resolved"] = result["offset_status"] != "unresolved"
    return result


def apply_compact_schema(
    entities: pd.DataFrame,
    relations: pd.DataFrame,
    compact_schema: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add compact analysis groups without replacing the raw WP2 labels."""
    entity_mapping = compact_schema.get("entity_type_mapping", {})
    predicate_mapping = compact_schema.get("predicate_group_mapping", {})

    compact_entities = entities.copy()
    compact_entities["raw_entity_type"] = compact_entities["entity_type"]
    compact_entities["entity_type_compact"] = compact_entities["raw_entity_type"].map(
        entity_mapping
    )
    missing_entity_types = sorted(
        compact_entities.loc[
            compact_entities["entity_type_compact"].isna(), "raw_entity_type"
        ].dropna().unique()
    )
    if missing_entity_types:
        raise ContractError(
            f"Compact schema does not map entity types: {missing_entity_types}"
        )

    compact_relations = relations.copy()
    compact_relations["raw_predicate"] = compact_relations["predicate"]
    compact_relations["predicate_group"] = compact_relations["raw_predicate"].map(
        predicate_mapping
    )
    missing_predicates = sorted(
        compact_relations.loc[
            compact_relations["predicate_group"].isna(), "raw_predicate"
        ].dropna().unique()
    )
    if missing_predicates:
        raise ContractError(
            f"Compact schema does not map predicates: {missing_predicates}"
        )

    if "subject_type_final" in compact_relations.columns:
        compact_relations["subject_type_compact"] = compact_relations[
            "subject_type_final"
        ].map(entity_mapping)
    if "object_type_final" in compact_relations.columns:
        compact_relations["object_type_compact"] = compact_relations[
            "object_type_final"
        ].map(entity_mapping)

    score = pd.to_numeric(compact_relations.get("rule_score"), errors="coerce")
    method = compact_relations.get(
        "extraction_method",
        pd.Series("upstream_unspecified", index=compact_relations.index),
    )
    compact_relations["quality_tier"] = np.select(
        [
            method.eq("wp1_lexicon"),
            score.ge(0.75),
            score.ge(0.60),
            score.notna(),
        ],
        [
            "curated_without_article_evidence",
            "high_score_candidate",
            "medium_score_candidate",
            "low_score_candidate",
        ],
        default="score_unavailable",
    )
    compact_relations["human_validation_status"] = "unreviewed"
    return compact_entities, compact_relations


def prepare_cooccurrence(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame(
            columns=["source_id", "target_id", "sentence_count", "document_count", "first_seen", "last_seen"]
        )
    result = _ensure_columns(
        frame,
        {"sentence_count": 1, "document_count": 1, "first_seen": pd.NA, "last_seen": pd.NA},
    )
    for column in ["sentence_count", "document_count"]:
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0).astype(int)
    return result


def validate_inputs(
    entities: pd.DataFrame,
    relations: pd.DataFrame,
    cooccurrence: pd.DataFrame,
    schema: dict[str, Any] | None,
    policies: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    entity_ids = set(entities["entity_id"].dropna())

    duplicate_entities = int(entities["entity_id"].duplicated().sum())
    duplicate_relations = int(relations["relation_id"].duplicated().sum())
    null_entity_keys = int(entities[["entity_id", "canonical_name", "entity_type"]].isna().any(axis=1).sum())
    null_relation_keys = int(relations[["subject_id", "predicate", "object_id"]].isna().any(axis=1).sum())
    dangling_subjects = int((~relations["subject_id"].isin(entity_ids)).sum())
    dangling_objects = int((~relations["object_id"].isin(entity_ids)).sum())
    self_relations = int((relations["subject_id"] == relations["object_id"]).sum())

    hard_checks = {
        "duplicate_entity_ids": duplicate_entities,
        "duplicate_relation_ids": duplicate_relations,
        "null_entity_keys": null_entity_keys,
        "null_relation_keys": null_relation_keys,
        "dangling_relation_subjects": dangling_subjects,
        "dangling_relation_objects": dangling_objects,
    }
    for name, count in hard_checks.items():
        if count:
            errors.append(f"{name}={count}")

    invalid_signatures = 0
    unknown_predicates: set[str] = set()
    if schema and schema.get("relation_signatures"):
        type_by_id = entities.set_index("entity_id")["entity_type"].to_dict()
        signatures = {
            predicate: {tuple(pair) for pair in pairs}
            for predicate, pairs in schema["relation_signatures"].items()
        }
        for row in relations.itertuples(index=False):
            if row.predicate not in signatures:
                unknown_predicates.add(str(row.predicate))
                invalid_signatures += 1
                continue
            actual = (type_by_id.get(row.subject_id), type_by_id.get(row.object_id))
            if actual not in signatures[row.predicate]:
                invalid_signatures += 1
        if invalid_signatures and policies.get("strict_schema", True):
            errors.append(f"invalid_relation_signatures={invalid_signatures}")
        elif invalid_signatures:
            warnings.append(f"invalid_relation_signatures={invalid_signatures}")
    else:
        warnings.append("No upstream relation schema found; signature validation was skipped.")

    cooccurrence_dangling = 0
    if not cooccurrence.empty:
        cooccurrence_dangling = int(
            (~cooccurrence["source_id"].isin(entity_ids) | ~cooccurrence["target_id"].isin(entity_ids)).sum()
        )
        if cooccurrence_dangling:
            errors.append(f"dangling_cooccurrence_endpoints={cooccurrence_dangling}")

    if self_relations:
        warnings.append(f"self_relations={self_relations}; policy may remove them.")
    if unknown_predicates:
        warnings.append(f"unknown_predicates={sorted(unknown_predicates)}")

    missing_provenance: dict[str, int] = {}
    for field in policies.get("required_relation_provenance", []):
        if field not in relations.columns:
            missing_provenance[field] = int(len(relations))
        else:
            values = relations[field]
            missing_provenance[field] = int(
                (values.isna() | values.astype("string").str.strip().eq("")).sum()
            )
    provenance_failures = {field: count for field, count in missing_provenance.items() if count}
    if provenance_failures:
        message = f"missing_relation_provenance={provenance_failures}"
        if policies.get("strict_provenance", False):
            errors.append(message)
        else:
            warnings.append(message)

    return {
        **hard_checks,
        "self_relations": self_relations,
        "invalid_relation_signatures": invalid_signatures,
        "dangling_cooccurrence_endpoints": cooccurrence_dangling,
        "missing_relation_provenance": missing_provenance,
        "errors": errors,
        "warnings": warnings,
        "passed": not errors,
    }


def _review_value(value: Any) -> bool | None:
    if pd.isna(value):
        return None
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "correct", "accept", "accepted"}:
        return True
    if normalized in {"false", "0", "no", "n", "incorrect", "reject", "rejected"}:
        return False
    return None


def apply_relation_policy(
    relations: pd.DataFrame,
    reviews: pd.DataFrame | None,
    policies: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, int]]:
    result = relations.copy()
    audit = {"input_relations": int(len(result))}

    if reviews is not None and "review_correct" in reviews.columns:
        review_map = (
            reviews[["relation_id", "review_correct"]]
            .drop_duplicates("relation_id", keep="last")
            .assign(review_correct=lambda frame: frame["review_correct"].map(_review_value))
        )
        result = result.merge(review_map, on="relation_id", how="left")
    else:
        result["review_correct"] = None

    allowed_statuses = policies.get("allowed_assertion_statuses")
    if allowed_statuses:
        before = len(result)
        result = result[result["assertion_status"].isin(allowed_statuses)].copy()
        audit["excluded_by_assertion_status"] = int(before - len(result))

    if policies.get("exclude_reviewed_false", True):
        before = len(result)
        result = result[result["review_correct"] != False].copy()  # noqa: E712
        audit["excluded_by_negative_review"] = int(before - len(result))

    if policies.get("require_reviewed_true", False):
        before = len(result)
        result = result[result["review_correct"] == True].copy()  # noqa: E712
        audit["excluded_as_unreviewed"] = int(before - len(result))

    if policies.get("drop_self_relations", True):
        before = len(result)
        result = result[result["subject_id"] != result["object_id"]].copy()
        audit["excluded_self_relations"] = int(before - len(result))

    if policies.get("require_valid_final_signature", False):
        if "final_type_valid" not in result.columns:
            raise ContractError(
                "Final relation types must be annotated before applying the relation policy."
            )
        if result["final_type_valid"].isna().all():
            raise ContractError(
                "Final relation signature filtering was requested, but no canonical signatures were loaded."
            )
        before = len(result)
        result = result[result["final_type_valid"].fillna(False)].copy()
        audit["excluded_by_final_signature"] = int(before - len(result))

    minimum_score = policies.get("minimum_rule_score")
    if minimum_score is not None:
        before = len(result)
        result = result[result["rule_score"].fillna(-np.inf) >= float(minimum_score)].copy()
        audit["excluded_by_minimum_score"] = int(before - len(result))

    audit["included_relations"] = int(len(result))
    return result.reset_index(drop=True), audit


def collapse_relations(
    relations: pd.DataFrame,
    documents: pd.DataFrame | None,
) -> pd.DataFrame:
    if relations.empty:
        return pd.DataFrame(
            columns=[
                "edge_id", "subject_id", "predicate", "object_id", "evidence_count",
                "document_count", "domain_count", "first_seen", "last_seen",
                "mean_rule_score", "max_rule_score", "relation_ids", "document_ids", "text_unit_ids",
            ]
        )

    domain_by_document: dict[Any, Any] = {}
    if documents is not None and "domain" in documents.columns:
        domain_by_document = documents.drop_duplicates("document_id").set_index("document_id")["domain"].to_dict()

    records: list[dict[str, Any]] = []
    for (subject_id, predicate, object_id), group in relations.groupby(
        ["subject_id", "predicate", "object_id"], sort=True, dropna=False
    ):
        document_ids = sorted({str(value) for value in group["document_id"].dropna()})
        text_unit_ids = sorted({str(value) for value in group["text_unit_id"].dropna()})
        relation_ids = sorted({str(value) for value in group["relation_id"].dropna()})
        domains = sorted(
            {
                str(domain_by_document[document_id])
                for document_id in document_ids
                if document_id in domain_by_document and pd.notna(domain_by_document[document_id])
            }
        )
        dates = pd.to_datetime(group["published_at"], errors="coerce").dropna()
        scores = pd.to_numeric(group["rule_score"], errors="coerce").dropna()
        status_counts = group["assertion_status"].value_counts().to_dict()
        key = f"{subject_id}|{predicate}|{object_id}"
        records.append(
            {
                "edge_id": _stable_id("edge", key),
                "subject_id": subject_id,
                "predicate": predicate,
                "object_id": object_id,
                "evidence_count": int(len(group)),
                "document_count": int(len(document_ids)),
                "domain_count": int(len(domains)),
                "first_seen": dates.min().strftime("%Y-%m-%d") if not dates.empty else None,
                "last_seen": dates.max().strftime("%Y-%m-%d") if not dates.empty else None,
                "asserted_count": int(status_counts.get("asserted", 0)),
                "negated_count": int(status_counts.get("negated", 0)),
                "uncertain_count": int(status_counts.get("uncertain", 0)),
                "mean_rule_score": float(scores.mean()) if not scores.empty else None,
                "max_rule_score": float(scores.max()) if not scores.empty else None,
                "relation_ids": json.dumps(relation_ids, ensure_ascii=False),
                "document_ids": json.dumps(document_ids, ensure_ascii=False),
                "text_unit_ids": json.dumps(text_unit_ids, ensure_ascii=False),
                "domains": json.dumps(domains, ensure_ascii=False),
            }
        )
    return pd.DataFrame(records).sort_values(
        ["predicate", "subject_id", "object_id"]
    ).reset_index(drop=True)


def _node_attributes(row: Any) -> dict[str, Any]:
    return {
        "label": _scalar(row.canonical_name),
        "entity_type": _scalar(row.entity_type),
        "mention_count": int(getattr(row, "mention_count", 0)),
        "document_count": int(getattr(row, "document_count", 0)),
        "first_seen": _scalar(getattr(row, "first_seen", "")),
        "last_seen": _scalar(getattr(row, "last_seen", "")),
    }


def _add_referenced_nodes(graph: nx.Graph, entities: pd.DataFrame, referenced: set[str]) -> None:
    for row in entities[entities["entity_id"].isin(referenced)].itertuples(index=False):
        graph.add_node(row.entity_id, **_node_attributes(row))


def build_graphs(
    entities: pd.DataFrame,
    relations: pd.DataFrame,
    collapsed: pd.DataFrame,
    cooccurrence: pd.DataFrame,
    policies: dict[str, Any],
) -> dict[str, nx.Graph]:
    include_isolates = bool(policies.get("include_isolated_nodes", False))
    all_ids = set(entities["entity_id"])

    assertion_graph = nx.MultiDiGraph(
        graph_view="assertion", pipeline_version=PIPELINE_VERSION
    )
    assertion_refs = set(relations["subject_id"]) | set(relations["object_id"])
    _add_referenced_nodes(assertion_graph, entities, all_ids if include_isolates else assertion_refs)
    for row in relations.itertuples(index=False):
        attributes = {
            "predicate": _scalar(row.predicate),
            "document_id": _scalar(row.document_id),
            "text_unit_id": _scalar(row.text_unit_id),
            "published_at": _scalar(row.published_at),
            "assertion_status": _scalar(row.assertion_status),
            "extraction_method": _scalar(row.extraction_method),
            "rule_score": float(row.rule_score) if pd.notna(row.rule_score) else -1.0,
        }
        assertion_graph.add_edge(row.subject_id, row.object_id, key=row.relation_id, **attributes)

    typed_graph = nx.MultiDiGraph(
        graph_view="collapsed_typed", pipeline_version=PIPELINE_VERSION
    )
    typed_refs = set(collapsed["subject_id"]) | set(collapsed["object_id"])
    _add_referenced_nodes(typed_graph, entities, all_ids if include_isolates else typed_refs)
    for row in collapsed.itertuples(index=False):
        attributes = {
            column: _scalar(getattr(row, column))
            for column in collapsed.columns
            if column not in {"edge_id", "subject_id", "object_id"}
        }
        typed_graph.add_edge(row.subject_id, row.object_id, key=row.predicate, **attributes)

    cooccurrence_graph = nx.Graph(
        graph_view="cooccurrence", pipeline_version=PIPELINE_VERSION
    )
    minimum_documents = int(policies.get("minimum_cooccurrence_documents", 1))
    filtered_cooccurrence = cooccurrence[
        cooccurrence["document_count"] >= minimum_documents
    ].copy()
    co_refs = set(filtered_cooccurrence["source_id"]) | set(filtered_cooccurrence["target_id"])
    _add_referenced_nodes(cooccurrence_graph, entities, all_ids if include_isolates else co_refs)
    for row in filtered_cooccurrence.itertuples(index=False):
        cooccurrence_graph.add_edge(
            row.source_id,
            row.target_id,
            sentence_count=int(row.sentence_count),
            document_count=int(row.document_count),
            first_seen=_scalar(row.first_seen),
            last_seen=_scalar(row.last_seen),
        )
    return {
        "assertion": assertion_graph,
        "typed": typed_graph,
        "cooccurrence": cooccurrence_graph,
    }


def _simple_weighted_projection(graph: nx.Graph, weight: str) -> nx.Graph:
    simple = nx.Graph()
    simple.add_nodes_from(graph.nodes(data=True))
    for source, target, data in graph.edges(data=True):
        value = float(data.get(weight, 1) or 1)
        if simple.has_edge(source, target):
            simple[source][target][weight] += value
        else:
            simple.add_edge(source, target, **{weight: value})
    for _, _, data in simple.edges(data=True):
        data["distance"] = 1.0 / max(float(data.get(weight, 1)), 1e-9)
    return simple


def calculate_node_metrics(
    graph: nx.Graph,
    graph_view: str,
    weight: str,
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    if graph.number_of_nodes() == 0:
        return pd.DataFrame()

    simple = _simple_weighted_projection(graph, weight)
    pagerank = nx.pagerank(simple, weight=weight) if simple.number_of_edges() else {node: 0.0 for node in simple}
    if simple.number_of_nodes() > 2 and simple.number_of_edges():
        k = min(sample_size, simple.number_of_nodes())
        betweenness = nx.betweenness_centrality(
            simple,
            k=None if k == simple.number_of_nodes() else k,
            normalized=True,
            weight="distance",
            seed=seed,
        )
        clustering = nx.clustering(simple, weight=weight)
    else:
        betweenness = {node: 0.0 for node in simple}
        clustering = {node: 0.0 for node in simple}

    weighted_degree = dict(simple.degree(weight=weight))
    records = []
    for node, data in graph.nodes(data=True):
        record = {
            "graph_view": graph_view,
            "entity_id": node,
            "canonical_name": data.get("label", node),
            "entity_type": data.get("entity_type", "UNKNOWN"),
            "degree": int(graph.degree(node)),
            "weighted_degree": float(weighted_degree.get(node, 0.0)),
            "pagerank": float(pagerank.get(node, 0.0)),
            "betweenness": float(betweenness.get(node, 0.0)),
            "clustering": float(clustering.get(node, 0.0)),
        }
        if graph.is_directed():
            record["in_degree"] = int(graph.in_degree(node))
            record["out_degree"] = int(graph.out_degree(node))
        records.append(record)
    return pd.DataFrame(records).sort_values(
        ["pagerank", "weighted_degree", "entity_id"], ascending=[False, False, True]
    ).reset_index(drop=True)


def calculate_relation_type_metrics(
    entities: pd.DataFrame,
    collapsed: pd.DataFrame,
    sample_size: int,
    seed: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if collapsed.empty:
        return pd.DataFrame()
    for predicate, edges in collapsed.groupby("predicate"):
        graph = nx.DiGraph(predicate=predicate)
        referenced = set(edges["subject_id"]) | set(edges["object_id"])
        _add_referenced_nodes(graph, entities, referenced)
        for row in edges.itertuples(index=False):
            graph.add_edge(
                row.subject_id,
                row.object_id,
                document_count=int(row.document_count),
                evidence_count=int(row.evidence_count),
            )
        metrics = calculate_node_metrics(
            graph,
            f"predicate:{predicate}",
            "document_count",
            sample_size,
            seed,
        )
        if not metrics.empty:
            metrics.insert(1, "predicate", predicate)
            frames.append(metrics)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_relation_type_summary(collapsed: pd.DataFrame) -> pd.DataFrame:
    if collapsed.empty:
        return pd.DataFrame(
            columns=[
                "predicate", "unique_edges", "subject_nodes", "object_nodes",
                "total_evidence", "total_documents", "first_seen", "last_seen",
            ]
        )
    records = []
    for predicate, group in collapsed.groupby("predicate"):
        first_dates = pd.to_datetime(group["first_seen"], errors="coerce").dropna()
        last_dates = pd.to_datetime(group["last_seen"], errors="coerce").dropna()
        records.append(
            {
                "predicate": predicate,
                "unique_edges": int(len(group)),
                "subject_nodes": int(group["subject_id"].nunique()),
                "object_nodes": int(group["object_id"].nunique()),
                "total_evidence": int(group["evidence_count"].sum()),
                "total_documents": int(group["document_count"].sum()),
                "first_seen": first_dates.min().strftime("%Y-%m-%d") if not first_dates.empty else None,
                "last_seen": last_dates.max().strftime("%Y-%m-%d") if not last_dates.empty else None,
            }
        )
    return pd.DataFrame(records).sort_values(
        ["unique_edges", "predicate"], ascending=[False, True]
    ).reset_index(drop=True)


def build_temporal_tables(
    entities: pd.DataFrame,
    relations: pd.DataFrame,
    collapsed: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    relation_rows = relations.copy()
    relation_rows["month"] = pd.to_datetime(
        relation_rows["published_at"], errors="coerce"
    ).dt.to_period("M").astype("string")
    monthly_assertions = (
        relation_rows.dropna(subset=["month"])
        .groupby(["month", "predicate"], as_index=False)
        .agg(assertion_count=("relation_id", "count"), document_count=("document_id", "nunique"))
        .sort_values(["month", "predicate"])
        .reset_index(drop=True)
    )

    entity_rows = entities.copy()
    entity_rows["month"] = pd.to_datetime(
        entity_rows["first_seen"], errors="coerce"
    ).dt.to_period("M").astype("string")
    monthly_entities = (
        entity_rows.dropna(subset=["month"])
        .groupby(["month", "entity_type"], as_index=False)
        .agg(new_entity_count=("entity_id", "nunique"))
        .sort_values(["month", "entity_type"])
        .reset_index(drop=True)
    )

    edge_rows = collapsed.copy()
    edge_rows["month"] = pd.to_datetime(
        edge_rows["first_seen"], errors="coerce"
    ).dt.to_period("M").astype("string")
    monthly_edges = (
        edge_rows.dropna(subset=["month"])
        .groupby(["month", "predicate"], as_index=False)
        .agg(new_typed_edge_count=("edge_id", "nunique"))
        .sort_values(["month", "predicate"])
        .reset_index(drop=True)
    )
    return {
        "monthly_relation_assertions": monthly_assertions,
        "monthly_new_entities": monthly_entities,
        "monthly_new_typed_edges": monthly_edges,
    }


def detect_cooccurrence_communities(graph: nx.Graph) -> tuple[pd.DataFrame, pd.DataFrame]:
    if graph.number_of_nodes() == 0:
        return pd.DataFrame(), pd.DataFrame()
    communities = list(
        nx.community.greedy_modularity_communities(graph, weight="document_count")
    )
    communities = sorted(communities, key=lambda members: (-len(members), sorted(members)[0]))
    node_records = []
    summary_records = []
    for community_id, members in enumerate(communities, start=1):
        subgraph = graph.subgraph(members)
        weighted_degree = dict(subgraph.degree(weight="document_count"))
        ranked = sorted(members, key=lambda node: (-weighted_degree.get(node, 0), node))
        for rank, node in enumerate(ranked, start=1):
            data = graph.nodes[node]
            node_records.append(
                {
                    "community_id": community_id,
                    "community_size": len(members),
                    "rank_in_community": rank,
                    "entity_id": node,
                    "canonical_name": data.get("label", node),
                    "entity_type": data.get("entity_type", "UNKNOWN"),
                    "weighted_degree_in_community": float(weighted_degree.get(node, 0)),
                }
            )
        summary_records.append(
            {
                "community_id": community_id,
                "node_count": len(members),
                "edge_count": int(subgraph.number_of_edges()),
                "top_entities": json.dumps(
                    [graph.nodes[node].get("label", node) for node in ranked[:10]],
                    ensure_ascii=False,
                ),
            }
        )
    return pd.DataFrame(node_records), pd.DataFrame(summary_records)


def build_finding_candidates(
    entities: pd.DataFrame,
    collapsed: pd.DataFrame,
    cooccurrence: pd.DataFrame,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    labels = entities.set_index("entity_id")["canonical_name"].to_dict()
    records: list[dict[str, Any]] = []

    for view in ["typed", "cooccurrence"]:
        ranked = metrics[metrics["graph_view"] == view].head(10)
        for rank, row in enumerate(ranked.itertuples(index=False), start=1):
            records.append(
                {
                    "candidate_type": f"top_pagerank_{view}",
                    "rank": rank,
                    "entity_or_pair": row.canonical_name,
                    "predicate": None,
                    "metric_name": "pagerank",
                    "metric_value": float(row.pagerank),
                    "document_count": None,
                    "evidence_count": None,
                    "first_seen": None,
                    "last_seen": None,
                    "supporting_ids": json.dumps([row.entity_id]),
                    "review_status": "unreviewed",
                }
            )

    top_edges = collapsed.sort_values(
        ["document_count", "evidence_count"], ascending=False
    ).head(15)
    for rank, row in enumerate(top_edges.itertuples(index=False), start=1):
        records.append(
            {
                "candidate_type": "strong_typed_edge",
                "rank": rank,
                "entity_or_pair": f"{labels.get(row.subject_id, row.subject_id)} -> {labels.get(row.object_id, row.object_id)}",
                "predicate": row.predicate,
                "metric_name": "document_count",
                "metric_value": float(row.document_count),
                "document_count": int(row.document_count),
                "evidence_count": int(row.evidence_count),
                "first_seen": row.first_seen,
                "last_seen": row.last_seen,
                "supporting_ids": row.text_unit_ids,
                "review_status": "unreviewed",
            }
        )

    top_cooccurrence = cooccurrence.sort_values(
        ["document_count", "sentence_count"], ascending=False
    ).head(15)
    for rank, row in enumerate(top_cooccurrence.itertuples(index=False), start=1):
        records.append(
            {
                "candidate_type": "strong_cooccurrence",
                "rank": rank,
                "entity_or_pair": f"{labels.get(row.source_id, row.source_id)} -- {labels.get(row.target_id, row.target_id)}",
                "predicate": "CO_OCCURS_WITH",
                "metric_name": "document_count",
                "metric_value": float(row.document_count),
                "document_count": int(row.document_count),
                "evidence_count": int(row.sentence_count),
                "first_seen": row.first_seen,
                "last_seen": row.last_seen,
                "supporting_ids": None,
                "review_status": "unreviewed",
            }
        )
    return pd.DataFrame(records)


def graph_summary(graph: nx.Graph) -> dict[str, Any]:
    if graph.is_directed():
        simple: nx.Graph = nx.DiGraph()
    else:
        simple = nx.Graph()
    simple.add_nodes_from(graph.nodes())
    simple.add_edges_from((source, target) for source, target in graph.edges())
    if graph.is_directed():
        components = nx.number_weakly_connected_components(graph) if graph.number_of_nodes() else 0
        component_name = "weakly_connected_components"
    else:
        components = nx.number_connected_components(graph) if graph.number_of_nodes() else 0
        component_name = "connected_components"
    return {
        "nodes": int(graph.number_of_nodes()),
        "edges": int(graph.number_of_edges()),
        "unique_node_pairs": int(simple.number_of_edges()),
        "simple_density": float(nx.density(simple)) if simple.number_of_nodes() > 1 else 0.0,
        component_name: int(components),
        "isolated_nodes": int(sum(1 for node in graph if graph.degree(node) == 0)),
    }


ENTITY_COLORS = {
    "MODEL": "#4C78A8",
    "MODEL_FAMILY": "#72B7B2",
    "ORGANIZATION": "#F58518",
    "LICENSE": "#E45756",
    "ARCHITECTURE": "#54A24B",
    "CAPABILITY": "#B279A2",
    "PERSON": "#9D755D",
    "PRODUCT_OTHER": "#BAB0AC",
}


def _draw_graph(
    graph: nx.Graph,
    metrics: pd.DataFrame,
    output_path: Path,
    title: str,
    weight: str,
    maximum_nodes: int,
    maximum_labels: int,
    seed: int,
) -> None:
    if graph.number_of_nodes() == 0 or metrics.empty:
        return
    selected = metrics.head(maximum_nodes)["entity_id"].tolist()
    view = graph.subgraph(selected).copy()
    if view.number_of_edges() == 0:
        return
    simple = _simple_weighted_projection(view, weight)
    simple.remove_nodes_from(list(nx.isolates(simple)))
    if simple.number_of_nodes() == 0:
        return
    spring_distance = 2.2 / np.sqrt(simple.number_of_nodes())
    positions = nx.spring_layout(
        simple, seed=seed, weight=weight, k=spring_distance, iterations=250
    )
    node_sizes = [350 + 3200 * metrics.set_index("entity_id").loc[node, "pagerank"] for node in simple.nodes]
    node_colors = [ENTITY_COLORS.get(simple.nodes[node].get("entity_type", ""), "#BAB0AC") for node in simple.nodes]
    edge_weights = [float(data.get(weight, 1)) for _, _, data in simple.edges(data=True)]
    max_weight = max(edge_weights) if edge_weights else 1
    widths = [0.5 + 3.0 * value / max_weight for value in edge_weights]
    labels_to_show = set(metrics.head(maximum_labels)["entity_id"])
    labels = {
        node: simple.nodes[node].get("label", node)
        for node in simple.nodes
        if node in labels_to_show
    }

    fig, ax = plt.subplots(figsize=(13, 9))
    nx.draw_networkx_edges(simple, positions, width=widths, alpha=0.30, edge_color="#6B7280", ax=ax)
    nx.draw_networkx_nodes(
        simple,
        positions,
        node_size=node_sizes,
        node_color=node_colors,
        linewidths=0.7,
        edgecolors="white",
        ax=ax,
    )
    nx.draw_networkx_labels(simple, positions, labels=labels, font_size=8, ax=ax)
    present_types = sorted({simple.nodes[node].get("entity_type", "UNKNOWN") for node in simple.nodes})
    legend = [
        Patch(facecolor=ENTITY_COLORS.get(entity_type, "#BAB0AC"), label=entity_type)
        for entity_type in present_types
    ]
    ax.legend(handles=legend, loc="upper left", frameon=False, fontsize=8)
    ax.set_title(title, fontsize=15)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


RELATION_COLORS = {
    "DEVELOPED_BY": "#4C78A8",
    "RELEASED_BY": "#F58518",
    "LICENSED_UNDER": "#E45756",
    "USES_ARCHITECTURE": "#54A24B",
    "HAS_CAPABILITY": "#B279A2",
}


def _draw_relation_subgraph(
    collapsed: pd.DataFrame,
    entities: pd.DataFrame,
    predicates: list[str],
    output_path: Path,
    title: str,
    maximum_edges: int,
    seed: int,
) -> None:
    selected = collapsed[collapsed["predicate"].isin(predicates)].copy()
    if selected.empty:
        return
    selected = selected.sort_values(
        ["document_count", "evidence_count", "edge_id"], ascending=[False, False, True]
    ).head(maximum_edges)
    labels = entities.set_index("entity_id")["canonical_name"].to_dict()
    types = entities.set_index("entity_id")["entity_type"].to_dict()
    graph = nx.MultiDiGraph()
    for row in selected.itertuples(index=False):
        graph.add_node(
            row.subject_id,
            label=labels.get(row.subject_id, row.subject_id),
            entity_type=types.get(row.subject_id, "UNKNOWN"),
        )
        graph.add_node(
            row.object_id,
            label=labels.get(row.object_id, row.object_id),
            entity_type=types.get(row.object_id, "UNKNOWN"),
        )
        graph.add_edge(
            row.subject_id,
            row.object_id,
            key=row.predicate,
            predicate=row.predicate,
            document_count=int(row.document_count),
        )
    positions = nx.spring_layout(graph, seed=seed, weight="document_count", k=1.4)
    fig, ax = plt.subplots(figsize=(13, 9))
    node_colors = [
        ENTITY_COLORS.get(graph.nodes[node].get("entity_type", "UNKNOWN"), "#BAB0AC")
        for node in graph.nodes
    ]
    node_sizes = [520 + 70 * graph.degree(node) for node in graph.nodes]
    nx.draw_networkx_nodes(
        graph, positions, node_color=node_colors, node_size=node_sizes,
        edgecolors="white", linewidths=0.7, ax=ax,
    )
    for predicate in predicates:
        edges = [
            (source, target, key)
            for source, target, key, data in graph.edges(keys=True, data=True)
            if data.get("predicate") == predicate
        ]
        if not edges:
            continue
        widths = [
            0.8 + 0.8 * graph[source][target][key].get("document_count", 1)
            for source, target, key in edges
        ]
        nx.draw_networkx_edges(
            graph, positions, edgelist=edges, width=widths,
            edge_color=RELATION_COLORS.get(predicate, "#6B7280"), alpha=0.60,
            arrows=True, arrowsize=15, connectionstyle="arc3,rad=0.08", ax=ax,
        )
    nx.draw_networkx_labels(
        graph,
        positions,
        labels={node: graph.nodes[node]["label"] for node in graph.nodes},
        font_size=8,
        ax=ax,
    )
    predicate_legend = [
        Patch(facecolor=RELATION_COLORS.get(predicate, "#6B7280"), label=predicate)
        for predicate in predicates
        if predicate in set(selected["predicate"])
    ]
    ax.legend(handles=predicate_legend, loc="upper left", frameon=False, fontsize=9)
    ax.set_title(title, fontsize=15)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def create_visualizations(
    graphs: dict[str, nx.Graph],
    metrics: pd.DataFrame,
    entities: pd.DataFrame,
    collapsed: pd.DataFrame,
    relation_summary: pd.DataFrame,
    temporal_tables: dict[str, pd.DataFrame],
    community_summary: pd.DataFrame,
    output_dir: Path,
    analysis: dict[str, Any],
) -> list[str]:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    maximum_nodes = int(analysis.get("maximum_visual_nodes", 40))
    maximum_labels = int(analysis.get("maximum_labels", 18))
    seed = int(analysis.get("random_seed", 42))
    created: list[str] = []

    for view, graph, weight, title, filename in [
        ("typed", graphs["typed"], "document_count", "Open Models - collapsed typed graph", "typed_graph_top.png"),
        ("cooccurrence", graphs["cooccurrence"], "document_count", "Open Models - co-occurrence graph", "cooccurrence_graph_top.png"),
    ]:
        path = figure_dir / filename
        _draw_graph(
            graph,
            metrics[metrics["graph_view"] == view],
            path,
            title,
            weight,
            maximum_nodes,
            maximum_labels,
            seed,
        )
        if path.exists():
            created.append(str(path))

    typed_metrics = metrics[metrics["graph_view"] == "typed"].head(15).sort_values("pagerank")
    if not typed_metrics.empty:
        path = figure_dir / "typed_pagerank_top15.png"
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(typed_metrics["canonical_name"], typed_metrics["pagerank"], color="#4C78A8")
        ax.set(title="Top PageRank - collapsed typed graph", xlabel="PageRank", ylabel="")
        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        created.append(str(path))

    for predicates, title, filename in [
        (["DEVELOPED_BY", "RELEASED_BY"], "Developer and release relations", "developer_release_subgraph.png"),
        (["LICENSED_UNDER", "USES_ARCHITECTURE", "HAS_CAPABILITY"], "Model attributes", "model_attributes_subgraph.png"),
    ]:
        path = figure_dir / filename
        _draw_relation_subgraph(
            collapsed, entities, predicates, path, title,
            int(analysis.get("maximum_relation_visual_edges", 35)), seed,
        )
        if path.exists():
            created.append(str(path))

    if not relation_summary.empty:
        path = figure_dir / "relation_type_counts.png"
        plot_data = relation_summary.sort_values("unique_edges")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(plot_data["predicate"], plot_data["unique_edges"], color="#4C78A8")
        ax.set(title="Unique typed edges by relation", xlabel="Unique edges", ylabel="")
        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        created.append(str(path))

    temporal = temporal_tables["monthly_relation_assertions"]
    if not temporal.empty:
        path = figure_dir / "monthly_relation_assertions.png"
        pivot = temporal.pivot(index="month", columns="predicate", values="assertion_count").fillna(0)
        fig, ax = plt.subplots(figsize=(12, 6))
        pivot.plot(ax=ax, marker="o", linewidth=1.6)
        ax.set(title="Monthly relation assertions", xlabel="Month", ylabel="Assertions")
        ax.legend(title="Predicate", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        created.append(str(path))

    if not community_summary.empty:
        path = figure_dir / "cooccurrence_community_sizes.png"
        plot_data = community_summary.head(20).sort_values("node_count")
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(
            plot_data["community_id"].astype(str), plot_data["node_count"], color="#72B7B2"
        )
        ax.set(title="Largest co-occurrence communities", xlabel="Nodes", ylabel="Community")
        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        created.append(str(path))
    return created


def _graphml_safe_copy(graph: nx.Graph) -> nx.Graph:
    safe = graph.__class__()
    safe.graph.update({key: _scalar(value) for key, value in graph.graph.items()})
    for node, data in graph.nodes(data=True):
        safe.add_node(node, **{key: _scalar(value) for key, value in data.items()})
    if graph.is_multigraph():
        for source, target, key, data in graph.edges(keys=True, data=True):
            safe.add_edge(source, target, key=key, **{name: _scalar(value) for name, value in data.items()})
    else:
        for source, target, data in graph.edges(data=True):
            safe.add_edge(source, target, **{name: _scalar(value) for name, value in data.items()})
    return safe


def _write_graph(graph: nx.Graph, output_dir: Path, stem: str) -> None:
    nx.write_graphml(_graphml_safe_copy(graph), output_dir / f"{stem}.graphml")
    node_link = nx.node_link_data(graph, edges="edges")
    (output_dir / f"{stem}.json").write_text(
        json.dumps(_json_safe(node_link), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_graph_schema(
    contract: dict[str, Any], upstream_schema: dict[str, Any] | None
) -> dict[str, Any]:
    return {
        "schema_version": "stage1-wp3-graph-schema-v1",
        "pipeline_version": PIPELINE_VERSION,
        "upstream_contract_version": contract["contract_version"],
        "node_key": "entity_id",
        "node_attributes": [
            "canonical_name", "entity_type", "mention_count", "document_count",
            "first_seen", "last_seen",
        ],
        "graph_views": {
            "assertion": {
                "directed": True,
                "multigraph": True,
                "edge_key": "relation_id",
                "purpose": "Evidence-preserving assertions for provenance and Graph-RAG.",
                "edge_attributes": [
                    "predicate", "document_id", "text_unit_id", "published_at",
                    "assertion_status", "extraction_method", "rule_score",
                ],
            },
            "collapsed_typed": {
                "directed": True,
                "multigraph": True,
                "edge_key": "predicate",
                "purpose": "Unique subject-predicate-object facts for analysis.",
                "edge_attributes": [
                    "evidence_count", "document_count", "domain_count", "first_seen",
                    "last_seen", "relation_ids", "document_ids", "text_unit_ids",
                ],
            },
            "cooccurrence": {
                "directed": False,
                "multigraph": False,
                "purpose": "Weighted entity co-mentions for community analysis.",
                "edge_attributes": [
                    "sentence_count", "document_count", "first_seen", "last_seen",
                ],
            },
        },
        "relation_signatures": (upstream_schema or {}).get("relation_signatures", {}),
        "policies": contract.get("policies", {}),
        "provenance_chain": [
            "entity_id", "relation_id", "text_unit_id", "document_id", "url", "published_at"
        ],
    }


def export_outputs(
    config: WP3PipelineConfig,
    contract: dict[str, Any],
    contract_report: dict[str, Any],
    validation_report: dict[str, Any],
    policy_audit: dict[str, int],
    entities: pd.DataFrame,
    relations: pd.DataFrame,
    collapsed: pd.DataFrame,
    cooccurrence: pd.DataFrame,
    corrected_mentions: pd.DataFrame | None,
    graphs: dict[str, nx.Graph],
    metrics: pd.DataFrame,
    relation_summary: pd.DataFrame,
    temporal_tables: dict[str, pd.DataFrame],
    community_membership: pd.DataFrame,
    community_summary: pd.DataFrame,
    finding_candidates: pd.DataFrame,
    figures: list[str],
    schema_path: Path | None,
    upstream_schema: dict[str, Any] | None,
) -> dict[str, Any]:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "nodes": entities,
        "typed_edges_assertions": relations,
        "typed_edges_collapsed": collapsed,
        "cooccurrence_edges": cooccurrence,
        "node_metrics": metrics,
        "relation_type_summary": relation_summary,
        "node_communities": community_membership,
        "community_summary": community_summary,
        "graph_finding_candidates": finding_candidates,
        **temporal_tables,
    }
    if corrected_mentions is not None:
        tables["entity_mentions_corrected"] = corrected_mentions
    for name, table in tables.items():
        table.to_parquet(output_dir / f"{name}.parquet", index=False)
        table.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    _write_graph(graphs["assertion"], output_dir, "assertion_graph")
    _write_graph(graphs["typed"], output_dir, "typed_graph_collapsed")
    _write_graph(graphs["cooccurrence"], output_dir, "cooccurrence_graph")

    summaries = {name: graph_summary(graph) for name, graph in graphs.items()}
    summary = {
        "pipeline_version": PIPELINE_VERSION,
        "contract_version": contract["contract_version"],
        "input_entities": int(len(entities)),
        "included_relation_assertions": int(len(relations)),
        "collapsed_typed_edges": int(len(collapsed)),
        "cooccurrence_edges": int(len(cooccurrence)),
        "relation_types": int(relation_summary["predicate"].nunique()) if not relation_summary.empty else 0,
        "cooccurrence_communities": int(len(community_summary)),
        "finding_candidates": int(len(finding_candidates)),
        "graphs": summaries,
        "policy_audit": policy_audit,
        "validation_passed": validation_report["passed"],
        "figures": [str(Path(path).relative_to(output_dir)) for path in figures],
    }
    (output_dir / "graph_summary.json").write_text(
        json.dumps(_json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "contract_report.json").write_text(
        json.dumps(_json_safe(contract_report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "validation_report.json").write_text(
        json.dumps(_json_safe(validation_report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "resolved_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "graph_schema.json").write_text(
        json.dumps(build_graph_schema(contract, upstream_schema), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if schema_path:
        (output_dir / "upstream_schema.json").write_text(
            schema_path.read_text(encoding="utf-8"), encoding="utf-8"
        )

    input_hashes = {
        name: _file_sha256(Path(details["path"]))
        for name, details in contract_report["tables"].items()
    }
    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {key: str(value) for key, value in asdict(config).items()},
        "contract_version": contract["contract_version"],
        "input_sha256": input_hashes,
        "networkx_version": nx.__version__,
        "pandas_version": pd.__version__,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    dictionary = """# Stage 1 WP3 output data dictionary

This directory is generated by the contract-driven WP3 pipeline.

- `nodes.*`: canonical WP2 entity table adapted to the WP3 contract.
- `typed_edges_assertions.*`: included evidence-bearing relation assertions after policy filters.
- `typed_edges_collapsed.*`: one row per unique subject-predicate-object triple with evidence aggregates.
- `cooccurrence_edges.*`: undirected co-mention edge input.
- `entity_mentions_corrected.*`: article offsets plus reconciled sentence-local offsets and status.
- `node_metrics.*`: centrality and clustering metrics, separated by `graph_view`.
- `relation_type_summary.*`: relation-specific edge, endpoint, evidence, and date counts.
- `monthly_relation_assertions.*`: monthly assertion counts by predicate.
- `monthly_new_entities.*`: first-seen entities by month and entity type.
- `monthly_new_typed_edges.*`: first-seen typed facts by month and predicate.
- `node_communities.*`: co-occurrence community membership and within-community rank.
- `community_summary.*`: community size and representative entities.
- `graph_finding_candidates.*`: evidence-linked candidates for human interpretation; not approved conclusions.
- `assertion_graph.*`: directed MultiGraph preserving each relation assertion.
- `typed_graph_collapsed.*`: directed MultiGraph with one edge per typed triple.
- `cooccurrence_graph.*`: undirected weighted co-occurrence graph.
- `graph_summary.json`: counts, density, components, policy audit, and figure paths.
- `contract_report.json`: resolved upstream files and column aliases.
- `validation_report.json`: structural and schema validation results.
- `resolved_contract.json`: exact adapter contract used for the run.
- `graph_schema.json`: explicit graph-view semantics and Stage 3 provenance chain.
- `upstream_schema.json`: copied WP2 relation schema when available.
- `run_manifest.json`: versions, paths, timestamp, and input hashes.
- `figures/`: filtered graph views and metric charts for inspection.

Parquet files are the canonical machine-readable outputs. CSV files are convenience views.
Graph metrics describe the selected dataset and graph policy, not real-world influence.
"""
    (output_dir / "DATA_DICTIONARY.md").write_text(dictionary, encoding="utf-8")
    return summary


def run_pipeline(config: WP3PipelineConfig) -> dict[str, Any]:
    contract = read_contract(config.contract_path)
    tables, contract_report = load_contract_tables(config.input_dir, contract)
    schema_path, upstream_schema = find_optional_json(
        config.input_dir, contract.get("schema_files", [])
    )
    schema = normalize_relation_schema(upstream_schema)

    entities = prepare_entities(tables["entities"])
    relations = prepare_relations(tables["relations"])
    cooccurrence = prepare_cooccurrence(tables.get("cooccurrence"))
    documents = tables.get("documents")
    reviews = tables.get("reviews")
    policies = contract.get("policies", {})
    analysis = contract.get("analysis", {})

    raw_validation = validate_inputs(entities, relations, cooccurrence, None, policies)
    if not raw_validation["passed"]:
        raise ContractError(
            "WP3 structural input validation failed: " + "; ".join(raw_validation["errors"])
        )

    relations = annotate_final_relation_types(entities, relations, schema)
    included_relations, policy_audit = apply_relation_policy(relations, reviews, policies)
    validation = validate_inputs(entities, included_relations, cooccurrence, schema, policies)
    validation["raw_input"] = raw_validation
    if not validation["passed"]:
        raise ContractError(
            "WP3 post-policy validation failed: " + "; ".join(validation["errors"])
        )

    corrected_mentions = None
    if tables.get("mentions") is not None and tables.get("text_units") is not None:
        corrected_mentions = correct_sentence_local_offsets(
            tables["mentions"], tables["text_units"]
        )
    collapsed = collapse_relations(included_relations, documents)
    minimum_documents = int(policies.get("minimum_cooccurrence_documents", 1))
    filtered_cooccurrence = cooccurrence[
        cooccurrence["document_count"] >= minimum_documents
    ].copy().reset_index(drop=True)
    graphs = build_graphs(entities, included_relations, collapsed, filtered_cooccurrence, policies)

    sample_size = int(analysis.get("betweenness_sample_size", 200))
    seed = int(analysis.get("random_seed", 42))
    metric_frames = [
        calculate_node_metrics(graphs["typed"], "typed", "document_count", sample_size, seed),
        calculate_node_metrics(
            graphs["cooccurrence"], "cooccurrence", "document_count", sample_size, seed
        ),
        calculate_relation_type_metrics(entities, collapsed, sample_size, seed),
    ]
    metric_frames = [frame for frame in metric_frames if not frame.empty]
    metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()

    relation_summary = build_relation_type_summary(collapsed)
    temporal_tables = build_temporal_tables(entities, included_relations, collapsed)
    community_membership, community_summary = detect_cooccurrence_communities(
        graphs["cooccurrence"]
    )
    finding_candidates = build_finding_candidates(
        entities, collapsed, filtered_cooccurrence, metrics
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    figures = create_visualizations(
        graphs,
        metrics,
        entities,
        collapsed,
        relation_summary,
        temporal_tables,
        community_summary,
        config.output_dir,
        analysis,
    )
    return export_outputs(
        config,
        contract,
        contract_report,
        validation,
        policy_audit,
        entities,
        included_relations,
        collapsed,
        filtered_cooccurrence,
        corrected_mentions,
        graphs,
        metrics,
        relation_summary,
        temporal_tables,
        community_membership,
        community_summary,
        finding_candidates,
        figures,
        schema_path,
        schema,
    )
