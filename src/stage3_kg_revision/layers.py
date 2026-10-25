"""Separate article claims from curated lexicon knowledge in the Stage 1 graph."""

from __future__ import annotations

import pandas as pd

from .inputs import TRIPLE

# extraction_method of an assertion -> where the knowledge comes from
KNOWLEDGE_SOURCE = {
    "glirel": "article_claim",          # relation stated in an article sentence (GLiREL)
    "model_name": "name_pattern",       # model name with a size, e.g. "Qwen3-8B" -> member of family Qwen
    "wp1_lexicon": "curated_lexicon",   # model family -> developer from the WP1 lexicon, no article sentence
}


def label_assertions(assertions: pd.DataFrame) -> pd.DataFrame:
    labelled = assertions.copy()
    labelled["knowledge_source"] = labelled["extraction_method"].map(KNOWLEDGE_SOURCE)
    unknown = labelled["knowledge_source"].isna()
    if unknown.any():
        raise ValueError(f"unknown extraction methods: {sorted(labelled.loc[unknown, 'extraction_method'].unique())}")
    return labelled


def edge_layers(collapsed: pd.DataFrame, assertions: pd.DataFrame) -> pd.DataFrame:
    """One row per triple with its knowledge layer and article-only support counts.

    Stage 1 merged a triple found both in the lexicon and in articles into one Tier A row, and counted the lexicon row
    as evidence. Here the article support is counted on article sentences only, and the layer says which kind of
    knowledge the edge carries: an article claim (possibly also in the lexicon), a name pattern, or lexicon only.
    """
    labelled = label_assertions(assertions)
    article = labelled[labelled["knowledge_source"] == "article_claim"]
    article_support = (article.groupby(TRIPLE)
                       .agg(article_sentences=("text_unit_id", "nunique"), article_documents=("document_id", "nunique"),
                            article_first_seen=("published_at", "min"), article_last_seen=("published_at", "max"))
                       .reset_index())
    flags = (labelled.assign(value=True)
             .pivot_table(index=TRIPLE, columns="knowledge_source", values="value", aggfunc="any", fill_value=False)
             .reindex(columns=list(KNOWLEDGE_SOURCE.values()), fill_value=False)
             .rename(columns={"article_claim": "has_article_claim", "name_pattern": "has_name_pattern",
                              "curated_lexicon": "in_lexicon"})
             .reset_index())
    flags.columns.name = None

    edges = (collapsed.merge(flags, on=TRIPLE, how="left", validate="one_to_one")
             .merge(article_support, on=TRIPLE, how="left", validate="one_to_one"))
    for column in ["has_article_claim", "has_name_pattern", "in_lexicon"]:
        edges[column] = edges[column].fillna(False).astype(bool)
    for column in ["article_sentences", "article_documents"]:
        edges[column] = edges[column].fillna(0).astype(int)
    edges["knowledge_layer"] = "curated_lexicon"
    edges.loc[edges["has_name_pattern"], "knowledge_layer"] = "name_pattern"
    edges.loc[edges["has_article_claim"], "knowledge_layer"] = "article_claim"
    edges["lexicon_agreement"] = pd.NA
    edges.loc[edges["in_lexicon"] & edges["has_article_claim"], "lexicon_agreement"] = "lexicon_and_articles"
    edges.loc[edges["in_lexicon"] & ~edges["has_article_claim"], "lexicon_agreement"] = "lexicon_only"
    return edges


def lexicon_conflicts(relations_all_tiers: pd.DataFrame) -> pd.DataFrame:
    """Article relations that Stage 1 excluded only because they contradict the lexicon developer."""
    reasons = relations_all_tiers["exclusion_reason"].fillna("")
    conflicts = relations_all_tiers[reasons.str.contains("conflicts_lexicon_developer")].copy()
    conflicts["only_reason"] = reasons[conflicts.index].eq("conflicts_lexicon_developer")
    columns = ["head_id", "relation", "tail_id", "head", "tail", "n_distinct_sentences", "n_docs", "exclusion_reason",
               "only_reason"]
    return conflicts[columns].sort_values("n_distinct_sentences", ascending=False).reset_index(drop=True)
