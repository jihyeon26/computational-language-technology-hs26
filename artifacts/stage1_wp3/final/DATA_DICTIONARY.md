# Stage 1 WP3 knowledge graph: data dictionary

Schema `stage1-wp3-open-models-kg-v1.0`, review status `human_reviewed_sample`, track Open Models.
All dates (`published_at`, `first_seen`, `last_seen`) are article publication dates.

## Nodes (`nodes.*`)
| column | description |
|---|---|
| `entity_id` | corrected node ID; the WP2 IDs merged into it are listed in `merged_ids` |
| `canonical_name`, `aliases` | display name and merged alternative names (JSON list) |
| `entity_type`, `entity_type_compact` | WP2 entity type and one of the 9 compact types |
| `mention_count`, `document_count` | mentions and articles mentioning the entity |
| `is_open_model_family` | model family from the WP1 lexicon |
| `parameter_sizes_b` | parameter sizes in billions (JSON list): the size in the name for model nodes, the sizes of the member models for family nodes |
| `typed_*`, `cooc_*`, `community_id` | metrics in the typed and co-occurrence graphs, co-occurrence community |

## Typed relations
- `typed_edges_assertions.*`: one row per evidence sentence: `relation_id`, `subject_id`, `predicate`, `object_id`,
  `document_id`, `text_unit_id` (`doc_id:sentence_id`), `evidence_text`, `published_at`, `extraction_method`
  (`glirel`, `wp1_lexicon` or `model_name`), `rule_score`.
- `typed_edges_collapsed.*`: one row per unique triple: `evidence_count`, `document_count` (edge weight),
  `n_distinct_sentences`, `first_seen`, `last_seen`, `tier` (A: lexicon, B: >= 2 distinct sentences, >= 3 for
  `competes with`), `sources`, `added_by` (refinement, role-based typing or model names), `review_status`
  (`reviewed_*` from the random sample, `checked_true_human` / `checked_true_ai` from the findings check),
  `predicate_group`, `directed`, `relation_ids` and `text_unit_ids` (JSON lists).

## Co-occurrence (`cooccurrence_edges.*`)
Entity pairs mentioned in the same sentence in at least 2 articles: `sentence_count`,
`document_count`, `npmi`, `merged_pairs` (original WP2 pairs combined by the refinement). Not a semantic relation.

## Graph files
`assertion_graph`, `typed_graph` and `cooccurrence_graph` as `.graphml` and node-link `.json`.

## Analysis tables
`node_metrics`, `relation_type_metrics`, `relation_type_summary`, `node_communities`, `community_summary`,
`quarterly_relation_assertions`, `quarterly_new_typed_edges`, `family_quarterly_share`, `entity_period_share`,
`relations_first_seen_2026`, `findings`, `model_name_mentions` (sentences behind the model nodes: family, model
node, size, original text `surface`).

## Correction tables (`correction/`)
`relations_all_tiers` (tier and exclusion reason of every relation), `evidence_status`, `nodes_corrected`,
`name_variants`, `alias_map`, `node_overrides`, `developer_equivalents`, `generic_nodes`, `manual_exclusions`.

## Provenance for Stage 3
`entity_id` -> `relation_id` -> `text_unit_id` -> `document_id` -> `published_at`. Article titles and URLs are
joined from the WP1 table `open_models_core.parquet` on `doc_id`.
