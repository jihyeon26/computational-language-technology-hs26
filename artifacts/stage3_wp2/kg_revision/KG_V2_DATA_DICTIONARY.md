# Knowledge graph v2 (Stage 3 WP2): data dictionary

Built from the submitted Stage 1 knowledge graph by `stage3_wp2_kg_revision.ipynb`. Node, relation and text unit IDs
are those of Stage 1; all dates are article publication dates. `GraphEvidenceRetriever.from_directories(dir, dir)`
loads the folder `outputs/stage3_kg_v2/`.

## `edges` - one row per triple (all Stage 1 graph edges)
Stage 1 columns, plus:
| column | description |
|---|---|
| `knowledge_layer` | `article_claim` (stated in an article sentence), `name_pattern` (model name with size -> family), `curated_lexicon` (WP1 lexicon only, no sentence) |
| `has_article_claim`, `has_name_pattern`, `in_lexicon` | sources that contain the triple |
| `lexicon_agreement` | `lexicon_and_articles` or `lexicon_only` for lexicon triples |
| `article_sentences`, `article_documents`, `article_first_seen`, `article_last_seen` | support counted on article sentences only |
| `support_status` | `supported` (a sentence entails the relation), `not_supported`, `name_pattern`, `no_passage` (lexicon only) |
| `support_max`, `support_mean`, `scored_sentences`, `supported_sentences` | NLI support of the sentences |
| `relation_reliability` | `low` for `competes with` (cannot be validated at sentence level), else `standard` |
| `retrieval_eligible` | used by the retriever: supported article claim or name pattern, `relation_reliability = standard` |

## `assertions` - one row per evidence sentence
Stage 1 columns (`relation_id`, `subject_id`, `predicate`, `object_id`, `document_id`, `text_unit_id`, `evidence_text`,
`published_at`, `extraction_method`, `rule_score`), plus `knowledge_source`, `support_score` (NLI entailment
probability), `contradiction_score`, `supported` (support_score >= 0.5), `hypothesis` (the sentence the relation was
checked as) and `retrieval_eligible`. `retrieval_assertions` / `typed_edges_assertions` hold the rows the retriever uses.

## `background_facts`
Lexicon facts (`subject_id`, `predicate`, `object_id`, `knowledge_source = curated_lexicon`), returned by the retriever
as `background`, never as cited evidence.

## `text_units`, `documents`
Sentences (`text_unit_id` = `doc_id:sentence_id`) and articles (publication date, title, URL, domain).

## `typed_graph_v2.graphml`
All edges with `knowledge_layer`, `support_status`, `retrieval_eligible`.
