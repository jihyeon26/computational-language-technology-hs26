# Stage 3 - WP1: Q&A test set and text-only RAG baseline (Open Models)

Notebook: `stage3/work_packages/wp1/stage3_wp1_qa_rag.ipynb`

| file | content |
|---|---|
| `qa_reviewed.json` | **final test set**: 213 questions (factual 57, relational 48, comparative 43, temporal 38, analytical 27). Fields: `id`, `type`, `question`, `reference_answer`, `source_ids` (Stage 1 `doc_id` of the source article), `date`, `passage_id`, `quarter`, `topic` |
| `qa_reviewed.csv` | all drafted questions after the manual review (keep / edit / drop), with notes and reviewer |
| `qa_draft.csv` | the drafted questions before the review |
| `passages_for_qa.csv` | the 150 passages the questions were drafted from |
| `passages_temporal_extra.csv` | the 15 extra passages with an explicit date (temporal questions) |
| `rag_settings.json` | settings of the RAG baseline: corpus, chunks, embedding model, retrieval, generation, RAG model |

## RAG baseline
- Corpus: 1,210 core articles (`om_tier == 'core'`), `id` = Stage 1 `doc_id`
- Chunks: {'size': 200, 'overlap': 40, 'contextualize': 'title'}; embeddings: `all-MiniLM-L6-v2`
- Retrieval: {'strategy': 'hybrid', 'size': 30}, provisional (chosen on 5 examples, to be confirmed by the evaluation)
- Generation: {'mode': 'grounded', 'max_passages': 6, 'max_per_source': 2, 'max_chars_per_passage': 1200, 'max_tokens': 400}
- RAG model: `qwen3:4b-instruct` with Ollama (local, no API key). The brief requires a **different** model for the evaluation (judge).

## Large files (shared Google Drive, not on GitHub)
`HSLU/CLT/extraction_outputs/stage3/rag/`: `core_articles.csv` (the ingested corpus) and the chunk embeddings `embeddings_core_articles_size200_overlap40_title_all_MiniLM_L6_v2.npy`. Copy them into your `RAG_DIR` to reload them instead of recomputing them.
