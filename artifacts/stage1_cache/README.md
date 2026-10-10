# Stage 1 cache

Files that let `stage1/stage1_final.ipynb` run end to end (for example in Colab) with the same results as the
submitted run. The first cell of the notebook copies them to the paths the notebook reads; existing files are
not overwritten.

| file | copied to | what it replaces |
|---|---|---|
| `extraction/gliner_entities.csv` | `data/stage1_wp2/` | the GLiNER run over all sentences (GPU, about 1 hour) |
| `extraction/relations.csv` | `data/stage1_wp2/` | the GLiREL run (GPU, several hours) |
| `preprocessing/lemma_cache.parquet` | `data/processed/` | spaCy lemmatisation of the core articles |
| `preprocessing/track_scan_cache.parquet` | `data/processed/` | the lexicon scan of the cleaned corpus |

The extraction files are needed for reproducible results: a new GLiNER or GLiREL run can give slightly different
relations, and the human review of the knowledge-graph relations refers to the relations of this run. The
preprocessing files only save time.

The raw dataset (`ai_media_dataset_20260911.csv`, 530 MB) is too large for GitHub. The notebook reads it from
`data/raw/`, from Google Drive, or downloads it from Kaggle, and checks its SHA-256 checksum
(`82dcea5ee2b15faf805994043fa7e8afb37b82fe8df7ab4fba8873db386f05b9`).

The topic models (Section 7) are not cached: with the pinned versions in `requirements.txt` (numpy 2.4.2,
pandas 3.0.1, scikit-learn 1.9.1) Colab reproduces the submitted topics. macOS gives different topics because
its numpy uses a different linear-algebra library (Accelerate instead of OpenBLAS).
