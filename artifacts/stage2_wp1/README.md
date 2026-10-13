# Stage 2 - WP1: classical word & sentence embeddings

| file | content |
|---|---|
| `splits.parquet` | training / validation split of every article (seed 42, 80/20, stratified by `om_tier`). **All Stage 2 work packages use this split.** |
| `terms_tsne.csv` | terms, categories and t-SNE coordinates of the semantic map |
| `figures/` | visualisations |

The trained models (`word2vec.model`, `fasttext.model`, `doc2vec.model`), the article vectors (`doc_vectors.parquet`) and the tokenised corpus are in the Google Drive folder `HSLU/CLT/extraction_outputs/stage2/wp1/` (shared with the team), not on GitHub (too large). To reuse them in Colab, add a shortcut to this folder in My Drive and set `OUT_DIR` to its path: the notebook then loads them instead of retraining, which reproduces the results of the submitted run.
