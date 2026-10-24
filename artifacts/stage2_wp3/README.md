# Stage 2 · Part 3 (WP3): Comparative Evaluation and Insights

Outputs of `stage2/work_packages/stage2_wp3_evaluation.ipynb`. The same files are on Google Drive in `HSLU/CLT/extraction_outputs/stage2/wp3/`, together with `tokens_sample.parquet` (cache of the tokens of the 8,037 compared articles, not versioned).

| file | content | section |
|---|---|---|
| `intrinsic_words_general.csv` | WordSim-353 and SimLex-999 Spearman correlations of Word2Vec and FastText | 3.2.1 |
| `intrinsic_words_developers.csv` | family → developer retrieval (hit@1, hit@3, MRR) and analogy accuracy | 3.2.2 |
| `intrinsic_words_categories.csv` | silhouette and 5-NN purity of the term categories of the Part 1 map | 3.2.3 |
| `intrinsic_articles_themes.csv` | kNN accuracy, MAP@10, silhouette, NMI and ARI of every representation (206 validation articles) | 3.3.1 |
| `intrinsic_articles_track.csv` | 10-NN purity of open-models vs other articles | 3.3.2 |
| `extrinsic_theme_classification.csv` | macro-F1 (95% bootstrap interval, hand-checked subset) of every representation | 3.4 |
| `extrinsic_paired.csv` | paired bootstrap differences | 3.4 |
| `tradeoffs.csv`, `timing_classical.json` | parameters, training and inference time, GPU memory, scores | 3.5 |
| `discourse_clusters.csv` | Doc2Vec discourse clusters of the whole corpus | 3.6.1 |
| `track_themes_by_year.csv`, `track_themes_by_region.csv`, `track_themes_by_family.csv` | predicted themes of the open-models track | 3.6.2 – 3.6.3 |
| `synthesis.csv` | overview of all representations | 3.6.5 |
| `figures/` | all figures of the notebook | |
