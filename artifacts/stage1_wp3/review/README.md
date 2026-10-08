# Stage 1 WP3 - relation review

The WP2 relations were first screened with Claude, and the errors it found were used to correct the relations.
This review is the human check of the corrected relations.

Check 90 relations extracted from the articles. Each reviewer takes the 30 rows of one
`reviewer_assignment` (`Reviewer_A`, `Reviewer_B`, `Reviewer_C`) in `relation_review_corrected_seed2026.xlsx`.

## For each row

1. Read the sentence in `evidence_text` (plus the sentences before and after it).
   If unclear, also read `support_2_context` and `support_3_context`.
2. Decide whether the sentences say that **head → predicate → tail** is true,
   e.g. `Llama → developed by → Meta`. Check the direction too (`competes with` has none).
3. Fill in only these columns:
   - `review_correct`: `TRUE`, `FALSE`, or `AMBIGUOUS` (the sentences are not enough to decide)
   - `reviewer`: your name
   - `review_error_type`: only for `FALSE`, pick from the dropdown
   - `review_notes`: a short reason

| error type | meaning |
|---|---|
| `entity_boundary` | wrong or too generic entity (e.g. "AI models") |
| `entity_linking` | wrong entity picked, e.g. from a list of names |
| `wrong_predicate` | right entities, wrong relation |
| `reversed_direction` | head and tail swapped |
| `negation_or_uncertainty` | negative or speculative statement |
| `insufficient_context` | the sentences do not support it |
| `duplicate_or_noise` | duplicate or noise |
| `schema_or_catalogue` | wrong entity type |
| `other` | explain in the notes |

When you are done, commit the filled-in workbook to GitHub.
