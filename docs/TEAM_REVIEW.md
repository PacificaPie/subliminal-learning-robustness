# Team review guide

Review snapshot: November 12, 2026. The main experiments and final artifact/accounting audit are complete. Both reports remain drafts; individual contributions and submission formatting still need team confirmation.

## Suggested reading order

1. Read the [3-page milestone draft](../report/milestone.pdf) for the question, design, results, and remaining work.
2. Review the [9-page final draft](../report/final_report.pdf), especially the development amendment, uncertainty, and limitations. Editable [milestone](../report/milestone.tex) and [final](../report/final_report.tex) LaTeX sources are included.
3. Check [results and claim boundaries](RESULTS_AND_LIMITATIONS.md) against the [figures](../report/figures/), [CSV tables](../report/tables/), and [frozen final analysis](../reference/confirmation_analysis.json).
4. Inspect the [protocol](PROTOCOL.md), [research chronology](RESEARCH_RECORD.md), [raw responses](../raw/), and [experimental code snapshots](../frozen/experiment/).
5. Follow the [README CPU reproduction instructions](../README.md#reproduce-the-analysis-without-a-gpu). The script verifies 103 curated artifacts and compares the full result, including 138 floating-point fields. The November 12 check matched exactly.

## Results to check

| Fresh plain-format confirmation | Cat answers | Frequency |
|---|---:|---:|
| Cat-trained | 396 / 720 | 55.00% |
| Matched neutral-trained | 10 / 720 | 1.39% |
| 50% neutral replacement (mix50) | 12 / 720 | 1.67% |

The excess-cat gap falls from 53.61 to 0.28 percentage points: 99.48% relative suppression. This is conditional on one model/trait, six designed prompt families, and three observed training seeds. On the same 200 GSM8K problems, mean accuracy is 86.50% for cat-trained and 88.83% for mix50; the paired loss interval is [-5.17, +0.33] pp. It passes the prespecified damage screen without establishing general capability preservation or a significant improvement.

The completed new study has 57 audited allocations, including one smoke check. Smoke verifies execution and is not research evidence. Total project usage is 57.4456 GPU hours, including historical failed/cancelled allocations.

## Feedback requested before submission

- Does the threat model and the practical meaning of paired neutral replacement come across clearly?
- Are the post-hoc dev-only casing correction, original null selection, and unchanged confirmation scorer described accurately?
- Are the single-seed robustness results, TAG-format compliance failures, and limited GSM8K screen interpreted cautiously enough?
- Which representative raw successes/failures should we discuss without changing the frozen scoring or selection?
- Confirm author names and actual contributions; agree on the course template/page limit and any broader license for original material.

For a useful comment, include the report section/page or repository path, the concern, and a proposed correction. Roles have not been assigned or inferred.

## Reproduction and remaining delivery work

CPU analysis is portable. Training and evaluation source snapshots document the original runs, but they depend on the historical workspace and are not a standalone GPU training CLI. Weights/checkpoints remain outside this repository. A portable training package remains to be completed if full retraining is part of the final deliverable; see [reproducibility](REPRODUCIBILITY.md).

The remaining work is team review, interpretation/failure cases, contributions, formatting, and that training package. The completed protocol does not need additional GPU experiments to fill the milestone-to-final interval.
