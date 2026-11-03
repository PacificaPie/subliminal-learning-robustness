# Subliminal learning: robustness and mitigation

Course research project: **Evaluating the Robustness and Mitigation of Subliminal Learning in Large Language Models**.

William Zhong · Darren Chen · Pacifica Zhang

CPSC 4710/5710, Trustworthy Deep Learning, Fall 2026

**Status:** experiments and final evidence audit complete; milestone and final reports are working drafts. This is a private repository preparation, not a new experiment authorization.

We study whether number-only synthetic training responses transfer a cat preference to Qwen2.5-7B-Instruct, how fragile that transfer is, and whether a practical data intervention can reduce it without detectable damage on a fixed math benchmark.

## Main findings

| Fresh confirmation, plain format | Cat responses | Frequency |
|---|---:|---:|
| Cat-trained | 396 / 720 | 55.00% |
| Matched neutral-trained | 10 / 720 | 1.39% |
| 50% neutral replacement (mix50) | 12 / 720 | 1.67% |

The paired excess-cat gap falls from **53.61 to 0.28 percentage points**. Relative suppression is **99.48%**, with a paired prompt-family bootstrap interval of **98.32–100.00%**. These intervals are conditional on six hand-designed families and three observed seeds.

On 200 fresh GSM8K questions, mean accuracy is 86.50% for cat-trained and 88.83% for mix50. The paired accuracy-loss interval is **[-5.17, +0.33] pp**, satisfying the predefined upper-loss limit of 5 pp. This is a damage screen, not a claim of general capability preservation or significant improvement.

Single-seed robustness checks show strong sensitivity to fewer training pairs, one epoch, and canonical number formatting. Strict TAG adherence is poor. A **post-hoc, development-only casing amendment** was approved before any fresh confirmation; the original null selection is preserved. See [protocol](docs/PROTOCOL.md) and [limitations](docs/RESULTS_AND_LIMITATIONS.md).

## Reproduce the analysis without a GPU

Python 3.12 was used for local verification. From this directory:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python analysis/reproduce.py
.venv/bin/python analysis/make_figures.py
```

The first command checks every curated input hash, reuses frozen scoring/statistical functions, and compares the entire recomputed final result with the archived reference (138 floating-point fields; tolerance 1e-12). The reference values reproduced exactly on the preparation host. It checks **6,000 dev, 11,520 animal, and 2,000 math answers**. No inference, cluster connection, or GPU allocation is involved.

To compile the drafts, install a LaTeX distribution containing the standard packages used in the source, then run:

```sh
make reports
```

## Repository map

- `raw/`: byte-exact model responses, including invalid outputs and development failures.
- `prepared/`: locked training variants, fresh prompt families, GSM8K inputs/targets, and data/mask manifests.
- `config/`: historical model, protocol, and training configuration snapshots.
- `frozen/`: original scoring/analysis and selected experiment sources, unchanged. These are archival sources; `frozen/experiment/` is **not** a standalone cluster training package.
- `analysis/`: portable CPU adapter and reporting scripts.
- `reference/`: original null selection, amended selection, and final analysis.
- `provenance/`: source hashes, amendment, seals, complete compute accounting summary and allocation table.
- `report/`: milestone and final working drafts, generated figures/tables, references.
- `docs/`: protocol, results, research chronology, reproducibility boundary, and release checklist.

Weights and optimizer checkpoints remain on Bouchet and were checked in the final remote audit; they are not included here. Full private operational evidence, environment receipts, historic source trees, and execution controllers remain preserved in the original workspace. See [reproducibility](docs/REPRODUCIBILITY.md).

## Compute and scope

Total project use: **206,804 GPU seconds = 57.4456 GPU hours**, including historical failures/cancellations. Historical 139,620 seconds are retained; the new study adds 67,184 seconds. The authorized ceiling remains 95 hours. No further GPU work is required for these reports.

Activation patching, component localization, and mechanistic ablation are outside the revised project scope. This repository does not package the older mechanistic results as new confirmation evidence.

## Attribution and release

See [third-party notices](THIRD_PARTY_NOTICES.md). A license for original team material has not yet been selected; included third-party licenses remain in force. Publication, team contribution statements, and course-specific final formatting remain review items. Do not infer deployment or job-submission authority from archived configuration fields.
