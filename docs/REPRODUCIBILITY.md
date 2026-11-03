# Reproducibility boundary

## Verified CPU path

`analysis/reproduce.py` verifies curated hashes, then loads named function ASTs from byte-exact frozen source snapshots. It reuses `classify`, `parse_animal`, `animal_matrix`, `paired_effect`, `paired_names`, `confirm`, and the complete original `scoring.py`. The frozen `confirm` function's import of the operational authorization module is replaced by a local reader for the hash-verified approved selection. No numeric formula, loop, prompt, output, seed, or decision threshold changes.

A portable `load_eval` maps the curated raw directory to original task/model IDs, checks source prompt/problem rows and output token counts, and checks raw hashes against the original analysis evidence. Original remote receipt/environment digests remain as provenance references; the host-specific receipt files themselves are not included and are not claimed to be reverified here. The full original private audit already checked them. Result comparison checks all fields, with absolute 1e-12 tolerance only for floats; observed maximum difference was zero.

`analysis/make_figures.py` requires a verified recomputation and produces CSV/TeX summaries and five plots from all prespecified formats/conditions. It makes no new inferential choices. Dev tables contain both strict and amended counts, including every robustness model. All prompt-level counts, invalid outputs, and token-limit hits remain available.

## Training provenance versus portable reruns

Selected original training, preparation, and evaluation sources, exact model/training configurations, and all seven prepared data variants are included. They document what actually ran. The frozen scripts depend on the historical workspace layout and operational runtime; they are not advertised as a standalone training CLI. The operational account/authorization runtime and submission controllers are excluded. Do not execute these archival entry points as a fresh run.

Weights/checkpoints are not included. A full independent GPU rerun would require pinned Qwen model/tokenizer files, a compatible CUDA/PyTorch/Transformers/PEFT environment, a reviewed portable training entry point and training-output paths, then explicit compute allocation. That packaging is a remaining final-delivery item, not a reason to redo completed experiments now. Remote recorded key versions: Transformers 4.51.3, PEFT 0.15.2, tokenizers 0.21.4, PyTorch 2.14.1; NVIDIA RTX A5000, one GPU per allocation. The CPU requirements are separately pinned and do not reproduce that GPU environment.

## Data and evidence retention

All prepared data and raw evaluation files are byte-exact copies. No test set was resampled during reporting. Old test outputs were not used as fresh confirmation. The original raw/frozen workspace remains intact; curation does not replace it. The archived `no_gpu_authorization`, proposal-state docstrings, and operational fields are historical snapshots, not current authority.

The full original project remains the authoritative operational evidence archive. A repository curation manifest authenticates copied artifacts, not a cryptographic proof that the original experiment occurred. The original remote Slurm/audit receipts provide that additional operational evidence.
