# Third-party material and licensing

- Subliminal-learning research/code provenance: https://github.com/MinhxLe/subliminal-learning , revision `db04f4150edf940559b5f3147f65d808e9313efd`. The original MIT license is retained in `licenses/subliminal-learning-MIT.txt`. Locally written experimental code and modifications are identified by this repository's provenance, not represented as unmodified upstream code.
- Original number-question source: https://huggingface.co/datasets/minhxle/subliminal-learning_numbers_dataset , revision `4fa5997d83c88edf028e2c13c7107baa731eb30c`. The paired cat/neutral responses and deterministic variants here are generated research artifacts; public cat/bear parquet files are not bundled. The pinned dataset card, checked November 12, 2026, does not declare a dataset license. Do not assume the upstream code's MIT license also licenses the dataset; this repository does not assign a license to that source material.
- GSM8K: Karl Cobbe et al., https://github.com/openai/grade-school-math . The 200-question subset and corresponding numeric targets retain the MIT license in `licenses/GSM8K-MIT.txt`; model-generated answers are study outputs.
- Qwen2.5-7B-Instruct: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct , Apache-2.0 model card. No model weights/tokenizer assets are bundled in this repository.
- Cited papers/blogs are referenced in `report/references.tex`; their full text is not redistributed.

This repository is public for research and team review. No blanket reuse license is assigned to the team's original material; any future license should be agreed with the authors. The third-party licenses above retain their own scope. The retained GSM8K and upstream-code MIT notices were checked against their official sources on November 12, 2026.
