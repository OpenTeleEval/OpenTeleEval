# OpenTeleEval — Evaluation Code (`code/`)

Everything under this directory is the **evaluation framework** (an OpenCompass fork),
plus configs, prompts, examples, and docs.

## License

All files in `code/` (and in the repository root) are licensed under the
**Apache License, Version 2.0**. Full text: [./LICENSE](./LICENSE).

The benchmark **datasets are licensed separately** — they live in the top-level
[`../datasets/`](../datasets/) directory under the Community Data License Agreement –
Permissive, Version 2.0 (CDLA-Permissive-2.0). See
[../DATASET_LICENSE.md](../DATASET_LICENSE.md). **Do not place raw data under `code/`**
— that would make it look Apache-licensed. Data configs reference the canonical store via
`path="../datasets/..."`.

## Layout

- `run.py` — CLI entry: `infer → eval → summarize`.
- `opencompass/` — core library (dataset loaders, models, evaluators, …).
- `configs/datasets/` — one reusable config per task.
- `examples/` — runnable templates (`python run.py examples/CoreNetwork.py`).

See [`../CONTRIBUTING.md`](../CONTRIBUTING.md) for the full structure and contribution
guide, and [`../NOTICE`](../NOTICE) for third-party attribution (OpenCompass, HuggingFace).
