# OpenTeleEval — Datasets (`datasets/`)

This directory holds the benchmark data: questions, labels, and task trajectories, split
into **Knowledge Comprehension** and **Knowledge Application**.

## License

All files in this directory and its subdirectories are licensed under the
**Community Data License Agreement – Permissive, Version 2.0** (CDLA-Permissive-2.0).
Full text: [./LICENSE](./LICENSE).

- **Sharing** this data (modified or not) requires including the license text
  (CDLA-Permissive-2.0 §2.1).
- **Results** — models, metrics, and insights derived from this data — carry **no**
  obligations under this license (CDLA-Permissive-2.0 §3.1). You may use them freely.
- This directory ships a **public subset only** (to prevent benchmark leakage). For the
  full benchmark needed to reproduce paper numbers, see
  [`../MAINTAINER.md`](../MAINTAINER.md).

The evaluation **code is licensed separately** under Apache-2.0 — see
[`../code/`](../code/) and [`../CODE_LICENSE.md`](../CODE_LICENSE.md). The two licenses
are mapped by directory; see [`../LICENSE`](../LICENSE) for the authoritative statement.

## Layout

```
datasets/
├── Knowledge_Comprehension/
│   ├── Basic Theory/        # Basic_Knowledge · 5G_Network · 3GPP_Protocols
│   └── Product Knowledge/   # Core_Network · Wireless_Network · Wired_Network
└── Knowledge_Application/
    ├── Intent_Recognition/  · Entity_Extraction/ · Event_Verification/
    ├── Root_Cause_Diagnosis/ · Solution_Generation/ · Tool_Invocation/
```
