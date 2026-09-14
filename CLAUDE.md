# CLAUDE.md

Guidance for Claude (and other AI coding assistants) working in this repository.
It explains how the OpenTeleEval evaluation framework is wired, how to run an
evaluation end to end, and the pitfalls that are easy to trip over when editing
this codebase.

## What this project is

OpenTeleEval is a telecommunications-domain LLM benchmark: 12 task groups /
22,678 samples covering *Knowledge Comprehension* (basic theory, 5G, 3GPP,
vendor product knowledge) and *Knowledge Application* (intent recognition,
entity extraction, tool invocation, event verification, root-cause diagnosis,
solution generation). The framework is a trimmed fork of OpenCompass 0.2.x
(package name `opencompass`, distribution name `openteleeval`).

**Data availability:** the repo ships only small public example subsets under
`datasets/` (~330 samples total; some subtypes are empty and legitimately
report `-` in the summary). Full data comes from the maintainers
(`MAINTAINER.md`).

## Repository layout

```
code/                       # the evaluation framework (pip-installable)
  run.py                    # entry point: python run.py <config.py>
  setup.py                  # package: openteleeval (module: opencompass)
  requirements/requirements.txt
  examples/                 # ready-to-copy eval configs
    CoreNetwork.py          # subjective QA + LLM-as-judge
    BasicKnowledge.py       # multiple-choice + rule-based scoring
  configs/datasets/         # per-dataset configs (prompt, retriever, evaluator)
  opencompass/
    models/                 # ReasoningAPI/NonReasoningAPI (OpenAI-compatible),
                            # HuggingFace local models (untested path)
    datasets/               # dataset loaders + evaluators; base.py has the
                            # BaseDataset / judge evaluator base classes
    judge_models/           # OpenAIJudge (judge over OpenAI-compatible API),
                            # JudgeLlama (legacy fallback, hardcoded URL)
    openicl/                # ICL pipeline: retrievers, inferencers, evaluators
    partitioners/ runners/ tasks/ summarizers/
datasets/                   # public example data (JSON), CDLA-Permissive-2.0
```

## Environment setup

```bash
conda create -n openteleeval python=3.10 -y
conda activate openteleeval
pip install -e code/
python -c "from opencompass import __version__; print(__version__)"  # -> 0.2.0
```

Python >= 3.8 works; 3.10 is the tested version. GPU is NOT required for
API-based evaluation (the default path); `torch` is installed but only used
by the local-model code path.

## Running an evaluation

Every eval is driven by one Python config file (mmengine `Config`). The
canonical workflow:

```bash
cd code
cp examples/CoreNetwork.py my_eval.py
# edit my_eval.py: model name, api_url, api_headers, judge_model_cfg
python run.py my_eval.py            # infer -> eval -> summarize
```

A minimal working config against a vLLM OpenAI-compatible endpoint:

```python
from mmengine import read_base
from opencompass.models import ReasoningAPI
from opencompass.partitioners import NaivePartitioner, SizePartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask

with read_base():
    from .configs.datasets.core_network.core_network_gen import core_network_datasets

datasets = sum((v for k, v in locals().items() if k.endswith("_datasets")), [])

judge_model_cfg = dict(base_url="http://<host>:<port>/v1",
                       model="<judge-model>", api_key="None")
for _ds in datasets:
    _ev = _ds.get("eval_cfg", {}).get("evaluator")
    # Only LLM-as-judge evaluators accept judge_model. IMPORTANT: under
    # mmengine lazy import, `type` is a placeholder object — match by string,
    # getattr(x, '__name__') returns the literal string '__name__'.
    if _ev and "CoreNetworkEvaluator" in str(_ev.get("type")):
        _ev["judge_model"] = judge_model_cfg

model = "your-model-name"
models = [dict(
    abbr=f"ReasoningAPI_{model}", type="ReasoningAPI",
    path=f"ReasoningAPI_{model}",
    api_url="http://<host>:<port>/v1/chat/completions",
    api_headers={"Content-Type": "application/json", "Authorization": "None"},
    api_data=dict(model=model, max_tokens=8192, temperature=0.7),
    meta_template=dict(round=[
        dict(role="SYSTEM", api_role="SYSTEM"),
        dict(role="HUMAN", api_role="HUMAN"),
        dict(role="BOT", api_role="BOT", generate=True)]),
    enable_thinking=True, batch_size=8)]

infer = dict(partitioner=dict(type=SizePartitioner, strategy="split",
                              gen_task_coef=1, max_task_size=256),
             runner=dict(type=LocalRunner, max_num_workers=8,
                         task=dict(type=OpenICLInferTask)))
eval = dict(partitioner=dict(type=NaivePartitioner),
            runner=dict(type=LocalRunner, max_num_workers=64,
                        task=dict(type=OpenICLEvalTask)))
work_dir = f"eval_result/{model}/"
```

### run.py cheatsheet

| Command | Effect |
|---|---|
| `python run.py cfg.py` | full pipeline: infer → eval → summary |
| `python run.py cfg.py -m infer` | inference only |
| `python run.py cfg.py -m eval -r latest` | re-score the newest run (requires `-r`) |
| `python run.py cfg.py -m viz -r latest` | re-print/write the summary table only |
| `python run.py cfg.py -r latest` | resume: reuse existing predictions, run what's missing |
| `--debug` | single-process, logs to stdout (use this first when debugging) |
| `--dry-run` | partition tasks and print them without running |
| `-w <dir>` | override `work_dir` |

Outputs land in `<work_dir>/<timestamp>/`:

```
predictions/<model>/<dataset>.json   # raw generations (+ gold)
results/<model>/<dataset>.json       # metrics per dataset
logs/infer|eval/...                  # per-task logs — check these on failure
summary/summary_<ts>.txt|.csv        # aggregated table
```

Datasets with zero samples (empty public subsets) show `-` in the summary —
that is expected, not a bug.

## Validated reference setup

The whole pipeline was validated end to end (all 50 public-subset dataset
configs, zero task failures) with:

- Python 3.10 conda env, CPU-only, `pip install -e code/`
- vLLM serving a Qwen3 model at `http://<host>:<port>/v1`
- `ReasoningAPI` + `enable_thinking=True`, judge = same endpoint

When modifying framework code, re-run a small config (e.g. CoreNetwork's
10-sample subset) with `--debug` before running the full suite, and always
finish with a full `-r latest` resume run to confirm nothing regressed.
