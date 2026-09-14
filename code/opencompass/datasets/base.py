# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The OpenTeleEval Authors.

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from opencompass.openicl import DatasetReader
from opencompass.openicl.icl_evaluator import BaseEvaluator


class BaseDataset:
    """Base class for all datasets.

    Subclasses should implement the static ``load`` method, which returns a
    ``datasets.Dataset`` (or ``DatasetDict``). The instance then exposes
    ``reader``, ``train`` and ``test`` for the OpenICL pipeline.
    """

    def __init__(self,
                 reader_cfg: Optional[dict] = None,
                 dataset_cfg: Optional[dict] = None,
                 **kwargs):
        if reader_cfg is None:
            reader_cfg = {}
        load_kwargs = dict(dataset_cfg or {})
        load_kwargs.update(kwargs)
        self.dataset = self.load(**load_kwargs)
        self._init_reader(**reader_cfg)

    def _init_reader(self, **kwargs):
        self.reader = DatasetReader(self.dataset, **kwargs)

    @property
    def train(self):
        return self.reader.dataset['train']

    @property
    def test(self):
        return self.reader.dataset['test']


class BaseJudgeACCEvaluator(BaseEvaluator):
    """Base class for LLM-as-judge accuracy evaluators.

    Subclasses should implement:

    - ``_get_prompt()``: return the judge prompt template containing
      ``{question}``, ``{reference}`` and ``{prediction}`` placeholders.
    - ``_get_judge_model()``: return a judge model instance exposing
      ``chat(message)`` or ``predict(message)``.
    - ``_extract_judge(judge_message)``: parse the judge output into
      ``True`` (correct), ``False`` (incorrect) or ``None`` (parse failure).
    """

    def __init__(self, prompt: Optional[str] = None, judge_model=None):
        self._prompt = prompt
        self._judge_model = judge_model

    def _get_prompt(self) -> str:
        raise NotImplementedError

    def _get_judge_model(self):
        raise NotImplementedError

    def _extract_judge(self, judge_message: str):
        raise NotImplementedError

    def _judge_one(self, judge_model, prompt, question, prediction,
                   reference):
        message = prompt.format(
            question=question, prediction=prediction, reference=reference)
        chat = getattr(judge_model, 'chat', None) or getattr(
            judge_model, 'predict', None) or judge_model
        judge_message = chat(message)
        return self._extract_judge(judge_message), judge_message

    def score(self, predictions, references, questions=None, **kwargs):
        prompt = self._prompt or self._get_prompt()
        judge_model = self._get_judge_model()

        n = len(predictions)
        if questions is None:
            questions = [''] * n

        results = [None] * n
        judge_messages = [None] * n
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(self._judge_one, judge_model, prompt,
                                questions[i], predictions[i], references[i]):
                i
                for i in range(n)
            }
            for future in as_completed(futures):
                i = futures[future]
                try:
                    results[i], judge_messages[i] = future.result()
                except Exception:
                    results[i], judge_messages[i] = None, 'LLM ERROR'

        correct = sum(1 for r in results if r is True)
        judged = sum(1 for r in results if r is not None)
        accuracy = 100 * correct / n if n else 0.0
        details = [
            dict(
                prediction=predictions[i],
                reference=references[i],
                judge_message=judge_messages[i],
                correct=bool(results[i]) if results[i] is not None else None,
            ) for i in range(n)
        ]
        return {
            'accuracy': accuracy,
            'correct': correct,
            'judged': judged,
            'total': n,
            'details': details,
        }


class BaseJudgeScoreEvaluator(BaseJudgeACCEvaluator):
    """Base class for LLM-as-judge scored evaluators (e.g. 5-point scale).

    Same subclass contract as :class:`BaseJudgeACCEvaluator`, except
    ``_extract_judge`` returns a numeric score (or ``None`` on parse
    failure). The aggregated result is the mean score scaled to 0-100.
    """

    def __init__(self, prompt: Optional[str] = None, judge_model=None,
                 max_score: int = 5):
        super().__init__(prompt=prompt, judge_model=judge_model)
        self._max_score = max_score

    def score(self, predictions, references, questions=None, **kwargs):
        prompt = self._prompt or self._get_prompt()
        judge_model = self._get_judge_model()

        n = len(predictions)
        if questions is None:
            questions = [''] * n

        scores = [None] * n
        judge_messages = [None] * n
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(self._judge_one, judge_model, prompt,
                                questions[i], predictions[i], references[i]):
                i
                for i in range(n)
            }
            for future in as_completed(futures):
                i = futures[future]
                try:
                    scores[i], judge_messages[i] = future.result()
                except Exception:
                    scores[i], judge_messages[i] = None, 'LLM ERROR'

        valid = [s for s in scores if s is not None]
        mean_score = sum(valid) / len(valid) if valid else 0.0
        details = [
            dict(
                prediction=predictions[i],
                reference=references[i],
                judge_message=judge_messages[i],
                score=scores[i],
            ) for i in range(n)
        ]
        return {
            'score': 100 * mean_score / self._max_score,
            'mean_score': mean_score,
            'judged': len(valid),
            'total': n,
            'details': details,
        }
