from functools import lru_cache
from allennlp.common.util import is_distributed
from typing import Dict, List, Optional, Union

import torch
import torch.distributed as dist
from overrides import overrides

from allennlp.training.metrics.metric import Metric

import datasets


@Metric.register("huggingface")
class HuggingFaceMetric(Metric):

    def __init__(
        self,
        path: str,
        name: str,
        keep_in_memory: bool = True,
        # **kwargs
    ):
        self.path = path
        self.name = name
        self.keep_in_memory = keep_in_memory
        # self.config_kwargs = kwargs
        self._metric: datasets.Metric
        self._num_calls = 0

        self.reset()

    # @lru_cache(maxsize=None)
    def get_huggingface_metric(
        self,
        path: str,
        name: str = None,
    ) -> datasets.Metric:
        return datasets.load_metric(path, name,
                                    keep_in_memory=self.keep_in_memory,
                                    # **self.config_kwargs
                                    )

    @overrides
    def reset(self) -> None:
        self._metric = self.get_huggingface_metric(self.path, self.name)
        self._num_calls = 0

    @overrides
    def __call__(
        self,
        predictions: torch.Tensor,
        gold_labels: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ):
        """
        # Parameters

        predictions : `torch.Tensor`, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : `torch.Tensor`, required.
            A tensor of integer class label of shape (batch_size, ...). It must be the same
            shape as the `predictions` tensor without the `num_classes` dimension.
        mask : `torch.BoolTensor`, optional (default = `None`).
            A masking tensor the same size as `gold_labels`.
        """
        if is_distributed():
            raise NotImplementedError
        self._num_calls += len(predictions)
        self._metric.add_batch(predictions=predictions.detach().cpu().numpy(),
                               references=gold_labels.detach().cpu().numpy())

    @overrides
    def get_metric(
        self,
        reset: bool = False
    ) -> Dict[str, float]:
        if not reset:
            # Huggingface metrics are not optimized for
            # realtime metric reporting
            # We'll better report them on the end of the epoch / loop
            return {}

        scores = self._metric.compute()
        scores['num_calls'] = self._num_calls
        if reset:
            self.reset()
        return scores
