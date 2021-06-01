import json
import os
from typing import Any, Dict, List

from torch.optim import Optimizer

from allennlp.data.data_loaders import TensorDict
from allennlp.models.archival import CONFIG_NAME
from allennlp.training.trainer import GradientDescentTrainer, TrainerCallback

import logging
import json
logger = logging.getLogger(__name__)


@TrainerCallback.register("aim")
class AimLogger(TrainerCallback):
    def __init__(self, hparams: Dict[str, Any] = None, **kwargs) -> None:
        super().__init__(**kwargs)

        self.trainer: GradientDescentTrainer
        self.optimizer: Optimizer
        self.repo: str
        self.experiment_name: str
        self.config: Any

        self.hparams = hparams
        logger.info('Hparams:')
        logger.info(json.dumps({
            **hparams,
            'extras': None
        }, indent=4))
        logger.info('Extras:')
        logger.info(json.dumps(hparams.get('extras'), indent=4))

        from aim.sdk.session import Session

        self.session: Session = None

    def on_start(self, trainer: GradientDescentTrainer, is_primary: bool) -> None:
        self.trainer = trainer
        self.model = trainer.model
        self.optimizer = trainer.optimizer

        abspath = os.path.abspath(self.serialization_dir)
        abspath = os.path.normpath(abspath)
        self.repo, _, self.experiment_name = abspath.rpartition(os.sep + ".aim" + os.sep)
        assert self.repo and self.experiment_name

        config_path = os.path.join(self.serialization_dir, CONFIG_NAME)
        with open(config_path, "r") as f:
            self.config = json.load(f)

        from aim.sdk.session import Session

        self.session = Session(repo=self.repo, experiment=self.experiment_name)

        self.session.set_params({"hparams": self.hparams, **self.config}, name="config")
        self.session.set_params(self.hparams, name="hparams")

    def _log_learning_rates(self, epoch: int):
        names = {param: name for name, param in self.model.named_parameters()}
        for group in self.optimizer.param_groups:
            if "lr" not in group:
                continue
            rate = group["lr"]
            for param in group["params"]:
                # check whether params has requires grad or not
                effective_rate = rate * float(param.requires_grad)
                # self.add_train_scalar("learning_rate/" + names[param], effective_rate)
                key = names[param].replace(".", "_")
                self.session.track(
                    effective_rate, name=f"lr_{key}", epoch=epoch, subset="train", is_training=True
                )

    def on_batch(
        self,
        trainer: GradientDescentTrainer,
        batch_inputs: List[List[TensorDict]],
        batch_outputs: List[Dict[str, Any]],
        batch_metrics: Dict[str, Any],
        epoch: int,
        batch_number: int,
        is_training: bool,
        is_primary: bool,
    ) -> None:
        if not is_training:
            return
        if hasattr(trainer.model, "on_batch"):
            trainer.model.on_batch(
                epoch=epoch,
                batch_number=batch_number,
                is_training=is_training,
                is_primary=is_primary,
            )
        if batch_number % 100 != 0:
            return
        for name, value in batch_metrics.items():
            if not isinstance(value, (float, int)):
                # TODO maybe warn one time
                continue

            self.session.track(
                value, name=name, epoch=epoch, subset="train", is_training=is_training
            )

        self._log_learning_rates(epoch=epoch)

    def on_epoch(
        self, trainer: GradientDescentTrainer, metrics: Dict[str, Any], epoch: int, is_primary: bool
    ) -> None:
        pass  # TODO epoch == -1 do nothing
        # 'best_epoch':0
        # 'peak_worker_0_memory_MB':3981.296875
        # 'peak_gpu_0_memory_MB':478.98828125
        # 'training_duration':'0:09:47.652715'  # TODO
        # 'training_start_epoch':0
        # 'training_epochs':0
        # 'epoch':0
        # 'training_accuracy':0.4024613506916192
        # 'training_loss':1.0746151166302818
        # 'training_worker_0_memory_MB':3981.296875
        # 'training_gpu_0_memory_MB':478.98828125
        # 'validation_accuracy':0.5105450840550179
        # 'validation_loss':0.9938879886718646
        # 'best_validation_accuracy':0.5105450840550179
        # 'best_validation_loss':0.9938879886718646
        for key, value in metrics.items():
            if key.startswith("best_"):
                _, _, key = key.partition("best_")
                is_best = True
            else:
                is_best = False

            if key.startswith("training_"):
                _, _, key = key.partition("training_")
                subset = "train"
            elif key.startswith("validation_"):
                _, _, key = key.partition("validation_")
                subset = "val"
            else:
                subset = None

            if not isinstance(value, (float, int)):
                # TODO maybe warn one time
                continue

            if is_best:
                key = f"best_{key}"

            self.session.track(value, name=key, epoch=epoch, subset=subset, is_training=False)

    def on_end(
        self,
        trainer: "GradientDescentTrainer",
        metrics: Dict[str, Any],
        epoch: int,
        is_primary: bool,
    ) -> None:
        self.session.close()
