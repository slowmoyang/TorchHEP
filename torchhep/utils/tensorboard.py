from pathlib import Path
from enum import Enum
import pandas as pd
import torch
import torch.utils.tensorboard
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


class TensorBoardWriter(torch.utils.tensorboard.SummaryWriter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_step = 0

    def add_scalar_dict(self, phase, step, **kwargs):
        for key, value in kwargs.items():
            self.add_scalar(tag=f'{key}/{phase}', scalar_value=value,
                            global_step=step)

    def add_result(self, phase, step, metric_dict=None, **kwargs):
        if isinstance(phase, Enum):
            phase = phase.value
        if metric_dict is not None:
            for metric_name, metric in metric_dict.items():
                kwargs[metric_name] = metric.compute()
        return self.add_scalar_dict(phase=phase, step=step, **kwargs)

    def add_train_step_result(self, phase='train', metric_dict=None, **kwargs):
        self.global_step += 1
        return self.add_result(phase=phase, step=self.global_step,
                               metric_dict=metric_dict, **kwargs)

    def add_eval_result(self, phase, metric_dict=None, step=None, **kwargs):
        r"""add the validation or test results"""
        step = step or self.global_step
        return self.add_result(phase=phase, step=step, metric_dict=metric_dict,
                               **kwargs)

    def add_epoch(self, epoch, global_step=None, tag='epoch'):
        global_step = global_step or self.global_step
        self.add_scalar(tag=tag, scalar_value=epoch, global_step=global_step)

class TensorBoardReader:
    log_dir: Path

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.event_accumulator = EventAccumulator(str(log_dir)).Reload()

    @classmethod
    def from_summary_writer(cls, summary_writer):
        summary_writer.flush()
        return cls(summary_writer.get_logdir())

    def read_scalars(self, tag):
        return pd.DataFrame(self.event_accumulator.Scalars(tag))

    def read_single_scalar(self, tag):
        scalars = self.event_accumulator.Scalars(tag)
        assert len(scalars) == 1
        return scalars[0]

    def to_step(self, epoch):
        for _, row in self.read_scalars('epoch').iterrows():
            if row.value == epoch:
                return row.step
        else:
            raise RuntimeError
