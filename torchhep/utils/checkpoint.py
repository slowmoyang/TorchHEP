from typing import Optional
from pathlib import Path
import torch


class Checkpoint:
    def __init__(self,
                 directory: Path,
                 top_k: Optional[int] = 1,
                 **kwargs
    ) -> None:
        self.state = {}
        if len(kwargs) > 0:
            self.register(**kwargs)
        self.directory = directory.resolve()
        self.top_k = top_k

        # FIXME
        self._name_template = 'checkpoint_epoch-{epoch:05d}_loss-{loss:.6f}.pt'
        self.best_symlink = directory / "best_checkpoint.pt"

        # best
        # FIXME
        self.best_loss = float('inf')
        self.best_epoch = 0
        # self.best_path = None

    def register(self, **kwargs):
        # FIXME check duplicates
        stateless_objs = [key for key, value in kwargs.items() if not hasattr(value, 'state_dict')]
        if len(stateless_objs) > 0:
            raise RuntimeError(stateless_objs)
        self.state |= {key: value.state_dict() for key, value in kwargs.items()}

    def __call__(self, loss: float, epoch: int, **kwargs) -> None:
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_epoch = epoch

            state = self.state | kwargs
            torch.save(state, self.best_path)

            if self.best_symlink.exists():
                self.best_symlink.unlink()
            self.best_symlink.symlink_to(self.best_path)

    def make_path(self, epoch, loss):
        return self.directory / self._name_template.format(epoch=epoch, loss=loss)

    @property
    def best_path(self):
        return self.make_path(self.best_epoch, self.best_loss)

    @property
    def best_state_dict(self):
        # FIXME
        if self.best_path is None:
            # TODO warning
            return
        else:
            return torch.load(self.best_path)
