from typing import Optional
from typing import Union

class EarlyStopping:
    def __init__(self,
                 patience: int = 10,
                 verbose: bool = False,
    ) -> None:
        super().__init__()
        self.patience = patience
        self.verbose = verbose

        self.min_loss = float('inf')
        self.wait = 0

    def __call__(self, loss: float) -> bool:
        need_stop = False

        # TODO delta
        if loss < self.min_loss:
            self.min_loss = loss
            self.wait = 0 # reset
        else:
            self.wait += 1
            if self.wait >= self.patience:
                need_stop = True

        if self.verbose and self.wait > 0:
            print(f'wait / patience = {self.wait: 5d} / {self.patience: 5d}')
        return need_stop


