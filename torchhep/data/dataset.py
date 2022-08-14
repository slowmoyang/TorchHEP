import abc
from typing import Optional, Union, Any
from pathlib import Path

import uproot
from torch.utils.data import Dataset
import tqdm


class ROOTDataset(Dataset, metaclass=abc.ABCMeta):

    def __init__(self,
                 paths: dict[Union[str, Path], str],
                 expressions: list[str],
                 cut: Optional[str] = None,
                 num_workers: int = 1,
                 step_size: str = '100 MB',
    ) -> None:
        self.paths = paths
        self.expressions = expressions
        self.cut = cut

        if num_workers == 1:
            file_handler = uproot.MemmapSource
        elif num_workers > 1:
            file_handler = uproot.MultithreadedFileSource
        else:
            raise ValueError(f'num_workers={num_workers}')

        tree_iter = uproot.iterate(paths,
                                   expressions=expressions,
                                   cut=cut,
                                   library='np',
                                   report=True,
                                   step_size=step_size,
                                   num_workers=num_workers,
                                   file_handler=file_handler)

        # TODO verbose
        cls_name = self.__class__.__name__
        total = sum(each for _, _, each in uproot.num_entries(paths))

        def set_description(pbar, done):
            progress = 100 * done / total
            description = f'[{cls_name}] {done} / {total} ({progress:.2f} %)'
            pbar.set_description(description)

        self._examples = []

        pbar = tqdm.tqdm(tree_iter)
        set_description(pbar, 0)
        for chunk, report in pbar:
            self._examples += self.process(chunk)
            set_description(pbar, report.stop)

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index):
        return self._examples[index]

    @abc.abstractmethod
    def process(self, chunk) -> list[Any]:
        ...

    @abc.abstractmethod
    def collate(self, batch):
        ...

    def apply(self, transform):
        self._examples = list(map(transform, self._examples))

    # TODO
    # def __add__(self, other):

# TODO
# def random_split():
