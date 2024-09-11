"""Base DataModule class."""

from pathlib import Path
from typing import Collection, Optional, Tuple

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader



def load_and_print_info(data_module_class) -> None:
    """Load dataset and print info."""
    dataset = data_module_class(None)
    dataset.prepare_data()
    dataset.setup()
    print(dataset)



BATCH_SIZE = 1
#NUM_AVAIL_CPUS = len(os.sched_getaffinity(0)) # For linux machine this code can be used
# For windows machine above code doesnt work, uncomment below code and intialize
NUM_AVAIL_CPUS = 1
NUM_AVAIL_GPUS = torch.cuda.device_count()

# sensible multiprocessing defaults: at most one worker per CPU
DEFAULT_NUM_WORKERS = NUM_AVAIL_CPUS
# but in distributed data parallel mode, we launch a training on each GPU, so must divide out to keep total at one worker per CPU
DEFAULT_NUM_WORKERS = NUM_AVAIL_CPUS // NUM_AVAIL_GPUS if NUM_AVAIL_GPUS else DEFAULT_NUM_WORKERS


class BaseDataModule(pl.LightningDataModule):
    """Base for all of our LightningDataModules.

    Learn more at about LDMs at https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
    """

    def __init__(self, opt) -> None:
        super().__init__()
        self.batch_size =  opt.batch_size if opt else BATCH_SIZE
        self.num_workers = DEFAULT_NUM_WORKERS

        self.on_gpu = None

        # Make sure to set the variables below in subclasses
        self.input_dims: Tuple[int, ...]
        self.output_dims: Tuple[int, ...]
        self.mapping: Collection
        self.data_train: torch.utils.data.Dataset # Type of dataset may vary according to problem
        self.data_val: torch.utils.data.Dataset
        self.data_test: torch.utils.data.Dataset

    @classmethod
    def data_dirname(cls):
        return Path(__file__).resolve().parents[2] / "data"


    def config(self):
        """Return important settings of the dataset, which will be passed to instantiate models."""
        return {}

    def prepare_data(self, *args, **kwargs) -> None:
        """Take the first steps to prepare data for use.

        Use this method to do things that might write to disk or that need to be done only from a single GPU
        in distributed settings (so don't set state `self.x = y`).
        """

    def setup(self, stage: Optional[str] = None) -> None:
        """Perform final setup to prepare data for consumption by DataLoader.

        Here is where we typically split into train, validation, and test. This is done once per GPU in a DDP setting.
        Should assign `torch Dataset` objects to self.data_train, self.data_val, and optionally self.data_test.
        """

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
            collate_fn=self._AlignCollate
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )