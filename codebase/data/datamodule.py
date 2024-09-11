# dataset class for image classfication task
import torch
from torch.utils.data import DataLoader

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0,currentdir)


from base_data_module import BaseDataModule, load_and_print_info


#other functions
from dataset import AlignCollate, OCRDataset

# Directory for downloading and storing data
DATA_DIRNAME = BaseDataModule.data_dirname() 

#not required 
"""
def _download_data():
    

"""

class datamodule(BaseDataModule):
    """Image DataModule."""

    def __init__(self, opt) -> None:
        super().__init__(opt)

        self.opt = opt
        
        self.train_transforms = None
        self.val_transforms = None
        if self.opt:
            self._AlignCollate = AlignCollate(imgH=self.opt.imgH, imgW=self.opt.imgW, keep_ratio_with_pad=self.opt.PAD, contrast_adjust = self.opt.contrast_adjust)
        else:
            self._AlignCollate = AlignCollate()

    def prepare_data(self, *args, **kwargs) -> None:
        """Download data."""
        #_download_data()

    def setup(self, stage=None) -> None:
        """Split into train, val, test, and set dims."""
        self.data_train:torch.utils.data.Dataset = OCRDataset(
                                        root=DATA_DIRNAME/'train',opt = self.opt
                                    )
        
        self.data_val:torch.utils.data.Dataset = OCRDataset(
                                        root=DATA_DIRNAME/'val',opt = self.opt
                                    )

        self.data_test:torch.utils.data.Dataset = {}


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
            collate_fn=self._AlignCollate
        )

    # change according to dataset properties 
    def __repr__(self):
        basic = f"Image Dataset\nNum classes: {len(self.mapping)}"
        if self.data_train is None and self.data_val is None and self.data_test is None:
            return basic
        
        x, y = next(iter(self.train_dataloader()))
        data = (
            f"Train/val/test sizes: {len(self.data_train)}, {len(self.data_val)}, {len(self.data_test)}\n"
            f"Batch x stats: {(x.shape, x.dtype)}\n"
            f"Batch y stats: {(y.shape, y.dtype)}\n"
        )
        return basic + data

if __name__ == "__main__":
    
    """
    load_and_print_info(datamodule)
    dm = datamodule(None)
    dm.setup()
    x, y = next(iter(dm.train_dataloader()))
    print(x,y)

    """






