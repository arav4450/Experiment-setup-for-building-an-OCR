"""Basic LightningModules on which other modules can be built."""
from pathlib import Path

import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0,currentdir)

import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import StepLR
from utils import CTCLabelConverter, Accuracy

ONE_CYCLE_TOTAL_STEPS = 100


class BaseLitModel(pl.LightningModule):
    """
    Generic PyTorch-Lightning class that must be initialized with a PyTorch module.
    """

    def __init__(self,opt, model):
        super().__init__()
        self.save_hyperparameters()
        self.opt = opt
        self.model = model

        self.converter = CTCLabelConverter(self.opt.character)

        #optimizer
        optimizer = self.opt.optim
        self.optimizer_class = getattr(torch.optim, optimizer)

        # learning rate
        self.lr = self.opt.lr

        # loss function
        self.loss_fn = torch.nn.CTCLoss(zero_infinity=True)
        #loss_avg = Averager()

        # evaluation metric
        self.acc = Accuracy(self.converter)

   
    
    @classmethod
    def log_dirname(cls):
        return Path(__file__).resolve().parents[2] / "logs"
     

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        scheduler = StepLR(
            optimizer=optimizer, step_size = 20, gamma=0.1
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        # code for prediction
        
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, preds, preds_size, loss = self._run_on_batch(batch)
       
        self.train_acc = self.acc.calculate( y, preds, preds_size,x.size(0))

        self.log("train/loss", loss, on_step=False, on_epoch=True,logger=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True,logger=True)

        outputs = {"loss": loss}

        return outputs

    def _run_on_batch(self, batch, with_preds=False):
        image_tensors, labels = batch
        text, length = self.converter.encode(labels, batch_max_length=self.opt.batch_max_length)
        batch_size = image_tensors.size(0)
        
        preds = self([image_tensors, text]).log_softmax(2)
        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
        preds = preds.permute(1, 0, 2)
        loss = self.loss_fn(preds, text, preds_size, length)

        return image_tensors, labels, preds, preds_size, loss

    def validation_step(self, batch, batch_idx):

        x, y, preds, preds_size, loss = self._run_on_batch(batch)
       
        self.val_acc = self.acc.calculate( y, preds, preds_size,x.size(0))

        self.log("validation/loss", loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True,logger=True)
        self.log("validation/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True,logger=True)

        outputs = {"loss": loss}

        return outputs

    def test_step(self, batch, batch_idx):
        x, y, preds, preds_size, loss = self._run_on_batch(batch)
       
        self.test_acc = self.acc.calculate( y, preds, preds_size,x.size(0))

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True)