"""Experiment-running framework."""

import numpy as np
import pytorch_lightning as pl
import torch 


import os, sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

import lit_model,data, model
import yaml

#config file
from util import get_config,AttrDict

# In order to ensure reproducible experiments, we must set random seeds.
np.random.seed(42)
torch.manual_seed(42)



def main():
    """
    Run an experiment.

    command:
    ```
    python training/run_experiment.py 
    ```
    """
   
    opt = get_config("config.yaml")

    data_obj = data.datamodule(opt)

    model_obj = model.Model(opt)
       
    lit_model_class = lit_model.BaseLitModel
    
    if opt.load_checkpoint is not None:
        with open('best_model_path.yaml', 'r', encoding="utf8") as stream:
            model_path = yaml.safe_load(stream)
            model_path = AttrDict(model_path)
            if model_path.path != '':
                 lit_mode = lit_model_class.load_from_checkpoint(model_path.path, opt = opt, model=model_obj)
    else:
        lit_mode = lit_model_class(opt=opt, model=model_obj)
    
    lit_mode = lit_model_class(opt=opt, model=model_obj)
    log_dir = lit_mode.log_dirname()
    logger = pl.loggers.TensorBoardLogger(log_dir)
    experiment_dir = logger.log_dir

    goldstar_metric = "validation/loss"
    filename_format = "epoch={epoch:04d}-validation.loss={validation/loss:.3f}"
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=5,
        filename=filename_format,
        monitor=goldstar_metric,
        mode="min",
        auto_insert_metric_name=False,
        dirpath=experiment_dir,
        every_n_epochs= 1, # can be passed as an argument
    )

    summary_callback = pl.callbacks.ModelSummary(max_depth=2)

    callbacks = [summary_callback, checkpoint_callback]
    if opt.stop_early:
        early_stopping_callback = pl.callbacks.EarlyStopping(
            monitor="validation/loss", mode="min", patience=opt.stop_early
        )
        callbacks.append(early_stopping_callback)

    trainer = pl.Trainer( callbacks=callbacks, logger=logger, accelerator = opt.accelerator, max_epochs = opt.max_epochs, devices = opt.devices,gradient_clip_val=opt.grad_clip, gradient_clip_algorithm="norm")

    trainer.fit(lit_mode, datamodule=data_obj)

    best_model_path = { 'path' : checkpoint_callback.best_model_path } 

    with open('best_model_path.yaml', 'w') as outfile:
        yaml.dump(best_model_path, outfile)
    
    """"
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        trainer.test(datamodule=data, ckpt_path=best_model_path)
    else:
        trainer.test(lit_mode, datamodule=data)
    """

if __name__ == "__main__":
    main()