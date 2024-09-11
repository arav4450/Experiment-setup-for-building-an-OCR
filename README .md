# Experiment-setup-for-building-an-OCR

This repository contains an experiment set up for building an OCR system using pytorch and pytorch lightninng.

<ul>
  <li>Create a new environment using environment.yaml file. Install the dependencies using the dev.in file. Please refer to this https://github.com/arav4450/Structuring-ML-Project repository for details regarding this step.</li>
  <li>Use this https://github.com/Belval/TextRecognitionDataGenerator repository to build the dataset for the task. The data should be available under the data folder.</li>
  <li>The entire code is placed under the codebase folder and split in to<ul><li>data - dataset building</li><li>model - model implementation</li><li>lit_model - implementation of training, validation and test steps</li><li>training - code to run the experiment</li></ul>The parameters are specified in config file. Run <i>python run_experiment.py</i> to start the experiment</li>
  <li> If you prefer using notebook, then use the exp.ipynb file under the folder notebooks</li>
</ul>  



