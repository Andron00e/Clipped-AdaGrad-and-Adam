# Gradient Clipping for Coping with Heavy-Tailed Noise in Neural Networks (Semester Project)

This code comes jointly with reference:

> Vsevolod Skorokhodov, Aleksandr Dremov, Andrei Semenov. "Gradient Clipping Improves AdaGrad when the Noise Is Heavy-Tailed".

Date:    Jun 2025

## Requirements

**Packages.** 
Jupyter notebooks and PyTorch.

## Organization of the code

The code is divided into two parts:
- Codes for experiments with different versions of AdaGrad on synthetic quadratic problem are given in the folder "Quadratic problem".
- Codes for experiments with different versions of Adam on RoBERTa Large and ALBERT Base models fine-tuning are given in the folder "ALBERT fine-tuning".
- Codes for experiments with different versions of Adam and SGD on the ResNet training problem.

## How to install

```bash
git clone https://github.com/yaroslavkliukin/Clipped-AdaGrad-and-Adam
cd Clipped-AdaGrad-and-Adam
pip install -r requirements.txt
```

## How to run

The repository contains 2 main folders: "Quadratic problem" and "ALBERT fine-tuning". 
To check the RoBERTa fine-tuning experiments you firstly need to set the hyperparameters values. Depending on the task, go to the ```config_cola.yaml``` or ```config_qnli.yaml``` in the ```configs``` subfolder.
We use the same set of hyperparameters for both CoLA and QNLI datasets: 
- Optimizer hyperparameters (```opt```): ```lr```, ```betas```, ```eps```, ```weight_decay```, ```correct_bias```, ```clipping``` (use ```local```), ```max_grad_norm``` (i.e., clipping level), ```exp_avg_sq_value``` ($\epsilon$), ```etta``` ($\eta$).
- training hyperparameters (```train```): ```model_checkpoint``` (```roberta-large```, ```albert-base-v2```), ```max_epoch```, ```batch_size``` (we suggest to use the ```8``` for RTE, and ```16``` for CoLA), ```seed```, ```classifier_dropout```, ```val_check_interval``` (pick ```12``` for RTE and ```20``` for CoLA).
- ```data``` hyperparameters: ```task``` (use ```cola``` or ```rte```).

When you have picked all the hyperparameters, please run the following scripts in the "ALBERT fine-tuning" directory: (dataset_name == cola, rte or qnli):
1. To see how the selected hyperparameters affect training, run ```one_run_{dataset_name}.py```.
2. To conduct many experiments, use the ```multi_runs_{dataset_name}.py``` script.
3. To check the heavy tails during training, utilize ```check_tails_{dataset_name}.py```.
4. Finally, run ```visualization.ipynb``` to reproduce the Figures from our report.

**We believe the details provided are clear enough to reproduce the experimental part of our project.**
