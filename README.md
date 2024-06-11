# Gradient Clipping Improves AdaGrad when the Noise Is Heavy-Tailed
[![arXiv](https://img.shields.io/badge/arXiv-2401.06766-b31b1b.svg)](https://www.arxiv.org/abs/2406.04443)

This code comes jointly with reference:

> Savelii Chezhegov, Yaroslav Klyukin, Andrei Semenov, Aleksandr Beznosikov, Alexander Gasnikov, Samuel Horvath, Martin Takac, Eduard Gorbunov. "Gradient Clipping Improves AdaGrad when the Noise Is Heavy-Tailed".

Date:    May 2024

## Requirements

**Packages.** 
Jupyter notebooks and PyTorch.

## Organization of the code

The code is divided into two parts:
- Codes for experiments with different versions of AdaGrad on synthetic quadratic problem are given in the folder "Quadratic problem".
- Codes for experiments with different versions of Adam on ALBERT Base model fine-tuning are given in the folder "ALBERT fine-tuning".

## How to install

```bash
git clone https://github.com/yaroslavkliukin/Clipped-AdaGrad-and-Adam
cd Clipped-AdaGrad-and-Adam
pip install -r requirements.txt
```

## How to run

The repository contains 2 main folders: "Quadratic problem" and "ALBERT fine-tuning". 
To validate the the results of the quadratic problem (similarly to Figures 1, 4, 5, 6), simply run the corresponding notebook ```heavy_tailed_convergence_demo.ipynb```.
To check the ALBERT fine-tuning experiments you firstly need to set the hyperparameters values. Depending on the task, go to the ```config_cola.yaml``` or ```config_rte.yaml``` in the ```configs``` subfolder.
We use the same set of hyperparameters for both CoLA and RTE datasets: 
- Optimizer hyperparameters (```opt```): ```lr```, ```betas```, ```eps```, ```weight_decay```, ```correct_bias```, ```clipping``` (use ```local```), ```max_grad_norm``` (i.e., clipping level), ```exp_avg_sq_value``` ($\epsilon$), ```etta``` ($\eta$).
- training hyperparameters (```train```): ```model_checkpoint``` (```albert-base-v2``` in all our experiments), ```max_epoch```, ```batch_size``` (we suggest to use the ```8``` for RTE, and ```16``` for CoLA), ```seed```, ```classifier_dropout```, ```val_check_interval``` (pick ```12``` for RTE and ```20``` for CoLA).
- ```data``` hyperparameters: ```task``` (use ```cola``` or ```rte```).

When you have picked all the hyperparameters, please run the following scripts in the "ALBERT fine-tuning" directory: (dataset_name == cola or rte):
1. To see how the selected hyperparameters affect training, run ```one_run_{dataset_name}.py```.
2. To conduct many experiments, use the ```multi_runs_{dataset_name}.py``` script.
3. To check the heavy tails during training, utilize ```check_tails_{dataset_name}.py```.
4. Finally, run ```visualization.ipynb``` to reproduce the Figures 2, 3, 7, and 8 from our work.

**We believe the details provided are clear enough to reproduce the experimental part of our paper.**

## Cite
```bib
@misc{chezhegov2024gradient,
      title={Gradient Clipping Improves AdaGrad when the Noise Is Heavy-Tailed}, 
      author={Savelii Chezhegov and Yaroslav Klyukin and Andrei Semenov and Aleksandr Beznosikov and Alexander Gasnikov and Samuel Horváth and Martin Takáč and Eduard Gorbunov},
      year={2024},
      eprint={2406.04443},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
