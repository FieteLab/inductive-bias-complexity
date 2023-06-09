﻿# Model-agnostic Measure of Generalization Difficulty

Code for "Model-agnostic Measure of Generalization Difficulty (ICML 2023)". 

### Setup

After cloning the repository, please run `pip install -r requirements.txt` to install the project's dependencies.

### Run
```
python3 task_difficulty.py
```

### Files

`task_difficulty.py` contains code to perform the following experiments:
* Task difficulty computation for Omniglot
* Task difficulty computation for image classification benchmarks
* Inductive bias information content for models achieving different error rates
* Task difficulty computation for simplified Cartpole task
* Task difficulty computation for MuJoCo tasks
* Task difficulty computation for task unions
* Task difficulty computation with a varying number of classes on ImageNet
* Task difficulty computation with a varying spatial resolution on ImageNet


## Citation
```
@inproceedings{boopathy2023model,
    author = {Boopathy, Akhilan and Liu, Kevin and Hwang, Jaedong and Ge, Shu and Mohammedsaleh, Asaad and Fiete, Ila},
    title = {Model-Agnostic Measure of Generalization Difficulty},
    booktitle = {ICML},
    year = {2023},
}   
```
