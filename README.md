# Explainable algorithm for melanoma detection using deep neural networks and game theory

_________________

This is the Pytorch official implementation of the framework proposed in the paper [Computer Aided Diagnosis of Melanoma using Deep Neural Networks and Game Theory](https://www.mdpi.com/1422-0067/23/22/13838).
## Abstract figure

![Alt text](proposed_framework.png?raw=true "Framework")
## Dependency
The code is build with following main libraries
- [Pytorch](https://www.tensorflow.org) 1.11.0
- [Numpy](https://numpy.org/) 
- [Pandas](https://pandas.pydata.org/)
- [Sklearn](https://scikit-learn.org/stable/)
- [Matlab](https://ch.mathworks.com/fr/products/matlab.html)


You can install all dependencies with requirements.txt following the command:
```bash
pip install -r requirements.txt 
```


## Dataset
- ISIC2018 [ISIC2018](https://challenge2018.isic-archive.com/). The original data will be preprocessed by `/preprocessing/preprocessImageConstancy.m`.


## Pipelines use case
### 1. Prepare data and models
#### Creates and divides the original dataset into dataset subgroups in order to train the models on the different tasks (six binary tasks and one 3-class task). It also defines the backbones of all models. 
```bash
python -u prepare_data_and_models.py
```

### 2. Train all models per task (See the 'launch_jobs_scripts' folder to know the order of the different scripts to launch).

#### Train the 3-class model  
```bash
python -u train_isic_multi.py
```
#### Example for binary task bekVSmel
```bash
python train_isic_bin.py --task bekVSmel  
```
### 3. Generate result for all runs
```bash
Python analyse_result.py
```
### 4. Evaluate mean and std
```bash
Python evaluate_mean_and_std.py
```

## Reference

If you find our paper and repo useful, please cite as

```
@Article{ijms232213838,
AUTHOR = {Foahom Gouabou, Arthur Cartel and Collenne, Jules and Monnier, Jilliana and Iguernaissi, Rabah and Damoiseaux, Jean-Luc and Moudafi, Abdellatif and Merad, Djamal},
TITLE = {Computer Aided Diagnosis of Melanoma Using Deep Neural Networks and Game Theory: Application on Dermoscopic Images of Skin Lesions},
JOURNAL = {International Journal of Molecular Sciences},
VOLUME = {23},
YEAR = {2022},
NUMBER = {22},
ARTICLE-NUMBER = {13838},
URL = {https://www.mdpi.com/1422-0067/23/22/13838},
ISSN = {1422-0067},
DOI = {10.3390/ijms232213838}
}
