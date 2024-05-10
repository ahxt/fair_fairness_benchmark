<p align="center">
<img width="600" align="left" src="./img/ffb_long.png">
</p>
<br><br>

##

## Update:
- [05/09/2024]: update processed Jiasaw datasets, refer to the [datasets/readme.md](./datasets/readme.md)
- [08/23/2023]: update all [running logs](#6-running-logs) from our $45,079$ experiments ($14,428$ GPU hours)!
- [08/18/2023]: add a jupyter notebook tutorial for running FFB!
- [08/18/2023]: add a step-by-step guideline for running FFB!
- [08/18/2023]: add NLP task - Jigsaw Toxic Comment Classification!
- [08/01/2023]: design a logo for FFB!
- [07/12/2023]: update the datasets and downloading instructions!


## 1. Overview 
The Fair Fairness Benchmark is a PyTorch-based framework for evaluating the fairness of machine learning models. The framework is designed to be simple and customizable, making it accessible to researchers with varying levels of expertise. The benchmark includes a set of predefined fairness metrics and algorithms, but users can easily modify or add new metrics and algorithms to suit their specific research questions. For more information, please refer to our paper [FFB: A Fair Fairness Benchmark for In-Processing Group Fairness Methods](https://arxiv.org/abs/2306.09468).


## 2. Our Goals

This benchmark aims to be

* **minimalistic**
* **hackable**
* **beginner-friendly**
* **torch-idiomatic**
* **reference implementation for researchers**
* ......


## 3. Fair Fairness Benchmark(FFB)
### 3.1 Datasets

Please refer to the [datasets/readme.md](./datasets/readme.md) for datasets downloading instructions.


- **UCI Adult**: U.S. census data predicting an individual's income over $50K using demographic and financial details.
- **COMPAS**: Criminal defendants' records used to predict recidivism within two years.
- **German Credit**: Information about credit applicants at a German bank used for credit risk rating prediction.
- **Bank Marketing**: Data from a Portuguese bank used to predict client subscription to term deposit.
- **ACS**: From the American Community Survey, used for multiple prediction tasks such as income and employment.
- **KDD Census**: Like UCI Adult but with more instances, used to predict if an individual’s income is over $50K.
- **CelebFaces Attributes**: 20k celebrity face images annotated with 40 binary labels of specific facial attributes.
- **UTKFace**: Over 20k face images from diverse ethnicities and ages, annotated with age, gender, and ethnicity.



The statistics of the datasets are as the following:

<p align="center">
<img width="600" src="./img/table3.png">




### 3.2 In-Processing Group Fairness Methods
- **ERM**: Standard machine learning method that minimizes the empirical risk of the training data. Serves as a common baseline for fairness methods.
- **DiffDP, DiffEopp, DiffEodd**: Gap regularization methods for demographic parity, equalized opportunity, and equalized odds. These fairness definitions cannot be optimized directly, but gap regularization is differentiable and can be optimized using gradient descent.
- **PRemover**: Aim to minimize the mutual information between the prediction accuracy and the sensitive attributes.
- **HSIC**: Minimizes the Hilbert-Schmidt Independence Criterion between the prediction accuracy and the sensitive attributes.
- **AdvDebias**: Learns a classifier that maximizes the prediction ability and simultaneously minimizes an adversary's ability to predict the sensitive attributes from the predictions.
- **LAFTR**: A fair representation learning method aiming to learn an intermediate representation that minimizes the classification loss, reconstruction error, and the adversary's ability to predict the sensitive attributes from the representation.


### 3.4 Our Results

**1. Not all widely used fairness datasets stably exhibit fairness issues.** We found that in some cases, the bias in these datasets is either not consistently present or its manifestation varies significantly. This finding indicates that relying on these datasets for fairness analysis might not always provide stable or reliable results.
<p align="center">
<img width="600" src="./img/table4.png">

**2.The utility-fairness performance of the current fairness method exhibits trade-offs.** We conduct experiments using various in-processing fairness methods and analyze the ability to adjust the trade-offs to cater to specific needs while maintaining a balance between accuracy and fairness.
<p align="center">
<img width="600" src="./img/tradeoffs.png">




## 4. How to Run
### 4.1 Setup
To install the Fair Fairness Benchmark, simply clone this repository and install the required dependencies by running the following command:

```
pip install -r requirements.txt
```
### 4.2 Run Example
```
python -u ./ffb_tabular_erm.py --dataset acs --model erm --sensitive_attr age --target_attr income --batch_size 32 --seed 89793 --log_freq 1 --num_training_steps 150
python -u ./ffb_tabular_diffdp.py --dataset acs --model diffdp --sensitive_attr race --target_attr income --batch_size 4096 --lam 1.4 --seed 89793 --log_freq 1 --num_training_steps 150
wait;
```

## 5. Step-by-Step Guideline for Running FFB

### Step 1: System Preparation
Ensure you have Anaconda or Miniconda installed on your system. If not, download and install from the official [Miniconda](https://docs.conda.io/en/latest/miniconda.html) site. The important python packages are:
```bash
pandas==1.5.3
torch==1.13.1+cu116
wandb==0.14.0
scikit-learn==1.2.2
tabulate==0.9.0
statsmodels==0.13.5
```

### Step 2: Clone the Repository
```bash
# Navigate to your preferred directory
cd path/to/your/directory

# Clone the repository from GitHub
git clone https://github.com/ahxt/fair_fairness_benchmark.git
```

### Step 3: Setting Up Conda Environment
```bash
# Navigate to the cloned directory
cd fair_fairness_benchmark

# Create a new conda environment
conda create --name ffb_env python=3.8

# Activate the environment
conda activate ffb_env

# Install required packages
pip install -r requirements.txt
```

### Step 4: Setting up wandb
Weights & Biases is a tool used for experiment tracking. Our code in the repository uses `wandb`. We highly recomend to use wandb for tracking, if not, you can just only delete the all lines of codes that inlcudes "wandb". Ples follow these steps to setup wandb.

```bash
# Install wandb
pip install wandb

# Login to your wandb account. If you don't have one, you'll be prompted to create it.
wandb login
```

### Step 5: Running the Code
Before running, ensure you've downloaded the necessary datasets as per the instructions in `datasets/readme.md`.

```bash
# Run the first example
python -u ./ffb_tabular_erm.py --dataset acs --model erm --sensitive_attr age --target_attr income --batch_size 32 --seed 89793 --log_freq 1 --num_training_steps 150

# Run the second example
python -u ./ffb_tabular_diffdp.py --dataset acs --model diffdp --sensitive_attr race --target_attr income --batch_size 4096 --lam 1.4 --seed 89793 --log_freq 1 --num_training_steps 150
```

## 6. Running Logs


|     Dataset (Method)       | name           | wandb logs | #Runs  | Wall_time/GPU_Hours |
|    :-------------          | :------------- |:------------- | :---- | :-------------- |
|    Tabular (ERM)           | exp1\.erm      |https://wandb.ai/fair_benchmark/exp1.erm      | 190   | 35              |
|    Tabular (Adv)           | exp1\.adv\_gr  |https://wandb.ai/fair_benchmark/exp1.adv_gr   | 2660  | 863             |
|    Image   (ERM)           | exp2\.erm      |https://wandb.ai/fair_benchmark/exp2.erm      | 720   | 166             |
|    Image   (DiffDP)        | exp2\.diffdp   |https://wandb.ai/fair_benchmark/exp2.diffdp   | 5040  | 1213            |
|    Image   (DiffEopp)      | exp2\.diffeopp |https://wandb.ai/fair_benchmark/exp2.diffeopp | 5040  | 1215            |
|    Tabular (ERM)           | exp1\.pr       |https://wandb.ai/fair_benchmark/exp1.pr       | 2850  | 312             |
|    Tabular (HSIC)          | exp1\.hsic     |https://wandb.ai/fair_benchmark/exp1.hsic     | 2850  | 749             |
|    Image   (ERM)           | exp2\.diffeodd |https://wandb.ai/fair_benchmark/exp2.diffeodd | 5040  | 2638            |
|    Tabular (ERM)           | exp1\.laftr    |https://wandb.ai/fair_benchmark/exp1.laftr    | 1902  | 199             |
|    Tabular (DiffEopp)      | exp1\.diffeopp |https://wandb.ai/fair_benchmark/exp1.diffeopp | 2660  | 533             |
|    Tabular (DiffEodd)      | exp1\.diffeodd |https://wandb.ai/fair_benchmark/exp1.diffeodd | 2664  | 529             |
|    Image   (Pr)            | exp2\.pr       |https://wandb.ai/fair_benchmark/exp2.pr       | 5402  | 2860            |
|    Tabular (DiffDP)        | exp1\.diffdp   |https://wandb.ai/fair_benchmark/exp1.diffdp   | 2660  | 523             |
|    Image   (HSIC)          | exp2\.hsic     |https://wandb.ai/fair_benchmark/exp2.hsic     | 5401  | 2593            |
|    Total                   | total          |https://wandb.ai/fair_benchmark/              | 45079 | 14428           |



## 7. Contributing
We welcome contributions from the research community to improve and extend the Fair Fairness Benchmark. If you have an idea for a new metric or algorithm, or would like to report a bug, please open an issue or submit a pull request.


## 8. License
The Fair Fairness Benchmark is released under the MIT License.

---
If you find our resources useful, please kindly cite our paper.

```bibtex
@misc{han2023ffb,
      title={FFB: A Fair Fairness Benchmark for In-Processing Group Fairness Methods}, 
      author={Xiaotian Han and Jianfeng Chi and Yu Chen and Qifan Wang and Han Zhao and Na Zou and Xia Hu},
      year={2023},
      eprint={2306.09468},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
