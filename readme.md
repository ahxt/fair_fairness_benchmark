# Fair Fairness Benchmark
## 1. Overview 
The Fair Fairness Benchmark is a PyTorch-based framework for evaluating the fairness of machine learning models. The framework is designed to be simple and customizable, making it accessible to researchers with varying levels of expertise. The benchmark includes a set of predefined fairness metrics and algorithms, but users can easily modify or add new metrics and algorithms to suit their specific research questions.

## 2. Our Goals

This bechmark aims to be

* minimalistic
* hackable
* beginner-friendly
* torch-idiomatic
* reference implementation for researchers
* ......


## 3. Fair Fairness Benchmark(FFB)
### 3.1 Datasets
- Adult
- German
- Compas
- Bank Marketing
- KDD
- ACS
- CelebA
- UTKFace


### 3.2 Backbone
- MLP
- ResNet

### 3.3 In-Processing Group Fairness Methods
- DiffDP
- DiffEOpp
- DiffEodd
- PrejudiceRemover
- HSIC
- Advers

### 3.4 Our Results




## 4 How to Run

### 4.1 Setup
To install the Fair Fairness Benchmark, simply clone this repository and install the required dependencies by running the following command:

```
pip install -r requirements.txt
```
### 4.2 Cmd Example
```
python -u ./ffb_tabular_erm.py --dataset acs --model erm --sensitive_attr age --target_attr income --batch_size 32 --seed 89793 --log_freq 1 --num_training_steps 150
python -u ./ffb_tabular_diffdp.py --dataset acs --model diffdp --sensitive_attr race --target_attr income --batch_size 4096 --lam 1.4 --seed 89793 --log_freq 1 --num_training_steps 150
wait;
```


## 5 Contributing
We welcome contributions from the research community to improve and extend the Fair Fairness Benchmark. If you have an idea for a new metric or algorithm, or would like to report a bug, please open an issue or submit a pull request.

## 6 License
The Fair Fairness Benchmark is released under the MIT License.