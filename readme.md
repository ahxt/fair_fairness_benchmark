# Fair Fairness Benchmark

## 1. Overview 
The Fair Fairness Benchmark is a PyTorch-based framework for evaluating the fairness of machine learning models. The framework is designed to be simple and customizable, making it accessible to researchers with varying levels of expertise. The benchmark includes a set of predefined fairness metrics and algorithms, but users can easily modify or add new metrics and algorithms to suit their specific research questions.

## 2. Our Goals

This bechmark aims to be

* **minimalistic**
* **hackable**
* **beginner-friendly**
* **torch-idiomatic**
* **reference implementation for researchers**
* ......


## 3. Fair Fairness Benchmark(FFB)
### 3.1 Datasets
- **UCI Adult**: U.S. census data predicting an individual's income over $50K using demographic and financial details.
- **COMPAS**: Criminal defendants' records used to predict recidivism within two years.
- **German Credit**: Information about credit applicants at a German bank used for credit risk rating prediction.
- **Bank Marketing**: Data from a Portuguese bank used to predict client subscription to term deposit.
- **ACS**: From the American Community Survey, used for multiple prediction tasks such as income and employment.
- **KDD Census**: Like UCI Adult but with more instances, used to predict if an individual’s income is over $50K.
- **CelebFaces Attributes**: 20k celebrity face images annotated with 40 binary labels of specific facial attributes.
- **UTKFace**: Over 20k face images from diverse ethnicities and ages, annotated with age, gender, and ethnicity.

The statistics of the datasets are as the following:

| Dataset   | Task | SensAttr | #Instances | #nFeat | #cFeat | #allFeat | y0:y1 | s0:s1 (1st) | s0:s1 (2nd) |
| --------- | ---- | -------- | ---------- | ------ | ------ | -------- | ---- | ----------- | ----------- |
| UCI Adult | income | gender, race | 45,222 | 7 | 5 | 101 | 1:0.33 | 1:2.08 | 1:9.20 |
| German    | credit | gender, age | 1,000 | 13 | 6 | 58 | 1:2.33 | 1:2.23 | 1:4.26 |
| KDD Census| income | gender, race | 292,550 | 32 | 8 | 509 | 1:14.76 | 1:0.92 | 1:8.14 |
| COMPAS    | credit | age | 6,172 | 400 | 5 | 405 | 1:0.83 | 1:4.25 | --- |
| Bank      | credit | gender, race | 41,188 | 10 | 9 | 62 | 1:0.13 | 1:37.58 | 1:37.58 |
| ACS-I      | income | gender, race | 195,665 | 8 | 1 | 908 | 1:0.70 | 1:0.89 | 1:1.62 |
| ACS-E      | employment | gender, race | 378,817 | 15 | 0 | 187 | 1:0.84 | 1:1.03 | 1:1.59 |
| ACS-P      | public | gender, race | 138,554 | 18 | 0 | 1696 | 1:0.58 | 1:1.27 | 1:1.31 |
| ACS-M      | mobility | gender, race | 80,329 | 20 | 0 | 2678 | 1:3.26 | 1:0.95 | 1:1.32 |
| ACS-T      | traveltime | gender, race | 172,508 | 15 | 0 | 1567 | 1:0.94 | 1:0.89 | 1:1.61 |
| CelebA-A  | attractive | gender, age | 202,599 | --- | --- | 48x48 | 1:0.95 | 1:1.40 | 1:0.29 |
| CelebA-W  | wavy hair | gender, age | 202,599 | --- | --- | 48x48 | 1:2.13 | 1:1.40 | 1:0.29 |
| CelebA-S  | smiling | gender, age | 202,599 | --- | --- | 48x48 | 1:1.07 | 1:1.40 | 1:0.29 |
| UTKFace   | age | gender, race | 23,705 | --- | --- | 48x48 | 1:1.15 | 1:1.10 | 1:1.35 |



### 3.2 In-Processing Group Fairness Methods
- **ERM**: Standard machine learning method that minimizes the empirical risk of the training data. Serves as a common baseline for fairness methods.
- **DiffDP, DiffEopp, DiffEodd**: Gap regularization methods for demographic parity, equalized opportunity, and equalized odds. These fairness definitions cannot be optimized directly, but gap regularization is differentiable and can be optimized using gradient descent.
- **PRemover**: Aims to minimize the mutual information between the prediction accuracy and the sensitive attributes.
- **HSIC**: Minimizes the Hilbert-Schmidt Independence Criterion between the prediction accuracy and the sensitive attributes.
- **AdvDebias**: Learns a classifier that maximizes the prediction ability and simultaneously minimizes an adversary's ability to predict the sensitive attributes from the predictions.
- **LAFTR**: A fair representation learning method aiming to learn an intermediate representation that minimizes the classification loss, reconstruction error, and the adversary's ability to predict the sensitive attributes from the representation.


### 3.4 Our Results

| Dataset | SenAttr | \accName | \aucName | \apName | \foName | \dpName | \abccName | \pruleName | \eoddName | \eoppName | Bias? |
|---------|---------|----------|----------|---------|---------|---------|-----------|------------|-----------|-----------|-------|
| Bank (strikethrough) | Age (strikethrough) | 91.17(0.57) | 94.05(0.19) | 62.41(1.83) | 54.31(11.14) | 10.88(4.27) | 10.64(1.63) | 44.48(5.95) | 10.71(5.68) | 6.16(4.90) | No (blue) |
|  | Gender (strikethrough) | 75.42(2.03) | 78.55(1.97) | 89.21(1.06) | 83.02(1.54) | 7.36(4.35) | 4.89(1.77) | 90.47(5.74) | 14.45(11.55) | 2.74(1.76) | No (blue) |
| German (strikethrough) | Age (strikethrough) | 75.19(2.16) | 77.21(2.60) | 88.28(1.74) | 82.90(1.62) | 12.20(6.02) | 10.01(1.63) | 84.44(7.17) | 17.97(10.71) | 8.26(6.12) | No (blue) |
| UCI Adult | Gender | 85.35(0.34) | 91.06(0.34) | 78.50(0.72) | 66.78(0.75) | 16.67(0.69) | 18.36(0.71) | 32.54(2.62) | 14.16(3.12) | 7.93(2.88) | Yes (red) |
|  | Race | 85.21(0.27) | 91.10(0.16) | 78.65(0.35) | 66.85(0.46) | 12.23(0.72) | 12.59(0.60) | 41.54(2.68) | 13.12(2.65) | 8.81(2.93) | Yes (red) |
| Compas | Gender | 67.07(0.80) | 72.56(0.74) | 67.99(0.93) | 59.77(2.27) | 13.43(2.48) | 5.80(1.12) | 65.12(8.25) | 19.67(6.02) | 11.54(4.73) | Yes (red) |
|  | Race | 67.13(1.06) | 72.98(0.59) | 68.24(0.72) | 60.58(3.06) | 


|Dataset          |SenAttr|accName|aucName|apName|foName|dpName|abccName|pruleName|eoddName|eoppName|Bias?| 
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| bank   |Age|91.17 ± 0.57|94.05 ± 0.19|62.41 ± 1.83|54.31 ± 11.14|10.88 ± 4.27|10.64 ± 1.63|44.48 ± 5.95|10.71 ± 5.68|6.16 ± 4.90|:x:|
|   |Gender|75.42 ± 2.03|78.55 ± 1.97|89.21 ± 1.06|83.02 ± 1.54|7.36 ± 4.35|4.89 ± 1.77|90.47 ± 5.74|14.45 ± 11.55|2.74 ± 1.76|:x:| 
|german|Age|75.19 ± 2.16|77.21 ± 2.60|88.28 ± 1.74|82.90 ± 1.62|12.20 ± 6.02|10.01 ± 1.63|84.44 ± 7.17|17.97 ± 10.71|8.26 ± 6.12|:x:|
|uciadult|Gender|85.35 ± 0.34|91.06 ± 0.34|78.50 ± 0.72|66.78 ± 0.75|16.67 ± 0.69|18.36 ± 0.71|32.54 ± 2.62|14.16 ± 3.12|7.93 ± 2.88|:white_check_mark:|
||Race|85.21 ± 0.27|91.10 ± 0.16|78.65 ± 0.35|66.85 ± 0.46|12.23 ± 0.72|12.59 ± 0.60|41.54 ± 2.68|13.12 ± 2.65|8.81 ± 2.93|:white_check_mark:|
|compas|Gender|67.07 ± 0.80|72.56 ± 0.74|67.99 ± 0.93|59.77 ± 2.27|13.43 ± 2.48|5.80 ± 1.12|65.12 ± 8.25|19.67 ± 6.02|11.54 ± 4.73|:white_check_mark:|
||Race|67.13 ± 1.06|72.98 ± 0.59|68.24 ± 0.72|60.58 ± 3.06|16.83 ± 3.48|8.15 ± 1.12|61.83 ± 4.56|29.03 ± 6.66|20.05 ± 3.95|:white_check_mark:|
|kddcensus|Gender|94.88 ± 0.48|94.03 ± 0.04|99.55 ± 0.00|97.32 ± 0.24|3.61 ± 1.60|5.20 ± 0.35|96.35 ± 1.62|14.97 ± 7.11|0.77 ± 0.37|:white_check_mark:| 
||Race|94.49 ± 0.78|94.40 ± 0.08|99.57 ± 0.01|97.13 ± 0.38|1.35 ± 1.17|3.31 ± 0.15|98.64 ± 1.18|6.56 ± 5.83|0.28 ± 0.25|:x:|
|acs-i|Gender|82.30 ± 0.12|90.28 ± 0.09|86.02 ± 0.15|77.91 ± 0.19|9.10 ± 0.31|8.27 ± 0.24|79.01 ± 0.65|3.38 ± 0.61|1.75 ± 0.41|:white_check_mark:|
||Race|82.40 ± 0.09|90.40 ± 0.09|86.17 ± 0.14|78.11 ± 0.11|9.81 ± 0.39|7.71 ± 0.29|77.24 ± 0.82|9.72 ± 0.73|6.21 ± 0.48|:white_check_mark:|
|acs-e|Gender|81.63 ± 0.12|88.95 ± 0.08|83.12 ± 0.12|81.31 ± 0.16|-|0.56 ± 0.12|98.87 ± 0.37|10.77 ± 0.20|0.90 ± 0.18|:x:| 
||Race|81.99 ± 0.16|90.00 ± 0.13|85.58 ± 0.24|81.38 ± 0.11|1.42 ± 0.35|0.99 ± 0.11|97.29 ± 0.62|3.48 ± 0.82|2.19 ± 0.45|:x:|
|acs-p|Gender|71.92 ± 0.18|75.25 ± 0.16|67.23 ± 0.22|52.93 ± 0.71|2.09 ± 0.64|2.35 ± 0.16|91.26 ± 2.52|2.30 ± 1.32|1.52 ± 1.00|:x:|
||Race|71.70 ± 0.22|75.00 ± 0.31|67.01 ± 0.28|52.06 ± 0.54|0.48 ± 0.32|1.98 ± 0.20|97.87 ± 1.36|4.63 ± 0.38|4.03 ± 0.72|:x:|
|acs-m|Gender|76.81 ± 0.32|72.85 ± 0.36|88.40 ± 0.22|86.54 ± 0.31|0.18 ± 0.17|0.84 ± 0.12 |99.80 ± 0.19|0.45 ± 0.51|0.08 ± 0.09|:x:|
||Race |76.98 ± 0.65|73.23 ± 0.61|88.53 ± 0.27|86.70 ± 0.04|0.11 ± 0.17|1.15 ± 0.17|99.88 ± 0.18 |0.82 ± 1.13|0.18 ± 0.29|:x:|  
|acs-t|Gender|66.36 ± 0.22|73.54 ± 0.18|71.59 ± 0.14|66.51 ± 0.31|8.60 ± 0.45|5.02 ± 0.28|84.65 ± 0.74|12.90 ± 0.82|5.72 ± 0.44|:white_check_mark:|
||Race|66.45 ± 0.20|73.64 ± 0.17|71.67 ± 0.19|66.26 ± 0.47|9.62 ± 0.67|6.07 ± 0.22 |83.09 ± 0.92|15.19 ± 1.24|6.50 ± 0.99|:white_check_mark:|
|celeba-a|Gender|78.19 ± 0.44|86.67 ± 0.53|86.66 ± 0.64|79.17 ± 0.48|52.39 ± 1.27|37.67 ± 0.98|30.42 ± 1.23|70.84 ± 3.15|35.53 ± 1.82|:white_check_mark:| 
||Race|78.19 ± 0.44|86.67 ± 0.53|86.66 ± 0.64|79.17 ± 0.47|41.90 ± 1.03|31.15 ± 1.17|33.43 ± 1.71|37.42 ± 2.26|18.83 ± 1.94|:white_check_mark:|
|celeba-w|Gender|82.50 ± 0.76|88.38 ± 0.86|80.38 ± 1.57|70.14 ± 1.64|33.92 ± 1.35 |29.52 ± 1.12|16.89 ± 2.01|52.71 ± 4.28|39.62 ± 3.49|:white_check_mark:|
||Race|82.50 ± 0.76|88.38 ± 0.86|80.38 ± 1.57|70.14 ± 1.64|10.27 ± 0.71|10.61 ± 0.47|64.59 ± 2.35|10.63 ± 2.18|6.48 ± 1.94|:white_check_mark:|
|celeba-s|Gender|89.95 ± 3.40|96.51 ± 1.84|96.67 ± 1.80|89.08 ± 6.79|14.09 ± 1.08|13.02 ± 1.46|72.76 ± 3.44|6.99 ± 1.16|6.51 ± 1.06|:white_check_mark:|
||Race|89.95 ± 3.40|96.51 ± 1.84|96.67 ± 1.80|89.08 ± 6.79|5.91 ± 0.75|5.59 ± 0.70|88.21 ± 2.50|6.46 ± 1.06|0.82 ± 0.80|:x:|
|utkface|Gender|83.34 ± 0.71|91.78 ± 0.57|91.36 ± 0.67|81.66 ± 0.85|25.68 ± 1.88|20.57 ± 1.23|54.61 ± 2.54|28.66 ± 4.12|17.16 ± 2.63|:white_check_mark:|
||Race|83.34 ± 0.71|91.78 ± 0.57|91.36 ± 0.67|81.66 ± 0.85|23.25 ± 1.71|18.99 ± 1.22|59.67 ± 2.63|23.07 ± 3.64|16.68 ± 2.70|:white_check_mark:|



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