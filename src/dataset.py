from tkinter import S
import folktables
from folktables import ACSDataSource
import pandas as pd
import os
import numpy as np
import pickle
from torch.utils.data import TensorDataset
import torch

import pdb


import numpy as np
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def _quantization_binning(data, num_bins=10):
        qtls = np.arange(0.0, 1.0 + 1 / num_bins, 1 / num_bins)
        bin_edges = np.quantile(data, qtls, axis=0)  # (num_bins + 1, num_features)
        bin_widths = np.diff(bin_edges, axis=0)
        bin_centers = bin_edges[:-1] + bin_widths / 2  # ()
        return bin_edges, bin_centers, bin_widths

def _quantize(inputs, bin_edges, num_bins=10):
        quant_inputs = np.zeros(inputs.shape[0])
        for i, x in enumerate(inputs):
            quant_inputs[i] = np.digitize(x, bin_edges)
        quant_inputs = quant_inputs.clip(1, num_bins) - 1  # Clip edges
        return quant_inputs

def _one_hot(a, num_bins=10):
    return np.squeeze(np.eye(num_bins)[a.reshape(-1).astype(np.int32)])

def DataQuantize(X, bin_edges=None, num_bins=10):
    '''
    Quantize: First 4 entries are continuos, and the rest are binary
    '''
    X_ = []
    for i in range(5):
        if bin_edges is not None:
            Xi_q = _quantize(X[:, i], bin_edges, num_bins)
        else:
            bin_edges, bin_centers, bin_widths = _quantization_binning(X[:, i], num_bins)
            Xi_q = _quantize(X[:, i], bin_edges, num_bins)
        Xi_q = _one_hot(Xi_q, num_bins)
        X_.append(Xi_q)

    for i in range(5, len(X[0])):
        if i == 39:     # gender attribute
            continue
        Xi_q = _one_hot(X[:, i], num_bins=2)
        X_.append(Xi_q)

    return np.concatenate(X_,1), bin_edges





def load_adult_data(path, sensitive_attributes = "sex"):
    '''
    We borrow the code from https://github.com/IBM/sensitive-subspace-robustness
    Preprocess the adult data set by removing some features and put adult data into a BinaryLabelDataset
    You need to download the adult dataset (both the adult.data and adult.test files) from https://archive.ics.uci.edu/ml/datasets/Adult
    '''

    headers = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-stataus', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'y']



    # train = pd.read_csv(os.path.join(path, "adult.data"), header = None, na_values="?", sep=r'\s*,\s*', engine='python')
    # test = pd.read_csv(os.path.join(path, "adult.test"), header = None,  na_values="?", sep=r'\s*,\s*', engine='python')

    train = pd.read_csv(os.path.join(path, "adult.data"), header = None, sep=r'\s*,\s*', engine='python')
    test = pd.read_csv(os.path.join(path, "adult.test"), header = None,  sep=r'\s*,\s*', engine='python', skiprows=1)


    df = pd.concat([train, test], ignore_index=True)
    df.columns = headers

    df['y'] = df['y'].replace({'<=50K.': 0, '>50K.': 1, '>50K': 1, '<=50K': 0 })

    df = df.drop(df[(df[headers[-2]] == ' ?') | (df[headers[6]] == ' ?')].index)
    df = pd.get_dummies(df, columns=[headers[1], headers[5], headers[6], headers[7], headers[9], headers[8], 'native-country'])

    # print( df.columns )

    delete_these = ['race_Amer-Indian-Eskimo','race_Asian-Pac-Islander','race_Black','race_Other', 'sex_Female']

    delete_these += ['native-country_Cambodia', 'native-country_Canada', 'native-country_China', 'native-country_Columbia', 'native-country_Cuba', 'native-country_Dominican-Republic', 'native-country_Ecuador', 'native-country_El-Salvador', 'native-country_England', 'native-country_France', 'native-country_Germany', 'native-country_Greece', 'native-country_Guatemala', 'native-country_Haiti', 'native-country_Holand-Netherlands', 'native-country_Honduras', 'native-country_Hong', 'native-country_Hungary', 'native-country_India', 'native-country_Iran', 'native-country_Ireland', 'native-country_Italy', 'native-country_Jamaica', 'native-country_Japan', 'native-country_Laos', 'native-country_Mexico', 'native-country_Nicaragua', 'native-country_Outlying-US(Guam-USVI-etc)', 'native-country_Peru', 'native-country_Philippines', 'native-country_Poland', 'native-country_Portugal', 'native-country_Puerto-Rico', 'native-country_Scotland', 'native-country_South', 'native-country_Taiwan', 'native-country_Thailand', 'native-country_Trinadad&Tobago', 'native-country_United-States', 'native-country_Vietnam', 'native-country_Yugoslavia']

    delete_these += ['fnlwgt', 'education']

    df.drop(delete_these, axis=1, inplace=True)

    # gender id = 39
    if sensitive_attributes == "sex":
        s = df["sex_Male"]
        df.drop(["sex_Male"], axis=1, inplace=True)
    elif sensitive_attributes == "race":
        s = df["race_White"]
        df.drop(["race_White"], axis=1, inplace=True)


    y = df["y"]
    df.drop(["y"], axis=1, inplace=True)

    # # C = np.delete(C, 1, 1) 
    # X_train = np.delete(X_train, 43, 1)
    # X_train = np.delete(X_train, 42, 1)

    # X_val = np.delete(X_val, 43, 1)
    # X_val = np.delete(X_val, 42, 1)

    # X_test = np.delete(X_test, 43, 1)
    # X_test = np.delete(X_test, 42, 1)

    return df, y, s



    # return BinaryLabelDataset(df = df, label_names = ['y'], protected_attribute_names = ['sex_Male', 'race_White'])





def get_adult_data(path):
    '''
    We borrow the code from https://github.com/IBM/sensitive-subspace-robustness
    Preprocess the adult data set by removing some features and put adult data into a BinaryLabelDataset
    You need to download the adult dataset (both the adult.data and adult.test files) from https://archive.ics.uci.edu/ml/datasets/Adult
    '''

    headers = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-stataus', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'y']



    # train = pd.read_csv(os.path.join(path, "adult.data"), header = None, na_values="?", sep=r'\s*,\s*', engine='python')
    # test = pd.read_csv(os.path.join(path, "adult.test"), header = None,  na_values="?", sep=r'\s*,\s*', engine='python')

    train = pd.read_csv(os.path.join(path, "adult.data"), header = None, sep=r'\s*,\s*', engine='python')
    test = pd.read_csv(os.path.join(path, "adult.test"), header = None,  sep=r'\s*,\s*', engine='python', skiprows=1)

    # input_data_train = pd.read_csv(os.path.join(path, "adult.data"), names=column_names,
    #                                na_values="?", sep=r'\s*,\s*', engine='python')
    # input_data_test = pd.read_csv(os.path.join(path, "adult.test"), names=column_names,
    #                               na_values="?", sep=r'\s*,\s*', engine='python', skiprows=1)

    df = pd.concat([train, test], ignore_index=True)
    df.columns = headers

    df['y'] = df['y'].replace({'<=50K.': 0, '>50K.': 1, '>50K': 1, '<=50K': 0 })

    df = df.drop(df[(df[headers[-2]] == ' ?') | (df[headers[6]] == ' ?')].index)
    df = pd.get_dummies(df, columns=[headers[1], headers[5], headers[6], headers[7], headers[9], headers[8], 'native-country'])

    # print( df.columns )

    delete_these = ['race_Amer-Indian-Eskimo','race_Asian-Pac-Islander','race_Black','race_Other', 'sex_Female']

    delete_these += ['native-country_Cambodia', 'native-country_Canada', 'native-country_China', 'native-country_Columbia', 'native-country_Cuba', 'native-country_Dominican-Republic', 'native-country_Ecuador', 'native-country_El-Salvador', 'native-country_England', 'native-country_France', 'native-country_Germany', 'native-country_Greece', 'native-country_Guatemala', 'native-country_Haiti', 'native-country_Holand-Netherlands', 'native-country_Honduras', 'native-country_Hong', 'native-country_Hungary', 'native-country_India', 'native-country_Iran', 'native-country_Ireland', 'native-country_Italy', 'native-country_Jamaica', 'native-country_Japan', 'native-country_Laos', 'native-country_Mexico', 'native-country_Nicaragua', 'native-country_Outlying-US(Guam-USVI-etc)', 'native-country_Peru', 'native-country_Philippines', 'native-country_Poland', 'native-country_Portugal', 'native-country_Puerto-Rico', 'native-country_Scotland', 'native-country_South', 'native-country_Taiwan', 'native-country_Thailand', 'native-country_Trinadad&Tobago', 'native-country_United-States', 'native-country_Vietnam', 'native-country_Yugoslavia']

    delete_these += ['fnlwgt', 'education']

    df.drop(delete_these, axis=1, inplace=True)



    return BinaryLabelDataset(df = df, label_names = ['y'], protected_attribute_names = ['sex_Male', 'race_White'])



def preprocess_adult_data(seed = 0, path = "", sensitive_attributes="sex"):
    '''
    Description: Ths code (1) standardizes the continuous features, (2) one hot encodes the categorical features, (3) splits into a train (80%) and test set (20%), (4) based on this data, create another copy where gender is deleted as a predictive feature and the feature we predict is gender (used by SenSR when learning the sensitive directions)
    Input: seed: the seed used to split data into train/test
    '''
    # Get the dataset and split into train and test
    dataset_orig = get_adult_data(path)
    # pdb.set_trace()

    # we will standardize continous features
    # continous_features = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    # continous_features_indices = [ dataset_orig.feature_names.index(feat) for feat in continous_features ]

    sex_features_indices = [dataset_orig.feature_names.index("sex_Male")]
    race_features_indices = [dataset_orig.feature_names.index("race_White")]

    # get a 80%/20% train/test split
    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.8], shuffle=True, seed = seed)
    # pdb.set_trace()


    X_train = dataset_orig_train.features
    X_test = dataset_orig_test.features

    y_train = dataset_orig_train.labels
    y_test = dataset_orig_test.labels

    X_val = X_train[:len(X_test)]
    y_val = y_train[:len(X_test)]
    
    X_train = X_train[len(X_test):]
    y_train = y_train[len(X_test):]

    # gender id = 39
    if sensitive_attributes == "sex":
        # gender id = 39
        # A_train = X_train[:,39]
        # A_val = X_val[:,39]
        # A_test = X_test[:,39]
        A_train = X_train[:, 42]
        A_val = X_val[:, 42]
        A_test = X_test[:, 42]
    elif sensitive_attributes == "race":
        # gender id = 39
        # A_train = X_train[:,40]
        # A_val = X_val[:,40]
        # A_test = X_test[:,40]
        A_train = X_train[:, 43]
        A_val = X_val[:, 43]
        A_test = X_test[:, 43]

    # C = np.delete(C, 1, 1) 
    X_train = np.delete(X_train, 43, 1)
    X_train = np.delete(X_train, 42, 1)

    X_val = np.delete(X_val, 43, 1)
    X_val = np.delete(X_val, 42, 1)

    X_test = np.delete(X_test, 43, 1)
    X_test = np.delete(X_test, 42, 1)


    # pdb.set_trace()

    # X_train, bin_edges = DataQuantize(X_train)
    # X_val, _ = DataQuantize(X_val, bin_edges)
    # X_test, _ = DataQuantize(X_test, bin_edges)

    # pdb.set_trace()


    return X_train, X_val, X_test, y_train, y_val, y_test, A_train, A_val, A_test





class PandasDataSet(TensorDataset):

    def __init__(self, *dataframes):
        tensors = (self._df_to_tensor(df) for df in dataframes)
        super(PandasDataSet, self).__init__(*tensors)

    def _df_to_tensor(self, df):
        if isinstance(df, pd.Series):
            df = df.to_frame('dummy')
        return torch.from_numpy(df.values).float()

# #new
# def load_adult_data(path, sensitive_attributes="sex"):
#     # | 48842 instances, mix of continuous and discrete(train=32561, test=16281)
#     # | 45222 if instances with unknown values are removed(train=30162, test=15060)
#     column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
#                     'martial_status', 'occupation', 'relationship', 'race', 'sex',
#                     'capital_gain', 'capital_loss', 'hours_per_week', 'country', 'target']

# #     input_data = (pd.read_csv(path, names=column_names,na_values="?", sep=r'\s*,\s*', engine='python')
# #                   .loc[lambda df: df['race'].isin(['White', 'Black'])])

#     input_data_train = pd.read_csv(os.path.join(path, "adult.data"), names=column_names,
#                                    na_values="?", sep=r'\s*,\s*', engine='python')
#     input_data_test = pd.read_csv(os.path.join(path, "adult.test"), names=column_names,
#                                   na_values="?", sep=r'\s*,\s*', engine='python', skiprows=1)

#     input_data = pd.concat([input_data_train, input_data_test])
#     # input_data = input_data.dropna()
#     # input_data = pd.read_csv(path, names=column_names, na_values="?", sep=r'\s*,\s*', engine='python')
#     # sensitive_attributes = "sex"

#     if sensitive_attributes == "race":
#         input_data = input_data[input_data['race'].isin(['White', 'Black'])]
#         s = input_data[sensitive_attributes][input_data['race'].isin(['White', 'Black'])]
#         s = (s == 'White').astype(int).to_frame()
#     else:
#         s = input_data[sensitive_attributes]
#         s = (s == 'Male').astype(int).to_frame()

#     # targets; 1 when someone makes over 50k , otherwise 0
#     y = (input_data['target'] == '>50K').astype(int)

#     # features; note that the 'target' and sentive attribute columns are dropped
#     X = (input_data
#          .drop(columns=['target', sensitive_attributes, 'fnlwgt'])
#          .fillna('Unknown')
#          .pipe(pd.get_dummies, drop_first=False))
#     return X, y, s





def load_german_data(path='/data/han/data/fairness/germen/', sensitive_attributes="sex"):
    input_data = pd.read_csv(os.path.join(path, 'german_credit_risk.csv'), na_values="NA", index_col=0)
    input_data["Job"] = input_data["Job"].astype(str)
    # data = data.fillna(0)
    # data = data.pipe(pd.get_dummies, drop_first=False)
    input_data.columns = input_data.columns.str.lower()

    # sensitive attribute
    s = input_data[sensitive_attributes]
    s = (s == 'male').astype(int).to_frame()

    # # targets; 1 , otherwise 0
    y = (input_data['class'] == 1).astype(int)

    # features; note that the 'target' and sentive attribute columns are dropped
    X = (input_data
         .drop(columns=['class', sensitive_attributes])
         .fillna('Unknown')
         .pipe(pd.get_dummies, drop_first=False))
    return X, y, s


# path = '/data/han/data/fairness/census-income-mld/'
# sensitive_attributes = "race"


def load_census_income_kdd_data(path='/ data/han/data/fairness/census-income-mld/', sensitive_attributes="sex"):
    data1 = pd.read_csv(os.path.join(path, 'census-income.data'), header=None,
                        names=["age", "workclass", "industry_code", "occupation_code", "education", "wage_per_hour",
                               "enrolled_in_edu_inst_last_wk", "marital_status", "major_industry_code",
                               "major_occupation_code", "race", "hispanic_origin", "sex", "member_of_a_labour_union",
                               "reason_for_unemployment", "employment_status", "capital_gains", "capital_losses",
                               "dividend_from_stocks", "tax_filler_status", "region_of_previous_residence",
                               "state_of_previous_residence", "detailed_household_and_family_stat",
                               "detailed_household_summary_in_household", "instance_weight", "migration_code_change_in_msa",
                               "migration_code_change_in_reg", "migration_code_move_within_reg",
                               "live_in_this_house_1_year_ag", "migration_prev_res_in_sunbelt",
                               "num_persons_worked_for_employer", "family_members_under_18", "country_of_birth_father",
                               "country_of_birth_mother", "country_of_birth_self", "citizenship",
                               "own_business_or_self_employed", "fill_inc_questionnaire_for_veteran's_admin",
                               "veterans_benefits", "weeks_worked_in_year", "year", "class"])
    data2 = pd.read_csv(os.path.join(path, 'census-income.test'), header=None,
                        names=["age", "workclass", "industry_code", "occupation_code", "education", "wage_per_hour",
                               "enrolled_in_edu_inst_last_wk", "marital_status", "major_industry_code",
                               "major_occupation_code", "race", "hispanic_origin", "sex", "member_of_a_labour_union",
                               "reason_for_unemployment", "employment_status", "capital_gains", "capital_losses",
                               "dividend_from_stocks", "tax_filler_status", "region_of_previous_residence",
                               "state_of_previous_residence", "detailed_household_and_family_stat",
                               "detailed_household_summary_in_household", "instance_weight", "migration_code_change_in_msa",
                               "migration_code_change_in_reg", "migration_code_move_within_reg",
                               "live_in_this_house_1_year_ag", "migration_prev_res_in_sunbelt",
                               "num_persons_worked_for_employer", "family_members_under_18", "country_of_birth_father",
                               "country_of_birth_mother", "country_of_birth_self", "citizenship",
                               "own_business_or_self_employed", "fill_inc_questionnaire_for_veteran's_admin",
                               "veterans_benefits", "weeks_worked_in_year", "year", "class"])

    input_data = pd.concat([data1, data2], ignore_index=True)
    input_data = input_data.drop_duplicates(keep='first', inplace=False)

    input_data.columns = input_data.columns.str.lower()

    if sensitive_attributes == "race":
        input_data = input_data[input_data['race'].isin([' White', ' Black'])]
        s = input_data[sensitive_attributes]
        s = (s == ' White').astype(int).to_frame()
    else:
        s = input_data[sensitive_attributes]
        s = (s == ' Male').astype(int).to_frame()

    # # targets; 1 , otherwise 0
    y = (input_data['class'] == " - 50000.").astype(int)

    # features; note that the 'target' and sentive attribute columns are dropped
    X = (input_data
         .drop(columns=['class', sensitive_attributes])
         .fillna('Unknown')
         .pipe(pd.get_dummies, drop_first=False))
    return X, y, s




def load_acs_dataset(task=None, sensitive_attributes="sex", survey_year = "2018", horizon= "1-Year", states = ["CA"]):
    if sensitive_attributes == "sex":
        # features = ['AGEP','SCHL','MAR','RELP','DIS','ESP','CIT','MIG','MIL','ANC','NATIVITY','DEAR','DEYE','DREM','SEX','RAC1P']
        features = ['AGEP','SCHL','MAR','RELP','DIS','ESP','CIT','MIG','MIL','ANC','NATIVITY','DEAR','DEYE','DREM','RAC1P']
        group = "SEX"
    elif sensitive_attributes == "race":
        # features = ['AGEP', 'SCHL', 'MAR', 'RELP', 'DIS', 'ESP', 'CIT', 'MIG', 'MIL', 'ANC', 'NATIVITY', 'DEAR', 'DEYE', 'DREM', 'SEX', 'RAC1P']
        features = ['AGEP', 'SCHL', 'MAR', 'RELP', 'DIS', 'ESP', 'CIT', 'MIG', 'MIL', 'ANC', 'NATIVITY', 'DEAR', 'DEYE', 'DREM', 'SEX']
        group = "RAC1P"
    else:
        # features = ['AGEP','SCHL','MAR','RELP','DIS','ESP','CIT','MIG','MIL','ANC','NATIVITY','DEAR','DEYE','DREM','SEX','RAC1P']
        features = ['AGEP','SCHL','MAR','RELP','DIS','ESP','CIT','MIG','MIL','ANC','NATIVITY','DEAR','DEYE','DREM','RAC1P']
        group = "SEX"


    ACSEmployment = folktables.BasicProblem(
        features=features,
        target='ESR',
        target_transform=lambda x: x == 1,
        group=group,
        preprocess=lambda x: x,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )


    data_source = ACSDataSource(survey_year=survey_year, horizon=horizon, survey='person',
                                root_dir="/data/han/data/fairness/folktables/")
    acs_data = data_source.get_data(states=states, download=True)
    features, label, group = ACSEmployment.df_to_numpy(acs_data)


    X = pd.DataFrame(features, columns=ACSEmployment.features)
    y = pd.Series(label)
    s = pd.Series(group, name=ACSEmployment.group).to_frame()
    if sensitive_attributes == "sex":
        s["SEX"] = s["SEX"] - 1
    elif sensitive_attributes == "race":
        s["RAC1P"] = (s["RAC1P"] == 1).astype(np.int)

    X = pd.get_dummies(X, columns=ACSEmployment.features)


    return X, y, s






def load_folktables_employment(task=None, sensitive_attributes="sex", survey_year = "2018", horizon= "1-Year", states = ["CA"]):
    if sensitive_attributes == "sex":
        # features = ['AGEP','SCHL','MAR','RELP','DIS','ESP','CIT','MIG','MIL','ANC','NATIVITY','DEAR','DEYE','DREM','SEX','RAC1P']
        features = ['AGEP','SCHL','MAR','RELP','DIS','ESP','CIT','MIG','MIL','ANC','NATIVITY','DEAR','DEYE','DREM','RAC1P']
        group = "SEX"
    elif sensitive_attributes == "race":
        # features = ['AGEP', 'SCHL', 'MAR', 'RELP', 'DIS', 'ESP', 'CIT', 'MIG', 'MIL', 'ANC', 'NATIVITY', 'DEAR', 'DEYE', 'DREM', 'SEX', 'RAC1P']
        features = ['AGEP', 'SCHL', 'MAR', 'RELP', 'DIS', 'ESP', 'CIT', 'MIG', 'MIL', 'ANC', 'NATIVITY', 'DEAR', 'DEYE', 'DREM', 'SEX']
        group = "RAC1P"
    else:
        # features = ['AGEP','SCHL','MAR','RELP','DIS','ESP','CIT','MIG','MIL','ANC','NATIVITY','DEAR','DEYE','DREM','SEX','RAC1P']
        features = ['AGEP','SCHL','MAR','RELP','DIS','ESP','CIT','MIG','MIL','ANC','NATIVITY','DEAR','DEYE','DREM','RAC1P']
        group = "SEX"


    ACSEmployment = folktables.BasicProblem(
        features=features,
        target='ESR',
        target_transform=lambda x: x == 1,
        group=group,
        preprocess=lambda x: x,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )


    data_source = ACSDataSource(survey_year=survey_year, horizon=horizon, survey='person',
                                root_dir="/data/han/data/fairness/folktables/")
    acs_data = data_source.get_data(states=states, download=True)
    features, label, group = ACSEmployment.df_to_numpy(acs_data)


    X = pd.DataFrame(features, columns=ACSEmployment.features)
    y = pd.Series(label)
    s = pd.Series(group, name=ACSEmployment.group).to_frame()
    if sensitive_attributes == "sex":
        s["SEX"] = s["SEX"] - 1
    elif sensitive_attributes == "race":
        s["RAC1P"] = (s["RAC1P"] == 1).astype(np.int)

    X = pd.get_dummies(X, columns=ACSEmployment.features)


    return X, y, s




def load_folktables_income(sensitive_attributes="sex", survey_year = "2018", horizon= "1-Year", states = ["CA"]):
    if sensitive_attributes == "sex":
        # features = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P']
        features = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'RAC1P']
        group = "SEX"
    elif sensitive_attributes == "race":
        # features = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P']
        features = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX']
        group = "RAC1P"
    else:
        features = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'RAC1P']
        group = "SEX"


    def adult_filter(data):
        """Mimic the filters in place for Adult data.

        Adult documentation notes: Extraction was done by Barry Becker from
        the 1994 Census database. A set of reasonably clean records was extracted
        using the following conditions:
        ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))
        """
        df = data
        df = df[df['AGEP'] > 16]
        df = df[df['PINCP'] > 100]
        df = df[df['WKHP'] > 0]
        df = df[df['PWGTP'] >= 1]
        return df


    ACSIncome = folktables.BasicProblem(
        features=features,
        target='PINCP',
        target_transform=lambda x: x > 50000,
        group=group,
        # preprocess=adult_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )

    data_source = ACSDataSource(survey_year=survey_year, horizon=horizon, survey='person',
                                root_dir="/data/han/data/fairness/folktables/")
    acs_data = data_source.get_data(states=states, download=True)
    features, label, group = ACSIncome.df_to_numpy(acs_data)

    X = pd.DataFrame(features, columns=ACSIncome.features)
    y = pd.Series(label)
    s = pd.Series(group, name=ACSIncome.group).to_frame()
    if sensitive_attributes == "sex":
        s["SEX"] = s["SEX"] - 1
    elif sensitive_attributes == "race":
        s["RAC1P"] = (s["RAC1P"] == 1).astype(np.int)

    X = pd.get_dummies(X, columns=ACSIncome.features)

    return X, y, s




def load_folktables_publiccoverage(sensitive_attributes="sex", survey_year = "2018", horizon= "1-Year", states = ["CA"]):
    if sensitive_attributes == "sex":
        features = ['AGEP','SCHL','MAR','SEX','DIS','ESP','CIT','MIG','MIL','ANC','NATIVITY','DEAR','DEYE','DREM','PINCP','ESR','ST','FER','RAC1P']
        group = "SEX"
    elif sensitive_attributes == "race":
        features = ['AGEP','SCHL','MAR','SEX','DIS','ESP','CIT','MIG','MIL','ANC','NATIVITY','DEAR','DEYE','DREM','PINCP','ESR','ST','FER','RAC1P']
        group = "RAC1P"
    else:
        features = ['AGEP','SCHL','MAR','SEX','DIS','ESP','CIT','MIG','MIL','ANC','NATIVITY','DEAR','DEYE','DREM','PINCP','ESR','ST','FER','RAC1P']
        group = "SEX"


    def public_coverage_filter(data):
        """
        Filters for the public health insurance prediction task; focus on low income Americans, and those not eligible for Medicare
        """
        df = data
        df = df[df['AGEP'] < 65]
        df = df[df['PINCP'] <= 30000]
        return df


    ACSPublicCoverage = folktables.BasicProblem(
        features=features,
        target='PUBCOV',
        target_transform=lambda x: x == 1,
        group=group,
        preprocess=public_coverage_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )

    data_source = ACSDataSource(survey_year=survey_year, horizon=horizon, survey='person',
                                root_dir="/data/han/data/fairness/folktables/")
    acs_data = data_source.get_data(states=states, download=True)
    features, label, group = ACSPublicCoverage.df_to_numpy(acs_data)

    X = pd.DataFrame(features, columns=ACSPublicCoverage.features)
    y = pd.Series(label)
    s = pd.Series(group, name=ACSPublicCoverage.group).to_frame()
    if sensitive_attributes == "sex":
        s["SEX"] = s["SEX"] - 1
    elif sensitive_attributes == "race":
        s["RAC1P"] = (s["RAC1P"] == 1).astype(np.int)

    X = pd.get_dummies(X, columns=ACSPublicCoverage.features)

    return X, y, s




def load_folktables_incomepovertyratio(sensitive_attributes="sex", survey_year = "2018", horizon= "1-Year", states = ["CA"]):
    if sensitive_attributes == "sex":
        features = ['AGEP','SCHL','MAR','SEX','DIS','ESP','CIT','MIG','MIL','ANC','NATIVITY','DEAR','DEYE','DREM','PINCP','ESR','ST','FER','RAC1P']
        group = "SEX"
    elif sensitive_attributes == "race":
        features = ['AGEP','SCHL','MAR','SEX','DIS','ESP','CIT','MIG','MIL','ANC','NATIVITY','DEAR','DEYE','DREM','PINCP','ESR','ST','FER','RAC1P']
        group = "RAC1P"
    else:
        features = ['AGEP','SCHL','MAR','SEX','DIS','ESP','CIT','MIG','MIL','ANC','NATIVITY','DEAR','DEYE','DREM','PINCP','ESR','ST','FER','RAC1P']
        group = "SEX"


    ACSIncomePovertyRatio = folktables.BasicProblem(
        features=features,
        target='POVPIP',
        target_transform=lambda x: x < 250,
        group=group,
        preprocess=lambda x: x,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )


    data_source = ACSDataSource(survey_year=survey_year, horizon=horizon, survey='person',
                                root_dir="/data/han/data/fairness/folktables/")
    acs_data = data_source.get_data(states=states, download=True)
    features, label, group = ACSIncomePovertyRatio.df_to_numpy(acs_data)

    X = pd.DataFrame(features, columns=ACSIncomePovertyRatio.features)
    y = pd.Series(label)
    s = pd.Series(group, name=ACSIncomePovertyRatio.group).to_frame()
    if sensitive_attributes == "sex":
        s["SEX"] = s["SEX"] - 1
    elif sensitive_attributes == "race":
        s["RAC1P"] = (s["RAC1P"] == 1).astype(np.int)

    X = pd.get_dummies(X, columns=['AGEP','SCHL','MAR','SEX','DIS','ESP','CIT','MIG','MIL','ANC','NATIVITY','DEAR','DEYE','DREM','RAC1P'])

    return X, y, s




def load_folktables_employment_5year(task=None, sensitive_attributes="sex"):
    if sensitive_attributes == "sex":
        features = ['AGEP','SCHL','MAR','RELP','DIS','ESP','CIT','MIG','MIL','ANC','NATIVITY','DEAR','DEYE','DREM','RAC1P']
        group = "SEX"
    elif sensitive_attributes == "race":
        features = ['AGEP', 'SCHL', 'MAR', 'RELP', 'DIS', 'ESP', 'CIT', 'MIG','MIL', 'ANC', 'NATIVITY', 'DEAR', 'DEYE', 'DREM', 'SEX']
        group = "RAC1P"
    else:
        features = ['AGEP','SCHL','MAR','RELP','DIS','ESP','CIT','MIG','MIL','ANC','NATIVITY','DEAR','DEYE','DREM','RAC1P']
        group = "SEX"


    ACSEmployment = folktables.BasicProblem(
        features=features,
        target='ESR',
        target_transform=lambda x: x == 1,
        group=group,
        preprocess=lambda x: x,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )


    data_source = ACSDataSource(survey_year='2018', horizon='5-Year', survey='person',
                                root_dir="/data/han/data/fairness/folktables/")
    acs_data = data_source.get_data(states=["CA"], download=True)
    features, label, group = ACSEmployment.df_to_numpy(acs_data)


    X = pd.DataFrame(features, columns=ACSEmployment.features)
    y = pd.Series(label)
    s = pd.Series(group, name=ACSEmployment.group).to_frame()
    if sensitive_attributes == "sex":
        s["SEX"] = s["SEX"] - 1
    elif sensitive_attributes == "race":
        s["RAC1P"] = (s["RAC1P"] == 1).astype(np.int)

    X = pd.get_dummies(X, columns=ACSEmployment.features)


    return X, y, s


def load_folktables_income_5year(sensitive_attributes="sex"):
    if sensitive_attributes == "sex":
        features = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'RAC1P']
        group = "SEX"
    elif sensitive_attributes == "race":
        features = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX']
        group = "RAC1P"
    else:
        features = ['AGEP','COW','SCHL','MAR','OCCP','POBP','RELP','WKHP','RAC1P']
        group = "SEX"


    def adult_filter(data):
        """Mimic the filters in place for Adult data.

        Adult documentation notes: Extraction was done by Barry Becker from
        the 1994 Census database. A set of reasonably clean records was extracted
        using the following conditions:
        ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))
        """
        df = data
        df = df[df['AGEP'] > 16]
        df = df[df['PINCP'] > 100]
        df = df[df['WKHP'] > 0]
        df = df[df['PWGTP'] >= 1]
        return df


    ACSIncome = folktables.BasicProblem(
        features=features,
        target='PINCP',
        target_transform=lambda x: x > 50000,
        group=group,
        preprocess=adult_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )


    data_source = ACSDataSource(survey_year='2018', horizon='5-Year', survey='person',
                                root_dir="/data/han/data/fairness/folktables/")
    acs_data = data_source.get_data(states=["CA"], download=True)
    features, label, group = ACSIncome.df_to_numpy(acs_data)

    X = pd.DataFrame(features, columns=ACSIncome.features)
    y = pd.Series(label)
    s = pd.Series(group, name=ACSIncome.group).to_frame()
    if sensitive_attributes == "sex":
        s["SEX"] = s["SEX"] - 1
    elif sensitive_attributes == "race":
        s["RAC1P"] = (s["RAC1P"] == 1).astype(np.int)

    X = pd.get_dummies(X, columns=ACSIncome.features)

    return X, y, s



def load_compas_dataset():
    pass