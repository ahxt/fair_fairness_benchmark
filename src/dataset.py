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
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# def load_adult_data_deperated(path, sensitive_attributes = "sex"):
#     '''
#     We borrow the code from https://github.com/IBM/sensitive-subspace-robustness
#     Preprocess the adult data set by removing some features and put adult data into a BinaryLabelDataset
#     You need to download the adult dataset (both the adult.data and adult.test files) from https://archive.ics.uci.edu/ml/datasets/Adult
#     '''

#     headers = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-stataus', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'y']

#     # train = pd.read_csv(os.path.join(path, "adult.data"), header = None, na_values="?", sep=r'\s*,\s*', engine='python')
#     # test = pd.read_csv(os.path.join(path, "adult.test"), header = None,  na_values="?", sep=r'\s*,\s*', engine='python')

#     train = pd.read_csv(os.path.join(path, "adult.data"), header = None, sep=r'\s*,\s*', engine='python')
#     test = pd.read_csv(os.path.join(path, "adult.test"), header = None,  sep=r'\s*,\s*', engine='python', skiprows=1)

#     df = pd.concat([train, test], ignore_index=True)
#     df.columns = headers

#     df['y'] = df['y'].replace({'<=50K.': 0, '>50K.': 1, '>50K': 1, '<=50K': 0 })

#     df = df.drop(df[(df[headers[-2]] == ' ?') | (df[headers[6]] == ' ?')].index)
#     df = pd.get_dummies(df, columns=[headers[1], headers[5], headers[6], headers[7], headers[9], headers[8], 'native-country'])

#     # print( df.columns )
#     delete_these = ['race_Amer-Indian-Eskimo','race_Asian-Pac-Islander','race_Black','race_Other', 'sex_Female']

#     delete_these += ['native-country_Cambodia', 'native-country_Canada', 'native-country_China', 'native-country_Columbia', 'native-country_Cuba', 'native-country_Dominican-Republic', 'native-country_Ecuador', 'native-country_El-Salvador', 'native-country_England', 'native-country_France', 'native-country_Germany', 'native-country_Greece', 'native-country_Guatemala', 'native-country_Haiti', 'native-country_Holand-Netherlands', 'native-country_Honduras', 'native-country_Hong', 'native-country_Hungary', 'native-country_India', 'native-country_Iran', 'native-country_Ireland', 'native-country_Italy', 'native-country_Jamaica', 'native-country_Japan', 'native-country_Laos', 'native-country_Mexico', 'native-country_Nicaragua', 'native-country_Outlying-US(Guam-USVI-etc)', 'native-country_Peru', 'native-country_Philippines', 'native-country_Poland', 'native-country_Portugal', 'native-country_Puerto-Rico', 'native-country_Scotland', 'native-country_South', 'native-country_Taiwan', 'native-country_Thailand', 'native-country_Trinadad&Tobago', 'native-country_United-States', 'native-country_Vietnam', 'native-country_Yugoslavia']

#     delete_these += ['fnlwgt', 'education']

#     df.drop(delete_these, axis=1, inplace=True)

#     # gender id = 39
#     if sensitive_attributes == "sex":
#         s = df["sex_Male"]
#         df.drop(["sex_Male"], axis=1, inplace=True)
#     elif sensitive_attributes == "race":
#         s = df["race_White"]
#         df.drop(["race_White"], axis=1, inplace=True)

#     y = df["y"]
#     df.drop(["y"], axis=1, inplace=True)

#     return df, y, s


class PandasDataSet(TensorDataset):

    def __init__(self, *dataframes):
        tensors = (self._df_to_tensor(df) for df in dataframes)
        super(PandasDataSet, self).__init__(*tensors)

    def _df_to_tensor(self, df):
        if isinstance(df, pd.Series):
            df = df.to_frame('dummy')
        return torch.from_numpy(df.values).float()


def load_adult_data(path="../datasets", sensitive_attribute="sex", keep_sensitive_attribute=False):

    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                    'marital-status', 'occupation', 'relationship', 'race', 'sex',
                    'capital_gain', 'capital_loss', 'hours_per_week', 'native-country', 'target']

    categorical_features = ['workclass', 'marital-status',
                            'occupation', 'relationship', 'native-country', "education"]
    features_to_drop = ['fnlwgt']

    df_train = pd.read_csv(os.path.join(
        path, "adult.data"), names=column_names, na_values="?", sep=r'\s*,\s*', engine='python')
    df_test = pd.read_csv(os.path.join(path, "adult.test"), names=column_names,
                          na_values="?", sep=r'\s*,\s*', engine='python', skiprows=1)

    df = pd.concat([df_train, df_test])
    df.drop(columns=features_to_drop, inplace=True)
    df.dropna(inplace=True)

    df = pd.get_dummies(df, columns=categorical_features)

    if sensitive_attribute == "race":
        df = df[df['race'].isin(['White', 'Black'])]
        s = df[sensitive_attribute][df['race'].isin(['White', 'Black'])]
        s = (s == 'White').astype(int).to_frame()
        df = pd.get_dummies(df, columns=["sex"])

    if sensitive_attribute == "sex":
        s = df[sensitive_attribute]
        s = (s == 'Male').astype(int).to_frame()
        df = pd.get_dummies(df, columns=["race"])

    df['target'] = df['target'].replace(
        {'<=50K.': 0, '>50K.': 1, '>50K': 1, '<=50K': 0})
    y = df['target']

    X = df.drop(columns=['target', sensitive_attribute])

    # Convert all non-uint8 columns to float32
    uint8_cols = X.select_dtypes(exclude='uint8').columns
    X[uint8_cols] = X[uint8_cols].astype('float32')

    return X, y, s


def load_german_data(path='../datasets/germen', sensitive_attribute="sex"):

    # chagne the personal_status name to sex
    column_names = ['status', 'month', 'credit_history',
                    'purpose', 'credit_amount', 'savings', 'employment',
                    'investment_as_income_percentage', 'sex',
                    'other_debtors', 'residence_since', 'property', 'age',
                    'installment_plans', 'housing', 'number_of_credits',
                    'skill_level', 'people_liable_for', 'telephone',
                    'foreign_worker', 'credit']
    categorical_features = ['status', 'credit_history', 'purpose',
                            'savings', 'employment', 'other_debtors', 'property',
                            'installment_plans', 'housing', 'skill_level', 'telephone',
                            'foreign_worker']
    numerical_features = ['month', 'credit_amount', 'credit_history',
                          "investment_as_income_percentage", "residence_since", "number_of_credits", "people_liable_for"]
    status_map = {'A91': 'male', 'A93': 'male', 'A94': 'male',
                  'A92': 'female', 'A95': 'female'}

    df = pd.read_csv(os.path.join(path, 'german.data'), na_values="NA",
                     index_col=None, sep=" ", header=None, names=column_names)
    df = df.fillna("none")

    df = pd.get_dummies(df, columns=categorical_features)

    df["age"] = df["age"].apply(lambda x: "x>25" if x > 25 else "x<=25")
    df["sex"] = df["sex"].apply(lambda x: status_map[x])

    if sensitive_attribute == "sex":
        s = df[sensitive_attribute]
        s = (s == 'male').astype(int).to_frame()
        df = pd.get_dummies(df, columns=["age"])
    if sensitive_attribute == "age":
        s = df[sensitive_attribute]
        s = (s == "x>25").astype(int).to_frame()
        df = pd.get_dummies(df, columns=["sex"])

    y = (df['credit'] == 1).astype(int).to_frame()
    X = df.drop(columns=['credit', sensitive_attribute])

    # Convert all non-uint8 columns to float32
    uint8_cols = X.select_dtypes(exclude='uint8').columns
    X[uint8_cols] = X[uint8_cols].astype('float32')

    return X, y, s


def load_bank_marketing_data(path="../datasets/bank/raw", sensitive_attribute="age"):
    df = pd.read_csv(os.path.join(path, 'bank-additional-full.csv'), sep=';')
    categorical_features = ['job', 'marital', 'education', 'default',
                            'housing', 'loan', 'contact', 'month', 'day_of_week',
                            'poutcome']
    df = pd.get_dummies(df, columns=categorical_features)
    df['y'] = df['y'].replace({'yes': 1, 'no': 0})
    y = df['y']
    s = df[sensitive_attribute]
    s = (s >= 25).astype(int).to_frame()
    X = df.drop(columns=['y', 'age'])

    # Convert all non-uint8 columns to float32
    uint8_cols = X.select_dtypes(exclude='uint8').columns
    X[uint8_cols] = X[uint8_cols].astype('float32')

    return X, y, s


def load_meps_data(path="../datasets/meps/raw", sensitive_attribute="age"):

    categorical_features = ['REGION', 'SEX', 'MARRY',
                            'FTSTU', 'ACTDTY', 'HONRDC', 'RTHLTH', 'MNHLTH', 'HIBPDX', 'CHDDX', 'ANGIDX',
                            'MIDX', 'OHRTDX', 'STRKDX', 'EMPHDX', 'CHBRON', 'CHOLDX', 'CANCERDX', 'DIABDX',
                            'JTPAIN', 'ARTHDX', 'ARTHTYPE', 'ASTHDX', 'ADHDADDX', 'PREGNT', 'WLKLIM',
                            'ACTLIM', 'SOCLIM', 'COGLIM', 'DFHEAR42', 'DFSEE42', 'ADSMOK42', 'PHQ242',
                            'EMPST', 'POVCAT', 'INSCOV'],
    features_to_keep = ['REGION', 'AGE', 'SEX', 'RACE', 'MARRY',
                        'FTSTU', 'ACTDTY', 'HONRDC', 'RTHLTH', 'MNHLTH', 'HIBPDX', 'CHDDX', 'ANGIDX',
                                 'MIDX', 'OHRTDX', 'STRKDX', 'EMPHDX', 'CHBRON', 'CHOLDX', 'CANCERDX', 'DIABDX',
                                 'JTPAIN', 'ARTHDX', 'ARTHTYPE', 'ASTHDX', 'ADHDADDX', 'PREGNT', 'WLKLIM',
                                 'ACTLIM', 'SOCLIM', 'COGLIM', 'DFHEAR42', 'DFSEE42', 'ADSMOK42',
                                 'PCS42',
                                 'MCS42', 'K6SUM42', 'PHQ242', 'EMPST', 'POVCAT', 'INSCOV', 'UTILIZATION', 'PERWT16F']

    df = pd.read_csv(os.path.join(path, 'h181.csv'))

    def race(row):
        # non-Hispanic Whites are marked as WHITE; all others as NON-WHITE
        if ((row['HISPANX'] == 2) and (row['RACEV2X'] == 1)):
            return 'White'
        return 'Non-White'

    df['RACEV2X'] = df.apply(lambda row: race(row), axis=1)
    df = df.rename(columns={'RACEV2X': 'RACE'})

    df = df[df['PANEL'] == 21]

    # RENAME COLUMNS
    df = df.rename(columns={'FTSTU53X': 'FTSTU', 'ACTDTY53': 'ACTDTY', 'HONRDC53': 'HONRDC', 'RTHLTH53': 'RTHLTH',
                            'MNHLTH53': 'MNHLTH', 'CHBRON53': 'CHBRON', 'JTPAIN53': 'JTPAIN', 'PREGNT53': 'PREGNT',
                            'WLKLIM53': 'WLKLIM', 'ACTLIM53': 'ACTLIM', 'SOCLIM53': 'SOCLIM', 'COGLIM53': 'COGLIM',
                            'EMPST53': 'EMPST', 'REGION53': 'REGION', 'MARRY53X': 'MARRY', 'AGE53X': 'AGE',
                            'POVCAT16': 'POVCAT', 'INSCOV16': 'INSCOV'})

    df = df[df['REGION'] >= 0]  # remove values -1
    df = df[df['AGE'] >= 0]  # remove values -1

    df = df[df['MARRY'] >= 0]  # remove values -1, -7, -8, -9

    df = df[df['ASTHDX'] >= 0]  # remove values -1, -7, -8, -9

    df = df[(df[['FTSTU', 'ACTDTY', 'HONRDC', 'RTHLTH', 'MNHLTH', 'HIBPDX', 'CHDDX', 'ANGIDX', 'EDUCYR', 'HIDEG',
                 'MIDX', 'OHRTDX', 'STRKDX', 'EMPHDX', 'CHBRON', 'CHOLDX', 'CANCERDX', 'DIABDX',
                 'JTPAIN', 'ARTHDX', 'ARTHTYPE', 'ASTHDX', 'ADHDADDX', 'PREGNT', 'WLKLIM',
                 'ACTLIM', 'SOCLIM', 'COGLIM', 'DFHEAR42', 'DFSEE42', 'ADSMOK42',
                 'PHQ242', 'EMPST', 'POVCAT', 'INSCOV']] >= -1).all(1)]  # for all other categorical features, remove values < -1

    def utilization(row):
        return row['OBTOTV16'] + row['OPTOTV16'] + row['ERTOT16'] + row['IPNGTD16'] + row['HHTOTD16']

    df['TOTEXP16'] = df.apply(lambda row: utilization(row), axis=1)
    lessE = df['TOTEXP16'] < 10.0
    df.loc[lessE, 'TOTEXP16'] = 0.0
    moreE = df['TOTEXP16'] >= 10.0
    df.loc[moreE, 'TOTEXP16'] = 1.0

    df = df.rename(columns={'TOTEXP16': 'UTILIZATION'})

    df['target'] = df['target'].replace({'yes': 1, 'no': 0})
    y = df['target']
    s = df[sensitive_attribute]
    s = (s >= 25).astype(int).to_frame()
    X = df.drop(columns=['target', 'age'])
    return X, y, s


def load_compas_dataset(path="../datasets/compas/compas-scores-two-years.csv", sensitive_attribute="sex"):

    # We use the same features_to_keep and categorical_features from AIF360 at https://github.com/Trusted-AI/AIF360/blob/master/aif360/datasets/compas_dataset.py

    features_to_keep = ['sex', 'age', 'age_cat', 'race', 'juv_fel_count', 'juv_misd_count',
                        'juv_other_count', 'priors_count', 'c_charge_degree', 'c_charge_desc', 'two_year_recid'],
    categorical_features = ['age_cat', 'c_charge_degree', 'c_charge_desc'],

    df = pd.read_csv(path=path)
    df = df.dropna()
    df = df[df['days_b_screening_arrest'] <= 30]
    df = df[df['days_b_screening_arrest'] >= -30]
    df = df[df['is_recid'] != -1]
    df = df[df['c_charge_degree'] != 'O']
    df = df[df['score_text'] != 'N/A']

    df['target'] = df['two_year_recid']

    if sensitive_attribute == "sex":
        s = df[sensitive_attribute]
        s = (s == 'male').astype(int).to_frame()
        df = pd.get_dummies(df, columns=["age"])
    if sensitive_attribute == "race":
        s = df[sensitive_attribute]
        s = (s == "x>25").astype(int).to_frame()
        df = pd.get_dummies(df, columns=["sex"])

    y = (df['credit'] == 1).astype(int).to_frame()
    X = df.drop(columns=['credit', sensitive_attribute])

    # Convert all non-uint8 columns to float32
    uint8_cols = X.select_dtypes(exclude='uint8').columns
    X[uint8_cols] = X[uint8_cols].astype('float32')

    return X, y, s

    # pass


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


def load_acs_dataset(task=None, sensitive_attributes="sex", survey_year="2018", horizon="1-Year", states=["CA"]):
    if sensitive_attributes == "sex":
        # features = ['AGEP','SCHL','MAR','RELP','DIS','ESP','CIT','MIG','MIL','ANC','NATIVITY','DEAR','DEYE','DREM','SEX','RAC1P']
        features = ['AGEP', 'SCHL', 'MAR', 'RELP', 'DIS', 'ESP', 'CIT',
                    'MIG', 'MIL', 'ANC', 'NATIVITY', 'DEAR', 'DEYE', 'DREM', 'RAC1P']
        group = "SEX"
    elif sensitive_attributes == "race":
        # features = ['AGEP', 'SCHL', 'MAR', 'RELP', 'DIS', 'ESP', 'CIT', 'MIG', 'MIL', 'ANC', 'NATIVITY', 'DEAR', 'DEYE', 'DREM', 'SEX', 'RAC1P']
        features = ['AGEP', 'SCHL', 'MAR', 'RELP', 'DIS', 'ESP', 'CIT',
                    'MIG', 'MIL', 'ANC', 'NATIVITY', 'DEAR', 'DEYE', 'DREM', 'SEX']
        group = "RAC1P"
    else:
        # features = ['AGEP','SCHL','MAR','RELP','DIS','ESP','CIT','MIG','MIL','ANC','NATIVITY','DEAR','DEYE','DREM','SEX','RAC1P']
        features = ['AGEP', 'SCHL', 'MAR', 'RELP', 'DIS', 'ESP', 'CIT',
                    'MIG', 'MIL', 'ANC', 'NATIVITY', 'DEAR', 'DEYE', 'DREM', 'RAC1P']
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


def load_folktables_employment(task=None, sensitive_attributes="sex", survey_year="2018", horizon="1-Year", states=["CA"]):
    if sensitive_attributes == "sex":
        # features = ['AGEP','SCHL','MAR','RELP','DIS','ESP','CIT','MIG','MIL','ANC','NATIVITY','DEAR','DEYE','DREM','SEX','RAC1P']
        features = ['AGEP', 'SCHL', 'MAR', 'RELP', 'DIS', 'ESP', 'CIT',
                    'MIG', 'MIL', 'ANC', 'NATIVITY', 'DEAR', 'DEYE', 'DREM', 'RAC1P']
        group = "SEX"
    elif sensitive_attributes == "race":
        # features = ['AGEP', 'SCHL', 'MAR', 'RELP', 'DIS', 'ESP', 'CIT', 'MIG', 'MIL', 'ANC', 'NATIVITY', 'DEAR', 'DEYE', 'DREM', 'SEX', 'RAC1P']
        features = ['AGEP', 'SCHL', 'MAR', 'RELP', 'DIS', 'ESP', 'CIT',
                    'MIG', 'MIL', 'ANC', 'NATIVITY', 'DEAR', 'DEYE', 'DREM', 'SEX']
        group = "RAC1P"
    else:
        # features = ['AGEP','SCHL','MAR','RELP','DIS','ESP','CIT','MIG','MIL','ANC','NATIVITY','DEAR','DEYE','DREM','SEX','RAC1P']
        features = ['AGEP', 'SCHL', 'MAR', 'RELP', 'DIS', 'ESP', 'CIT',
                    'MIG', 'MIL', 'ANC', 'NATIVITY', 'DEAR', 'DEYE', 'DREM', 'RAC1P']
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


def load_folktables_income(sensitive_attributes="sex", survey_year="2018", horizon="1-Year", states=["CA"]):
    if sensitive_attributes == "sex":
        # features = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P']
        features = ['AGEP', 'COW', 'SCHL', 'MAR',
                    'OCCP', 'POBP', 'RELP', 'WKHP', 'RAC1P']
        group = "SEX"
    elif sensitive_attributes == "race":
        # features = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P']
        features = ['AGEP', 'COW', 'SCHL', 'MAR',
                    'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX']
        group = "RAC1P"
    else:
        features = ['AGEP', 'COW', 'SCHL', 'MAR',
                    'OCCP', 'POBP', 'RELP', 'WKHP', 'RAC1P']
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


def load_folktables_publiccoverage(sensitive_attributes="sex", survey_year="2018", horizon="1-Year", states=["CA"]):
    if sensitive_attributes == "sex":
        features = ['AGEP', 'SCHL', 'MAR', 'SEX', 'DIS', 'ESP', 'CIT', 'MIG', 'MIL',
                    'ANC', 'NATIVITY', 'DEAR', 'DEYE', 'DREM', 'PINCP', 'ESR', 'ST', 'FER', 'RAC1P']
        group = "SEX"
    elif sensitive_attributes == "race":
        features = ['AGEP', 'SCHL', 'MAR', 'SEX', 'DIS', 'ESP', 'CIT', 'MIG', 'MIL',
                    'ANC', 'NATIVITY', 'DEAR', 'DEYE', 'DREM', 'PINCP', 'ESR', 'ST', 'FER', 'RAC1P']
        group = "RAC1P"
    else:
        features = ['AGEP', 'SCHL', 'MAR', 'SEX', 'DIS', 'ESP', 'CIT', 'MIG', 'MIL',
                    'ANC', 'NATIVITY', 'DEAR', 'DEYE', 'DREM', 'PINCP', 'ESR', 'ST', 'FER', 'RAC1P']
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


def load_folktables_incomepovertyratio(sensitive_attributes="sex", survey_year="2018", horizon="1-Year", states=["CA"]):
    if sensitive_attributes == "sex":
        features = ['AGEP', 'SCHL', 'MAR', 'SEX', 'DIS', 'ESP', 'CIT', 'MIG', 'MIL',
                    'ANC', 'NATIVITY', 'DEAR', 'DEYE', 'DREM', 'PINCP', 'ESR', 'ST', 'FER', 'RAC1P']
        group = "SEX"
    elif sensitive_attributes == "race":
        features = ['AGEP', 'SCHL', 'MAR', 'SEX', 'DIS', 'ESP', 'CIT', 'MIG', 'MIL',
                    'ANC', 'NATIVITY', 'DEAR', 'DEYE', 'DREM', 'PINCP', 'ESR', 'ST', 'FER', 'RAC1P']
        group = "RAC1P"
    else:
        features = ['AGEP', 'SCHL', 'MAR', 'SEX', 'DIS', 'ESP', 'CIT', 'MIG', 'MIL',
                    'ANC', 'NATIVITY', 'DEAR', 'DEYE', 'DREM', 'PINCP', 'ESR', 'ST', 'FER', 'RAC1P']
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

    X = pd.get_dummies(X, columns=['AGEP', 'SCHL', 'MAR', 'SEX', 'DIS', 'ESP',
                       'CIT', 'MIG', 'MIL', 'ANC', 'NATIVITY', 'DEAR', 'DEYE', 'DREM', 'RAC1P'])

    return X, y, s


def load_folktables_employment_5year(task=None, sensitive_attributes="sex"):
    if sensitive_attributes == "sex":
        features = ['AGEP', 'SCHL', 'MAR', 'RELP', 'DIS', 'ESP', 'CIT',
                    'MIG', 'MIL', 'ANC', 'NATIVITY', 'DEAR', 'DEYE', 'DREM', 'RAC1P']
        group = "SEX"
    elif sensitive_attributes == "race":
        features = ['AGEP', 'SCHL', 'MAR', 'RELP', 'DIS', 'ESP', 'CIT',
                    'MIG', 'MIL', 'ANC', 'NATIVITY', 'DEAR', 'DEYE', 'DREM', 'SEX']
        group = "RAC1P"
    else:
        features = ['AGEP', 'SCHL', 'MAR', 'RELP', 'DIS', 'ESP', 'CIT',
                    'MIG', 'MIL', 'ANC', 'NATIVITY', 'DEAR', 'DEYE', 'DREM', 'RAC1P']
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
        features = ['AGEP', 'COW', 'SCHL', 'MAR',
                    'OCCP', 'POBP', 'RELP', 'WKHP', 'RAC1P']
        group = "SEX"
    elif sensitive_attributes == "race":
        features = ['AGEP', 'COW', 'SCHL', 'MAR',
                    'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX']
        group = "RAC1P"
    else:
        features = ['AGEP', 'COW', 'SCHL', 'MAR',
                    'OCCP', 'POBP', 'RELP', 'WKHP', 'RAC1P']
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
