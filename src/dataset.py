from tkinter import S
import pandas as pd
import os
import numpy as np
from torch.utils.data import TensorDataset
import torch
import os
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.datasets.folder import pil_loader
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class PandasDataSet(TensorDataset):
    def __init__(self, *dataframes):
        tensors = (self._df_to_tensor(df) for df in dataframes)
        super(PandasDataSet, self).__init__(*tensors)

    def _df_to_tensor(self, df):
        if isinstance(df, pd.Series):
            df = df.to_frame("dummy")
        return torch.from_numpy(df.values).float()


def load_adult_data(path="../datasets", sensitive_attribute="sex"):
    column_names = ["age","workclass","fnlwgt","education","education_num","marital-status","occupation","relationship","race","sex","capital_gain","capital_loss","hours_per_week","native-country","target"]

    categorical_features = ["workclass", "marital-status", "occupation", "relationship", "native-country", "education"]
    features_to_drop = ["fnlwgt"]

    df_train = pd.read_csv(os.path.join(path, "adult.data"), names=column_names, na_values="?", sep=r"\s*,\s*", engine="python")
    df_test = pd.read_csv(os.path.join(path, "adult.test"), names=column_names, na_values="?", sep=r"\s*,\s*", engine="python", skiprows=1)

    df = pd.concat([df_train, df_test])
    df.drop(columns=features_to_drop, inplace=True)
    df.dropna(inplace=True)

    # df = pd.get_dummies(df, columns=categorical_features)

    if sensitive_attribute == "race":
        df = df[df["race"].isin(["White", "Black"])]
        s = df[sensitive_attribute][df["race"].isin(["White", "Black"])]
        s = (s == "White").astype(int).to_frame()
        categorical_features.append( "sex" )

    if sensitive_attribute == "sex":
        s = df[sensitive_attribute]
        s = (s == "Male").astype(int).to_frame()
        categorical_features.append( "race" )

    df["target"] = df["target"].replace({"<=50K.": 0, ">50K.": 1, ">50K": 1, "<=50K": 0})
    y = df["target"]

    X = df.drop(columns=["target", sensitive_attribute])
    # X = pd.get_dummies(X, columns=categorical_features)
    X[categorical_features] = X[categorical_features].astype("string")


    # Convert all non-uint8 columns to float32
    string_cols = X.select_dtypes(exclude="string").columns
    X[string_cols] = X[string_cols].astype("float32")

    return X, y, s


def load_german_data(path="../datasets/germen", sensitive_attribute="sex"):
    # chagne the personal_status name to sex
    column_names = ["status","month","credit_history","purpose","credit_amount","savings","employment","investment_as_income_percentage","sex","other_debtors","residence_since","property","age","installment_plans","housing","number_of_credits","skill_level","people_liable_for","telephone","foreign_worker","credit"]
    categorical_features = ["status","credit_history","purpose","savings","employment","other_debtors","property","installment_plans","housing","skill_level","telephone","foreign_worker"]
    status_map = {"A91": "male", "A93": "male", "A94": "male", "A92": "female", "A95": "female"}

    df = pd.read_csv(os.path.join(path, "german.data"), na_values="NA", index_col=None, sep=" ", header=None, names=column_names)
    df = df.fillna("none")

    df["age"] = df["age"].apply(lambda x: "x>25" if x > 25 else "x<=25")
    df["sex"] = df["sex"].apply(lambda x: status_map[x])

    if sensitive_attribute == "sex":
        s = df[sensitive_attribute]
        s = (s == "male").astype(int).to_frame()
        categorical_features.append( "age" )
    if sensitive_attribute == "age":
        s = df[sensitive_attribute]
        s = (s == "x>25").astype(int).to_frame()
        categorical_features.append( "sex" )

    y = (df["credit"] == 1).astype(int).to_frame()
    X = df.drop(columns=["credit", sensitive_attribute])

    X[categorical_features] = X[categorical_features].astype("string")


    # Convert all non-uint8 columns to float32
    string_cols = X.select_dtypes(exclude="string").columns
    X[string_cols] = X[string_cols].astype("float32")

    return X, y, s


def load_bank_marketing_data(path="../datasets/bank/raw", sensitive_attribute="age"):
    df = pd.read_csv(os.path.join(path, "bank-additional-full.csv"), sep=";")
    categorical_features = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", "poutcome"]

    
    df["y"] = df["y"].replace({"yes": 1, "no": 0})
    y = df["y"].to_frame()
    s = df[sensitive_attribute]
    s = (s >= 25).astype(int).to_frame()

    X = df.drop(columns=["y", "age"])

    X[categorical_features] = X[categorical_features].astype("string")


    # Convert all non-uint8 columns to float32
    string_cols = X.select_dtypes(exclude="string").columns
    X[string_cols] = X[string_cols].astype("float32")

    return X, y, s




def load_census_income_kdd_data(path="../datasets/census_income_kdd/raw", sensitive_attribute="sex"):

    colum_names = ["age","workclass","industry_code","occupation_code","education","wage_per_hour","enrolled_in_edu_inst_last_wk",
    "marital_status","major_industry_code","major_occupation_code","race","hispanic_origin","sex","member_of_a_labour_union","reason_for_unemployment",
    "employment_status","capital_gains","capital_losses","dividend_from_stocks","tax_filler_status","region_of_previous_residence","state_of_previous_residence",
    "detailed_household_and_family_stat","detailed_household_summary_in_household","instance_weight","migration_code_change_in_msa","migration_code_change_in_reg",
    "migration_code_move_within_reg","live_in_this_house_1_year_ag","migration_prev_res_in_sunbelt","num_persons_worked_for_employer","family_members_under_18","country_of_birth_father",
    "country_of_birth_mother","country_of_birth_self","citizenship","own_business_or_self_employed","fill_inc_questionnaire_for_veteran's_admin","veterans_benefits","weeks_worked_in_year","year","class"]


    categorical_features = [
    "workclass","industry_code","occupation_code","education","enrolled_in_edu_inst_last_wk",
    "marital_status","major_industry_code","major_occupation_code","hispanic_origin","member_of_a_labour_union","reason_for_unemployment",
    "employment_status","tax_filler_status","region_of_previous_residence","state_of_previous_residence",
    "detailed_household_and_family_stat","detailed_household_summary_in_household","migration_code_change_in_msa","migration_code_change_in_reg",
    "migration_code_move_within_reg","live_in_this_house_1_year_ag","migration_prev_res_in_sunbelt","family_members_under_18","country_of_birth_father",
    "country_of_birth_mother","country_of_birth_self","citizenship","own_business_or_self_employed","fill_inc_questionnaire_for_veteran's_admin","veterans_benefits","year"
    ]

    feature_to_keep = [ "workclass","industry_code","occupation_code","education","enrolled_in_edu_inst_last_wk",
    "marital_status","major_industry_code","major_occupation_code","hispanic_origin","member_of_a_labour_union","reason_for_unemployment",
    "employment_status","tax_filler_status","region_of_previous_residence","state_of_previous_residence",
    "detailed_household_and_family_stat","detailed_household_summary_in_household","instance_weight","migration_code_change_in_msa","migration_code_change_in_reg",
    "migration_code_move_within_reg","live_in_this_house_1_year_ag","migration_prev_res_in_sunbelt","family_members_under_18","country_of_birth_father",
    "country_of_birth_mother","country_of_birth_self","citizenship","own_business_or_self_employed","fill_inc_questionnaire_for_veteran's_admin","veterans_benefits","year"]


    df1 = pd.read_csv(os.path.join(path, "census-income.data"),header=None,names=colum_names)
    df2 = pd.read_csv(os.path.join(path, "census-income.test"),header=None,names=colum_names)



    df = pd.concat([df1, df2], ignore_index=True)
    df = df.drop_duplicates(keep="first", inplace=False)

    # df.columns = df.columns.str.lower()

    if sensitive_attribute == "race":
        df = df[df["race"].isin([" White", " Black"])]
        s = df[sensitive_attribute]
        s = (s == " White").astype(int).to_frame()
        categorical_features.append("sex")
    if sensitive_attribute == "sex":
        s = df[sensitive_attribute]
        s = (s == " Male").astype(int).to_frame()
        categorical_features.append("race")

    # # targets; 1 , otherwise 0
    y = (df["class"] == " - 50000.").astype(int)


    # features; note that the 'target' and sentive attribute columns are dropped
    X = df.drop(columns=["class", sensitive_attribute])
    X[categorical_features] = X[categorical_features].astype("string")


    # Convert all non-uint8 columns to float32
    string_cols = X.select_dtypes(exclude="string").columns
    X[string_cols] = X[string_cols].astype("float32")
    return X, y, s



def load_acs_data(path = '../datasets/acs/raw', target_attr="income", sensitive_attribute="sex", survey_year="2018",  states=["CA"], horizon="1-Year",survey='person'):
    from folktables import ACSDataSource, ACSIncome, ACSEmployment, ACSPublicCoverage, ACSMobility, ACSTravelTime
    data_source = ACSDataSource(survey_year=survey_year, horizon=horizon, survey=survey, root_dir=path)
    data = data_source.get_data(states=states, download=True)

    if target_attr == "income":
        features, labels, _ = ACSIncome.df_to_pandas(data)
        categorical_features = ["COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP"]
    elif target_attr == "employment":
        features, labels, _ = ACSEmployment.df_to_pandas(data)
        categorical_features = ["AGEP", "SCHL", "MAR", "RELP", "DIS", "ESP", "CIT", "MIG", "MIL", "ANC", "NATIVITY", "DEAR", "DEYE", "DREM"]
    elif target_attr == "publiccoverage":
        features, labels, _ = ACSPublicCoverage.df_to_pandas(data)
        categorical_features = ['AGEP','SCHL','MAR','DIS','ESP','CIT','MIG','MIL','ANC','NATIVITY','DEAR','DEYE','DREM','PINCP','ESR','ST','FER']
    elif target_attr == "mobility":
        features, labels, _ = ACSMobility.df_to_pandas(data)
        categorical_features = ['AGEP','SCHL','MAR','DIS','ESP','CIT','MIL','ANC','NATIVITY','RELP','DEAR','DEYE','DREM','GCL','COW','ESR','WKHP','JWMNP','PINCP']
    elif target_attr == "traveltime":
        features, labels, _ = ACSTravelTime.df_to_pandas(data)
        categorical_features = ['AGEP','SCHL','MAR','DIS','ESP','MIG','RELP','PUMA','ST','CIT','OCCP','JWTR','POWPUMA','POVPIP']

    else:
        print( "error" )
    

    df = features
    y = labels.astype(np.int32)
    if sensitive_attribute == "sex":
        sensitive_attribute = "SEX"
        s = (df["SEX"] == 2).astype(np.int32).to_frame()
        categorical_features.append("RAC1P")

    elif sensitive_attribute == "race":
        sensitive_attribute = "RAC1P"
        s = (df["RAC1P"] == 1).astype(np.int32).to_frame()
        categorical_features.append("SEX")

    X = df.drop(columns=[sensitive_attribute])
    X[categorical_features] = X[categorical_features].astype("string")


    # Convert all non-uint8 columns to float32
    string_cols = X.select_dtypes(exclude="string").columns
    X[string_cols] = X[string_cols].astype("float32")

    return X, y, s

# def load_meps_data(path="../datasets/meps/raw", sensitive_attribute="age"):
#     categorical_features = (
#         ["REGION","SEX","MARRY","FTSTU","ACTDTY","HONRDC","RTHLTH","MNHLTH","HIBPDX","CHDDX","ANGIDX","MIDX","OHRTDX","STRKDX","EMPHDX","CHBRON","CHOLDX","CANCERDX","DIABDX","JTPAIN","ARTHDX","ARTHTYPE","ASTHDX","ADHDADDX","PREGNT","WLKLIM","ACTLIM","SOCLIM","COGLIM","DFHEAR42","DFSEE42","ADSMOK42","PHQ242","EMPST","POVCAT","INSCOV",],
#     )
#     features_to_keep = ["REGION","AGE","SEX","RACE","MARRY","FTSTU","ACTDTY","HONRDC","RTHLTH","MNHLTH","HIBPDX","CHDDX","ANGIDX","MIDX","OHRTDX","STRKDX","EMPHDX","CHBRON","CHOLDX","CANCERDX","DIABDX","JTPAIN","ARTHDX","ARTHTYPE","ASTHDX","ADHDADDX","PREGNT","WLKLIM","ACTLIM","SOCLIM","COGLIM","DFHEAR42","DFSEE42","ADSMOK42","PCS42","MCS42","K6SUM42","PHQ242","EMPST","POVCAT","INSCOV","UTILIZATION","PERWT16F",
#     ]

#     df = pd.read_csv(os.path.join(path, "h181.csv"))

#     def race(row):
#         # non-Hispanic Whites are marked as WHITE; all others as NON-WHITE
#         if (row["HISPANX"] == 2) and (row["RACEV2X"] == 1):
#             return "White"
#         return "Non-White"

#     df["RACEV2X"] = df.apply(lambda row: race(row), axis=1)
#     df = df.rename(columns={"RACEV2X": "RACE"})

#     df = df[df["PANEL"] == 21]

#     # RENAME COLUMNS
#     df = df.rename(
#         columns={"FTSTU53X": "FTSTU","ACTDTY53": "ACTDTY","HONRDC53": "HONRDC","RTHLTH53": "RTHLTH","MNHLTH53": "MNHLTH","CHBRON53": "CHBRON","JTPAIN53": "JTPAIN","PREGNT53": "PREGNT","WLKLIM53": "WLKLIM","ACTLIM53": "ACTLIM","SOCLIM53": "SOCLIM","COGLIM53": "COGLIM","EMPST53": "EMPST","REGION53": "REGION","MARRY53X": "MARRY","AGE53X": "AGE","POVCAT16": "POVCAT","INSCOV16": "INSCOV",
#         }
#     )

#     df = df[df["REGION"] >= 0]  # remove values -1
#     df = df[df["AGE"] >= 0]  # remove values -1

#     df = df[df["MARRY"] >= 0]  # remove values -1, -7, -8, -9

#     df = df[df["ASTHDX"] >= 0]  # remove values -1, -7, -8, -9

#     df = df[
#         (
#             df[
#                 ["FTSTU","ACTDTY","HONRDC","RTHLTH","MNHLTH","HIBPDX","CHDDX","ANGIDX","EDUCYR","HIDEG","MIDX","OHRTDX","STRKDX","EMPHDX","CHBRON","CHOLDX","CANCERDX","DIABDX","JTPAIN","ARTHDX","ARTHTYPE","ASTHDX","ADHDADDX","PREGNT","WLKLIM","ACTLIM","SOCLIM","COGLIM","DFHEAR42","DFSEE42","ADSMOK42","PHQ242","EMPST","POVCAT","INSCOV",
#                 ]
#             ]
#             >= -1
#         ).all(1)
#     ]  # for all other categorical features, remove values < -1

#     def utilization(row):
#         return row["OBTOTV16"] + row["OPTOTV16"] + row["ERTOT16"] + row["IPNGTD16"] + row["HHTOTD16"]

#     df["TOTEXP16"] = df.apply(lambda row: utilization(row), axis=1)
#     lessE = df["TOTEXP16"] < 10.0
#     df.loc[lessE, "TOTEXP16"] = 0.0
#     moreE = df["TOTEXP16"] >= 10.0
#     df.loc[moreE, "TOTEXP16"] = 1.0

#     df = df.rename(columns={"TOTEXP16": "UTILIZATION"})

#     df["target"] = df["target"].replace({"yes": 1, "no": 0})
#     y = df["target"]
#     s = df[sensitive_attribute]
#     s = (s >= 25).astype(int).to_frame()
#     X = df.drop(columns=["target", "age"])
#     return X, y, s


def load_compas_data(path="../datasets/compas/raw", sensitive_attribute="sex"):
    # We use the same features_to_keep and categorical_features from AIF360 at https://github.com/Trusted-AI/AIF360/blob/master/aif360/datasets/compas_dataset.py

    features_to_keep = ["sex","age","age_cat","race","juv_fel_count","juv_misd_count","juv_other_count","priors_count","c_charge_degree","c_charge_desc","two_year_recid"]
    categorical_features = ["age_cat", "c_charge_degree", "c_charge_desc"]

    df = pd.read_csv(os.path.join(path, "compas-scores-two-years.csv"), index_col = 0)


    # df = df.dropna()
    df = df[df["days_b_screening_arrest"] <= 30]
    df = df[df["days_b_screening_arrest"] >= -30]
    df = df[df["is_recid"] != -1]
    df = df[df["c_charge_degree"] != "O"]
    df = df[df["score_text"] != "N/A"]
    df = df[features_to_keep]

    if sensitive_attribute == "sex":
        s = df[sensitive_attribute]
        s = (s == "Male").astype(int).to_frame()
        categorical_features.append("race")
    elif sensitive_attribute == "race":
        s = df[sensitive_attribute]
        s = (s == "Caucasian").astype(int).to_frame()
        categorical_features.append("sex")
    else:
        print("error")


    y = (df["two_year_recid"] ==  1 ).astype(int).to_frame()


    X = df.drop(columns=["two_year_recid", sensitive_attribute])
    X = pd.get_dummies(X, columns=categorical_features)

    # Convert all non-uint8 columns to float32
    uint8_cols = X.select_dtypes(exclude="uint8").columns
    X[uint8_cols] = X[uint8_cols].astype("float32")

    return X, y, s

    # pass




# We set ethnicity and age as the sensitive attribute and the target label, respectively. 
def load_utkface_data(path="../datasets/utkface/raw/", sensitive_attribute="race"):
    # chagne the personal_status name to sex race and the target label to age

    df = pd.read_csv(os.path.join(path, "age_gender.csv"), na_values="NA", index_col=None, sep=",", header=0)

    # df['pixels'] = df['pixels'].apply(lambda x:  np.reshape(np.array(x.split(), dtype="float32"), (48,48)))
    df['pixels']= df['pixels'].apply(lambda x:  np.array(x.split(), dtype="float32"))
    df['pixels'] = df['pixels'].apply(lambda x: x/255)
    df['pixels'] = df['pixels'].apply(lambda x:  np.reshape(x, (1, 48,48)))
    df['pixels'] = df['pixels'].apply(lambda x: np.repeat(x, 3, axis=0))
    
    df["age"] = df["age"] > 30
    df["age"] = df["age"].astype(int)

    df["race"] = df["ethnicity"]
    df["race"] = df["race"] == 0
    df["race"] = df["race"].astype(int)   

    X = df['pixels'].to_frame()
    # s = df[ sensitive_attribute ].to_frame()
    attr = df[ ["age", "gender", "race" ]]

    return X, attr


# We set ethnicity and age as the sensitive attribute and the target label, respectively. 
def load_celeba_data(path="../datasets/celeba/raw/", sensitive_attribute="race"):
    # chagne the personal_status name to sex race and the target label to age

    df = pd.read_csv( os.path.join(path, "celeba.csv"), na_values="NA", index_col=None, sep=",", header=0)
    df['pixels']= df['pixels'].apply(lambda x:  np.array(x.split(), dtype="float32"))
    df['pixels'] = df['pixels'].apply(lambda x: x/255)
    df['pixels'] = df['pixels'].apply(lambda x:  np.reshape(x, (3, 48,48)))

    X = df['pixels'].to_frame()
    
    df["Gender"] = df["Male"]
    attr = df[ ["Smiling", "Wavy_Hair", "Attractive", "Male", "Young"  ]]

    return X, attr

if __name__ == '__main__':
    X, attr = load_utkface_data()
    print( type(X), type(y), type(s) )
    print( X.shape, y.shape, s.shape )
    print( X.iloc[0].shape )
    print( X.iloc[0] )

    X, y, s = load_compas_data()
    print( type(X), type(y), type(s) )
    print( X.shape, y.shape, s.shape )
