
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import logging
import time
import argparse

import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import torch.nn.functional as F
from table_logger import TableLogger


from dataset import load_census_income_kdd_data,load_census_income_kdd_data,load_adult_data,load_german_data, load_compas_data, load_german_data, load_bank_marketing_data, load_acs_data
from utils import seed_everything, PandasDataSet, print_metrics
from metrics import metric_evaluation
from networks import MLP, CFairNet


def test(model, test_loader, criterion, device, prefix="test", args=None):
    model.eval()
    test_loss = 0
    target_hat_list = []
    target_list = []
    sensitive_list = []

    with torch.no_grad():
        for data, target, sensitive in test_loader:
            data, target, sensitive = (data.to(device), target.to(device), sensitive.to(device))
            h, output = model.inference(data)
            test_loss += criterion(output, target).item()
            target_hat_list.append(output.cpu().numpy())
            target_list.append(target.cpu().numpy())
            sensitive_list.append(sensitive.cpu().numpy())

    target_hat_list = np.concatenate(target_hat_list, axis=0)
    target_list = np.concatenate(target_list, axis=0)
    sensitive_list = np.concatenate(sensitive_list, axis=0)
    metric = metric_evaluation(y_gt=target_list, y_pre=target_hat_list, s=sensitive_list, prefix=f"{prefix}_")

    test_loss /= len(test_loader.dataset)
    
    metric[f"{prefix}_loss"] = test_loss

    return metric



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="/data/han/data/fairness")
    parser.add_argument('--dataset', type=str, default="adult")
    parser.add_argument('--model', type=str, default="MLP")
    parser.add_argument('--target_attr', type=str, default="income")
    parser.add_argument('--sensitive_attr', type=str, default="sex")
    parser.add_argument("--evaluation_metrics", type=str, default="acc,ap,dp,eopp,eodd")

    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--clf_num_epochs', type=int, default=10)
    parser.add_argument('--adv_num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_hidden', type=int, default=512)

    parser.add_argument('--lam', type=float, default=1.0)

    parser.add_argument('--seed', type=int, default=1314)
    parser.add_argument('--exp_name', type=str, default="hhh_torch")
    parser.add_argument('--round', type=int, default=0)
    parser.add_argument('--log_interval', type=int, default=100)

    args = parser.parse_args()



    seed_everything(seed=args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("args:{}".format(args))

    if args.dataset == "adult":
        print(f"Dataset: adult")
        X, y, s = load_adult_data(path="../datasets/adult/raw", sensitive_attribute=args.sensitive_attr)

    elif args.dataset == "german":
        print(f"Dataset: german")
        X, y, s = load_german_data(path="../datasets/german/raw", sensitive_attribute=args.sensitive_attr)

    elif args.dataset == "kdd":
        print(f"Dataset: kdd")
        X, y, s = load_census_income_kdd_data("../datasets/census_income_kdd/raw", sensitive_attribute=args.sensitive_attr)

    elif args.dataset == "compas":
        print(f"Dataset: compas")
        X, y, s = load_compas_data(path="../datasets/compas/raw", sensitive_attribute=args.sensitive_attr)

    elif args.dataset == "bank_marketing":
        print(f"Dataset: bank_marketing")
        X, y, s = load_bank_marketing_data(path="../datasets/bank_marketing/raw", sensitive_attribute=args.sensitive_attr)

    elif args.dataset == "acs":
        X, y, s = load_acs_data( path= "../datasets/acs/raw", target_attr=args.acs_target_attr, sensitive_attribute=args.sensitive_attr)

    else:
        print(f"Wrong args.dataset")

    print(f'X.shape: {X.shape}')
    print(f'y.shape: {y.shape}')
    print(f's.shape: {s.shape}')
    print(f's.shape: {s.value_counts().to_dict()}')


    n_features = X.shape[1]

    # split into train/val/test set
    X_train, X_testvalid, y_train, y_testvalid, s_train, s_testvalid = train_test_split(X, y, s, test_size=0.6, stratify=y, random_state=args.seed)
    X_test, X_val, y_test, y_val, s_test, s_val = train_test_split(X_testvalid, y_testvalid, s_testvalid, test_size=0.5, stratify=y_testvalid, random_state=args.seed)



    print(f'X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}, s_train.shape: {s_train.shape}')
    print(f'X_val.shape: {X_val.shape}, y_val.shape: {y_val.shape}, s_val.shape: {s_val.shape}')
    print(f'X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}, s_test.shape: {s_test.shape}')




    scaler = StandardScaler().fit(X_train)
    scale_df = lambda df, scaler: pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
    X_train = X_train.pipe(scale_df, scaler) 
    X_val = X_val.pipe(scale_df, scaler) 
    X_test = X_test.pipe(scale_df, scaler) 


    train_data = PandasDataSet(X_train, y_train, s_train)
    val_data = PandasDataSet(X_val, y_val, s_val)
    test_data = PandasDataSet(X_test, y_test, s_test)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader( val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader( test_data, batch_size=args.batch_size, shuffle=False)


    s_val = s_val.values
    y_val = y_val.values

    s_test = s_test.values
    y_test = y_test.values

    s_train = s_train.values
    y_train = y_train.values


    advfairnet = CFairNet( n_features=n_features, num_classes=1 ).to(device)
    clf_criterion = nn.BCELoss()
    adv_criterion = nn.BCELoss(reduction="mean")

    optimizer = optim.Adam( advfairnet.parameters(), lr=args.lr )

    print(advfairnet)


    def train(advfairnet, data_loader, clf_criterion, adv_criterion, optimizer, device):
        advfairnet.train()

        for batch_idx, (X, y, s) in enumerate(data_loader):
            X, y, s = X.to(device), y.to(device), s.to(device)

            optimizer.zero_grad()

            y_hat, s_hat = advfairnet(X)
            clf_loss = clf_criterion(y_hat, y)
            adv_loss = adv_criterion(s_hat, s)

            loss = clf_loss + args.lam*adv_loss
            loss.backward()
            optimizer.step()        
        
        return advfairnet


    tbl = TableLogger(columns='Epoch(Tr|Val|Te),'+ args.evaluation_metrics, float_format='{:,.2f}'.format, default_colwidth=17)


    for epoch in range(1, args.num_epochs+1):
        
        advfairnet = train(advfairnet, train_loader, clf_criterion, adv_criterion, optimizer, device)


        if epoch % 5 == 0 or epoch == 3 or epoch == args.num_epochs:
            train_metrics = test(model=advfairnet, test_loader=train_loader, criterion=clf_criterion, device=device,  prefix="train")
            val_metrics   = test(model=advfairnet, test_loader=val_loader, criterion=clf_criterion, device=device, prefix="val")
            test_metrics  =  test(model=advfairnet, test_loader=test_loader, criterion=clf_criterion, device=device, prefix="test")

            res_dict = {}
            res_dict["epoch"] = epoch
            res_dict.update(train_metrics)
            res_dict.update(val_metrics)
            res_dict.update(test_metrics)

            res = print_metrics(res_dict, args.evaluation_metrics)
            tbl( epoch, *res)


    tbl.print_line( tbl.make_horizontal_border() )











