import logging
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from table_logger import TableLogger


from dataset import load_census_income_kdd_data,load_census_income_kdd_data,load_folktables_income,load_adult_data,load_german_data, load_compas_data, load_german_data, load_bank_marketing_data
from dataset import load_census_income_kdd_data,load_census_income_kdd_data,load_folktables_income,load_folktables_employment,load_folktables_income_5year,load_folktables_employment_5year
from utils import seed_everything, PandasDataSet
from metrics import metric_evaluation
from networks import MLP


logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s: - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def train_epoch(model, train_loader, optimizer, criterion, device, args=None):
    model.train()
    for batch_idx, (data, target, sensitive) in enumerate(train_loader):
        data, target, sensitive = data.to(device), target.to(device), sensitive.to(device)
        optimizer.zero_grad()
        h, output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    return model


def test(model, test_loader, criterion, device, prefix="test", args=None):
    model.eval()
    test_loss = 0
    target_hat_list = []
    target_list = []
    sensitive_list = []

    with torch.no_grad():
        for data, target, sensitive in test_loader:
            data, target, sensitive = (data.to(device), target.to(device), sensitive.to(device))
            h, output = model(data)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../datasets/adult/raw")
    parser.add_argument("--dataset", type=str,default="adult",choices=["adult","kdd","folktables_income","folktables_employment","folktables_income_5year","folktables_employment_5year","german"])
    parser.add_argument("--sensitive_attr", type=str, default="sex")

    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_hidden", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)

    parser.add_argument("--seed", type=int, default=1314)
    parser.add_argument("--exp_name", type=str, default="fair_fairness_benchmark")


    args = parser.parse_args()

    seed_everything(seed=args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"


    print("args:{}".format(args))

    if args.dataset == "folktables_income":
        print(f"Dataset: folktables_income")
        X, y, s = load_folktables_income(sensitive_attributes=args.sensitive_attr)

    elif args.dataset == "folktables_employment":
        print(f"Dataset: folktables_employment")
        X, y, s = load_folktables_employment(sensitive_attributes=args.sensitive_attr)

    elif args.dataset == "folktables_income_5year":
        print(f"Dataset: folktables_income_5year")
        X, y, s = load_folktables_income_5year(sensitive_attributes=args.sensitive_attr)

    elif args.dataset == "folktables_employment_5year":
        print(f"Dataset: folktables_employment_5year")
        X, y, s = load_folktables_employment_5year(sensitive_attributes=args.sensitive_attr)

    elif args.dataset == "adult":
        print(f"Dataset: adult")
        X, y, s = load_adult_data(path="../datasets/adult/raw", sensitive_attribute=args.sensitive_attr)

    elif args.dataset == "german":
        print(f"Dataset: german")
        X, y, s = load_german_data(path="../datasets/german/raw", sensitive_attribute=args.sensitive_attr)

    elif args.dataset == "kdd":
        print(f"Dataset: kdd")
        X, y, s = load_census_income_kdd_data("/data/han/data/fairness/census-income-mld", sensitive_attributes=args.sensitive_attr)

    elif args.dataset == "compas":
        print(f"Dataset: compas")
        X, y, s = load_compas_data(path="../datasets/compas/raw", sensitive_attribute=args.sensitive_attr)
    else:
        print(f"Wrong args.dataset")

    print(f"X.shape: {X.shape}")
    print(f"y.shape: {y.shape}")
    print(f"s.shape: {s.shape}")
    print(f"s.value_counts(): {s.value_counts().to_dict()}")
    print(f"y.value_counts(): {y.value_counts().to_dict()}")

    n_features = X.shape[1]
    n_classes = len(np.unique(y))

    X_train, X_testvalid, y_train, y_testvalid, s_train, s_testvalid = train_test_split(X, y, s, test_size=0.6, stratify=y, random_state=args.seed)
    X_test, X_val, y_test, y_val, s_test, s_val = train_test_split(X_testvalid, y_testvalid, s_testvalid, test_size=0.5, stratify=y_testvalid, random_state=args.seed)


    print(f"X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}, s_train.shape: {s_train.shape}")
    print(f"X_val.shape: {X_val.shape}, y_val.shape: {y_val.shape}, s_val.shape: {s_val.shape}")
    print(f"X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}, s_test.shape: {s_test.shape}")

    scaler = StandardScaler().fit(X_train)

    def scale_df(df, scaler):
        return pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)

    X_train = X_train.pipe(scale_df, scaler)
    X_val = X_val.pipe(scale_df, scaler)
    X_test = X_test.pipe(scale_df, scaler)

    train_data = PandasDataSet(X_train, y_train, s_train)
    val_data = PandasDataSet(X_val, y_val, s_val)
    test_data = PandasDataSet(X_test, y_test, s_test)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    clf = MLP(n_features=n_features, n_hidden=args.num_hidden, num_layers=args.num_layers).to(device)
    clf_criterion = nn.BCELoss()
    clf_optimizer = optim.Adam(clf.parameters(), lr=args.lr)

    print(clf)


    tbl = TableLogger(columns='Epoch,Acc(Tr|Val|Te),Ap, AUC, DP, prule, Eopp, Eodd, ABPC, ABCC', float_format='{:,.2f}'.format, colwidth={0: 5, 1: 17, 2: 17, 3: 17, 4: 17, 5: 17, 6: 17, 7: 17, 8: 17})

    for epoch in range(1, args.num_epochs+1):
        clf = train_epoch(model=clf, train_loader=train_loader, optimizer=clf_optimizer, criterion=clf_criterion, device=device)

        if epoch % 5 == 0 or epoch == 1 or epoch == args.num_epochs:
            train_metrics = test(model=clf, test_loader=train_loader, criterion=clf_criterion, device=device,  prefix="train")
            val_metrics   = test(model=clf, test_loader=val_loader, criterion=clf_criterion, device=device, prefix="val")
            test_metrics  =  test(model=clf, test_loader=test_loader, criterion=clf_criterion, device=device, prefix="test")

            metrics = train_metrics
            metrics.update(val_metrics)
            metrics.update(test_metrics)
            metrics["epoch"] = epoch

            acc  = "{:0>2.2f}|{:0>2.2f}|{:0>2.2f}".format(metrics["train_accuracy"], metrics["val_accuracy"], metrics["test_accuracy"])
            ap   = "{:0>2.2f}|{:0>2.2f}|{:0>2.2f}".format(metrics["train_ap"],       metrics["val_ap"],       metrics["test_ap"])
            auc  = "{:0>2.2f}|{:0>2.2f}|{:0>2.2f}".format(metrics["train_auc"],      metrics["val_auc"],      metrics["test_auc"])
            dp   = "{:0>2.2f}|{:0>2.2f}|{:0>2.2f}".format(metrics["train_dp"],       metrics["val_dp"],       metrics["test_dp"])
            prule= "{:0>2.2f}|{:0>2.2f}|{:0>2.2f}".format(metrics["train_prule"],    metrics["val_prule"],    metrics["test_prule"])
            eopp = "{:0>2.2f}|{:0>2.2f}|{:0>2.2f}".format(metrics["train_eopp"],     metrics["val_eopp"],     metrics["test_eopp"])
            eodd = "{:0>2.2f}|{:0>2.2f}|{:0>2.2f}".format(metrics["train_eodd"],     metrics["val_eodd"],     metrics["test_eodd"])
            abpc = "{:0>2.2f}|{:0>2.2f}|{:0>2.2f}".format(metrics["train_abpc"],     metrics["val_abpc"],     metrics["test_abpc"])
            abcc = "{:0>2.2f}|{:0>2.2f}|{:0>2.2f}".format(metrics["train_abcc"],     metrics["val_abcc"],     metrics["test_abcc"])

            tbl( epoch, acc, ap, auc, dp, prule, eopp, eodd, abpc, abcc)

    tbl.print_line( tbl.make_horizontal_border() )
