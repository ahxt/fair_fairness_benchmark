import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from tabulate import tabulate
import wandb


from dataset import load_census_income_kdd_data,load_census_income_kdd_data,load_adult_data,load_german_data, load_compas_data, load_german_data, load_bank_marketing_data, load_acs_data
from utils import seed_everything, PandasDataSet, print_metrics, clear_lines, InfiniteDataLoader
from metrics import metric_evaluation
from networks import MLP

from loss import DiffEOdd





def train_epoch(model, train_loader, optimizer, clf_criterion, fair_criterion, lambda1, device, args=None):
    model.train()
    for batch_idx, (data, target, sensitive) in enumerate(train_loader):
        data, target, sensitive = data.to(device), target.to(device), sensitive.to(device)
        optimizer.zero_grad()
        h, output = model(data)
        clf_loss = clf_criterion(output, target)
        fair_loss = fair_criterion(output, sensitive, target)
        loss = clf_loss + lambda1 * fair_loss
        loss.backward()
        optimizer.step()
    return model, loss.item(), clf_loss.item(), fair_loss.item()


def train_step(model, data, target, sensitive, scheduler, optimizer, clf_criterion, fair_criterion, lam, device, args=None):
    model.train()
    optimizer.zero_grad()
    h, output = model(data)
    clf_loss = clf_criterion(output, target)
    fair_loss = fair_criterion(output, sensitive, target)
    loss = clf_loss + lam * fair_loss
    loss.backward()
    optimizer.step()
    scheduler.step()
    return model, loss.item(), clf_loss.item(), fair_loss.item()

def test(model, test_loader, clf_criterion, fair_criterion, lam, device, prefix="test", args=None):
    model.eval()
    clf_loss = 0
    fair_loss = 0
    target_hat_list = []
    target_list = []
    sensitive_list = []

    with torch.no_grad():
        for data, target, sensitive in test_loader:
            data, target, sensitive = (data.to(device), target.to(device), sensitive.to(device))
            h, output = model(data)

            clf_loss += clf_criterion(output, target).item()
            fair_loss = fair_criterion(output, sensitive, target).item()
            target_hat_list.append(output.cpu().numpy())
            target_list.append(target.cpu().numpy())
            sensitive_list.append(sensitive.cpu().numpy())

    target_hat_list = np.concatenate(target_hat_list, axis=0)
    target_list = np.concatenate(target_list, axis=0)
    sensitive_list = np.concatenate(sensitive_list, axis=0)
    metric = metric_evaluation(y_gt=target_list, y_pre=target_hat_list, s=sensitive_list, prefix=f"{prefix}")

    clf_loss /= len(test_loader)
    fair_loss /= len(test_loader)
    
    metric[f"{prefix}/clf_loss"] = clf_loss
    metric[f"{prefix}/fair_loss"] = fair_loss
    metric[f"{prefix}/loss"] = clf_loss + lam*fair_loss

    return metric




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../datasets/adult/raw")
    parser.add_argument("--dataset", type=str,default="adult",choices=["adult","kdd","acs","german", "compas" ,"bank_marketing"], help="e.g. adult,kdd,acs,german,compas,bank_marketing")
    parser.add_argument("--model", type=str, default="diffeodd")
    parser.add_argument("--target_attr", type=str, default="income")
    parser.add_argument("--sensitive_attr", type=str, default="sex")
    parser.add_argument("--evaluation_metrics", type=str, default="acc,ap,dp,eopp,eodd", help="e.g. acc,ap,dp")
    parser.add_argument("--log_freq", type=int, default=1)

    parser.add_argument("--lam", type=float, default=1.0)

    parser.add_argument("--num_training_steps", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--evaluation_batch_size", type=int, default=1024000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--mlp_layers", type=str, default="512,256,64", help="e.g. 512,256,64")

    parser.add_argument("--seed", type=int, default=1314)
    parser.add_argument("--exp_name", type=str, default="uuid")
    parser.add_argument("--wandb_project", type=str, default="fair_fairness_benchmark")
    parser.add_argument("--job_id", type=str, default="000_00")


    args = parser.parse_args()
    wandb.init(project=args.wandb_project, config=args)
    table = tabulate([(k, v) for k, v in vars(args).items()], tablefmt='grid')
    print(table)




    seed_everything(seed=args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"


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
        X, y, s = load_acs_data( path= "../datasets/acs/raw", target_attr=args.target_attr, sensitive_attribute=args.sensitive_attr)

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    

    categorical_cols = X.select_dtypes("string").columns
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols)


    n_features = X.shape[1]
    n_classes = len(np.unique(y))

    X_train, X_testvalid, y_train, y_testvalid, s_train, s_testvalid = train_test_split(X, y, s, test_size=0.6, stratify=y, random_state=args.seed)
    X_test, X_val, y_test, y_val, s_test, s_val = train_test_split(X_testvalid, y_testvalid, s_testvalid, test_size=0.5, stratify=y_testvalid, random_state=args.seed)

    dataset_stats = {
        "dataset": args.dataset,
        "num_features": X.shape[1],
        "num_classes": len(np.unique(y)),
        "num_sensitive": len(np.unique(s)),
        "num_samples": X.shape[0],
        "num_train": X_train.shape[0],
        "num_val": X_val.shape[0],
        "num_test": X_test.shape[0],
        "num_y1": (y.values == 1).sum(),
        "num_y0": (y.values == 0).sum(),
        "num_s1": (s.values == 1).sum(),
        "num_s0": (s.values == 0).sum(),
    }

    wandb.config.update(dataset_stats)

    # Create the table using the tabulate function
    table = tabulate([(k, v) for k, v in dataset_stats.items()], tablefmt='grid')

    print(table)


    numurical_cols = X.select_dtypes("float32").columns
    if len(numurical_cols) > 0:
        # scaler = StandardScaler().fit(X[numurical_cols])

        def scale_df(df, scaler):
            return pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)

        scaler = StandardScaler().fit(X_train[numurical_cols])

        def scale_df(df, scaler):
            return pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)

        X_train[numurical_cols] = X_train[numurical_cols].pipe(scale_df, scaler)
        X_val[numurical_cols]   = X_val[numurical_cols].pipe(scale_df, scaler)
        X_test[numurical_cols]  = X_test[numurical_cols].pipe(scale_df, scaler)


    train_data = PandasDataSet(X_train, y_train, s_train)
    val_data = PandasDataSet(X_val, y_val, s_val)
    test_data = PandasDataSet(X_test, y_test, s_test)


    train_infinite_loader = InfiniteDataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    train_loader = DataLoader(train_data, batch_size=args.evaluation_batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.evaluation_batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.evaluation_batch_size, shuffle=False)




    mlp_layers = [int(x) for x in args.mlp_layers.split(",")]
    net = MLP(n_features=n_features, num_classes=1, mlp_layers=mlp_layers ).to(device)
    clf_criterion = nn.BCELoss()
    fair_criterion = DiffEOdd()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

    print(net)
    wandb.watch( models=net, criterion=clf_criterion, log="all", log_freq=args.log_freq)


    logs = []
    headers = ["Step(Tr|Val|Te)"] + args.evaluation_metrics.split(",")

    # evaluation_metrics = "ap,dp,prule"


    for step, (X, y, s) in enumerate(train_infinite_loader):
        if step >= args.num_training_steps:
            break

        X, y, s = X.to(device), y.to(device), s.to(device)
        net, loss, clf_loss, fair_loss = train_step(model=net, data=X, target=y, sensitive=s, optimizer=optimizer, scheduler=scheduler,  clf_criterion=clf_criterion, fair_criterion=fair_criterion, lam=args.lam,  device=device)


        if step % args.log_freq == 0 or step == 1 or step == args.num_training_steps:
            train_metrics = test(model=net, test_loader=train_loader, clf_criterion=clf_criterion, fair_criterion=fair_criterion, lam=args.lam, device=device, prefix="train")
            val_metrics   = test(model=net, test_loader=val_loader,   clf_criterion=clf_criterion, fair_criterion=fair_criterion, lam=args.lam, device=device, prefix="val")
            test_metrics  = test(model=net, test_loader=test_loader,  clf_criterion=clf_criterion, fair_criterion=fair_criterion, lam=args.lam, device=device, prefix="test")
            res_dict = {}
            res_dict["training/step"] = step
            res_dict["training/loss"] = loss
            res_dict["training/clf_loss"] = clf_loss
            res_dict["training/fair_loss"] = fair_loss
            res_dict["training/lr"] = optimizer.param_groups[0]["lr"]
            res_dict.update(train_metrics)
            res_dict.update(val_metrics)
            res_dict.update(test_metrics)

            # for wandb logging
            wandb.log(res_dict)

            # for printing
            if step % (args.log_freq*10) == 0:
                res = print_metrics(res_dict, args.evaluation_metrics, train=False)
                logs.append( [ step, *res] )
                if  step > 3:
                    clear_lines(len(logs)*2 + 1)
                table = tabulate(logs, headers=headers, tablefmt="grid", floatfmt="02.2f")
                print(table)



    wandb.finish()