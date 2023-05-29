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
import time

from dataset import load_census_income_kdd_data,load_census_income_kdd_data,load_adult_data,load_german_data, load_compas_data, load_german_data, load_bank_marketing_data, load_acs_data
from utils import seed_everything, PandasDataSet, print_metrics, clear_lines, InfiniteDataLoader
from metrics import metric_evaluation
from networks import MLP, LAFTR


def train(model, data_loader, optimizer, device):
    model.train()

    for batch_idx, (X, y, s) in enumerate(data_loader):
        X, y, s = X.to(device), y.to(device), s.to(device)

        optimizer.zero_grad()

        loss = model.loss(X, y, s)

        loss.backward()
        optimizer.step()        
    
    return model
    


def test(model, test_loader, device, prefix="test", args=None):
    model.eval()
    test_loss = 0
    target_hat_list = []
    target_list = []
    sensitive_list = []

    with torch.no_grad():
        for data, target, sensitive in test_loader:
            data, target, sensitive = (data.to(device), target.to(device), sensitive.to(device))
            h, decoded, output, adv_pred = model(data, sensitive)
            test_loss += model.loss(data, target, sensitive).item()
            target_hat_list.append(output.cpu().numpy())
            target_list.append(target.cpu().numpy())
            sensitive_list.append(sensitive.cpu().numpy())

    target_hat_list = np.concatenate(target_hat_list, axis=0)
    target_list = np.concatenate(target_list, axis=0)
    sensitive_list = np.concatenate(sensitive_list, axis=0)
    metric = metric_evaluation(y_gt=target_list, y_pre=target_hat_list, s=sensitive_list, prefix=f"{prefix}")

    test_loss /= len(test_loader.dataset)
    

    metric[f"{prefix}/loss"] = test_loss


    return metric


def train_step(model, data, target, sensitive, optimizer, scheduler, lam=None, device=None, args=None):
    model.train()
    optimizer.zero_grad()
    loss = model.loss(data, target, sensitive)
    loss.backward()
    optimizer.step()        
    scheduler.step()
    return model, loss.item(), None, None




if __name__ == '__main__':
    parser = argparse.ArgumentParser()


    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--clf_num_epochs', type=int, default=10)
    parser.add_argument('--adv_num_epochs', type=int, default=10)
    parser.add_argument('--num_hidden', type=int, default=512)



    parser.add_argument("--data_path", type=str, default="../datasets/adult/raw")
    parser.add_argument("--dataset", type=str,default="adult",choices=["adult","kdd","acs","german", "compas" ,"bank_marketing"], help="e.g. adult,kdd,acs,german,compas,bank_marketing")
    parser.add_argument("--model", type=str, default="diffdp")
    parser.add_argument("--target_attr", type=str, default="income")
    parser.add_argument("--sensitive_attr", type=str, default="sex")
    parser.add_argument("--evaluation_metrics", type=str, default="acc,ap,dp,eopp,eodd", help="e.g. acc,ap,dp")
    parser.add_argument("--log_freq", type=int, default=1)

    parser.add_argument("--lam", type=float, default=1.0)

    parser.add_argument("--num_training_steps", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--evaluation_batch_size", type=int, default=1024000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--mlp_layers", type=str, default="512,256,64", help="e.g. 512,256,64")

    parser.add_argument("--seed", type=int, default=1314)
    parser.add_argument("--exp_name", type=str, default="uuid")
    parser.add_argument("--wandb_project", type=str, default="fair_fairness_benchmark")


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
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader( val_data, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader( test_data, batch_size=args.batch_size, shuffle=False)


    encoder = MLP(n_features=n_features, num_classes=1, mlp_layers=[128]).to(device)
    decoder = MLP(n_features=128, num_classes=1, mlp_layers=[n_features]).to(device)
    adversary = MLP(n_features=128, num_classes=1, mlp_layers=[64]).to(device)
    classifier = MLP(n_features=128, num_classes=1, mlp_layers=[64]).to(device)

    # model definition
    laftr = LAFTR( encoder, decoder, adversary, classifier, rec_loss=None, adv_loss=None, classif_loss=None, A_x=0.1, A_y=1, A_z=100).to(device)

    optimizer = optim.Adam( laftr.parameters(), lr=args.lr )
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)


    print(laftr)
    logs = []
    headers = ["Step(Tr|Val|Te)"] + args.evaluation_metrics.split(",")




    for step, (X, y, s) in enumerate(train_infinite_loader):
        if step >= args.num_training_steps:
            break

        X, y, s = X.to(device), y.to(device), s.to(device) 
        laftr, loss, _, _ = train_step(model=laftr, data=X, target=y, sensitive=s, optimizer=optimizer, scheduler=scheduler, lam=args.lam,  device=device)

        # advfairnet = train_step(advfairnet, train_infinite_loader, optimizer, clf_criterion, adv_criterion, device, args)


        if step % args.log_freq == 0 or step == 1 or step == args.num_training_steps:
            train_metrics = test(model=laftr, test_loader=train_loader, device=device,  prefix="train")
            val_metrics   = test(model=laftr, test_loader=val_loader,  device=device, prefix="val")
            test_metrics  =  test(model=laftr, test_loader=test_loader, device=device, prefix="test")
            res_dict = {}
            res_dict["training/step"] = step
            res_dict["training/loss"] = loss
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
