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
import torchvision
from torchvision import transforms
from torch.utils.data import TensorDataset

from dataset import load_utkface_data, load_celeba_data
from utils import seed_everything, PandasDataSet, print_metrics, clear_lines, InfiniteDataLoader
from metrics import metric_evaluation
from networks import MLP, resnet18_encoder, resnet152_encoder, resnet101_encoder, resnet34_encoder, resnet50_encoder, resnext101_32x8d_encoder, resnext50_32x4d_encoder, wide_resnet101_2_encoder, wide_resnet50_2_encoder, swin_t_encoder
from loss import DiffDP


tfms = transforms.Compose([
    # transforms.Resize((256, 256)),
    # transforms.Resize((128, 128)),
    # transforms.Resize((224, 224)),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



def train_step(model, data, target, sensitive, scheduler, optimizer, clf_criterion, fair_criterion, lam, device, args=None):
    model.train()
    optimizer.zero_grad()
    h, output = model(data)
    clf_loss = clf_criterion(output, target)
    fair_loss = fair_criterion(output, sensitive)
    loss = clf_loss + lam * fair_loss
    loss.backward()
    optimizer.step()
    scheduler.step()
    return model, loss.item(), clf_loss.item(), fair_loss.item()



def test(model, test_loader, clf_criterion, fair_criterion, lam, device, target_index = 31, sensitive_index = 20, prefix="test", args=None):
# def test(model, test_loader, clf_criterion, fair_criterion, lam, device, prefix="test", args=None):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        target_hat_list = []
        target_list = []
        sensitive_list = []

        with torch.no_grad():
            for X, attr in test_loader:
                
                data = X.to( device )
                target = attr[:,target_index].to(  device )
                target = target.unsqueeze(1).type(torch.float)
                sensitive = attr[:,sensitive_index].to(  device )
                sensitive = sensitive.unsqueeze(1).type(torch.float)

                h, output = model(data)
                test_loss += clf_criterion(output, target).item()
                fair_loss = fair_criterion(output, sensitive).item()
                target_hat_list.append(output.cpu().numpy())
                target_list.append(target.cpu().numpy())
                sensitive_list.append(sensitive.cpu().numpy())

        target_hat_list = np.concatenate(target_hat_list, axis=0)
        target_list = np.concatenate(target_list, axis=0)
        sensitive_list = np.concatenate(sensitive_list, axis=0)

        metric = metric_evaluation(y_gt=target_list, y_pre=target_hat_list, s=sensitive_list, prefix=f"{prefix}")

        test_loss /= len(test_loader)
        fair_loss /= len(test_loader)
        
        metric[f"{prefix}/loss"] = test_loss + lam * fair_loss
        metric[f"{prefix}/clf_loss"] = test_loss
        metric[f"{prefix}/fair_loss"] = fair_loss

    return metric




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../datasets/", help="Path to the dataset folder")
    parser.add_argument("--dataset", type=str, default="utkface", choices=["utkface","celeba"], help="Choose a dataset from the available options: utkface, celeba, acs")
    parser.add_argument("--model", type=str, default="erm", help="Model type")
    parser.add_argument("--architecture", type=str, default="resnet18", help="the architecture of the model")
    parser.add_argument("--target_attr", type=str, default="Smiling", help="Target attribute for prediction")
    parser.add_argument("--sensitive_attr", type=str, default="Gender", help="Sensitive attribute for fairness analysis")
    parser.add_argument("--evaluation_metrics", type=str, default="acc,ap,dp,eopp,eodd", help="Evaluation metrics separated by commas, e.g., acc,ap,dp")
    parser.add_argument("--log_freq", type=int, default=1, help="Logging frequency")

    parser.add_argument("--lam", type=float, default=1.0)

    parser.add_argument("--num_training_steps", type=int, default=150, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--evaluation_batch_size", type=int, default=2048, help="Batch size for evaluation")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension size")
    
    parser.add_argument("--seed", type=int, default=1314, help="Random seed for reproducibility")
    parser.add_argument("--exp_name", type=str, default="uuid", help="Experiment name")
    parser.add_argument("--wandb_project", type=str, default="fair_fairness_benchmark", help="Weights & Biases project name")


    # Parse the command-line arguments
    args = parser.parse_args()

    # Initialize the Weights & Biases experiment
    wandb.init(project=args.wandb_project, config=args)

    # Create a table of the arguments and their values
    table = tabulate([(k, v) for k, v in vars(args).items()], tablefmt='grid')
    print(table)


    seed_everything(seed=args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"




    if args.dataset == "celeba_torchvision":

        if args.target_attr == "Smiling":
            target_index = 31
            print( "target_attr is Smiling!" )
        elif args.target_attr == "Wavy_Hair":
            target_index = 33
            print( "target_attr is Wavy_Hair!" )
        elif args.target_attr == "Attractive":
            target_index = 2
            print( "target_attr is Attractive!" )
        else:
            NotImplementedError("target_attr is not implemented!")



        if args.sensitive_attr == "Gender":
            sensitive_index = 20
            print( "sensitive_attr is Gender!" )
        elif args.sensitive_attr == "Young":
            sensitive_index = 39
            print( "sensitive_attr is Young!" )
        else:
            NotImplementedError("sensitive_attr is not implemented!")


        train_dataset = torchvision.datasets.CelebA(root=args.data_path, split = 'train', target_type = 'attr', transform = tfms, download = False)
        val_dataset = torchvision.datasets.CelebA(root=args.data_path, split = 'valid', target_type = 'attr', transform = tfms, download = False)
        test_dataset = torchvision.datasets.CelebA(root=args.data_path, split = 'test', target_type = 'attr', transform = tfms, download = False)

        pd_attr = pd.DataFrame(train_dataset.attr, columns=train_dataset.attr_names[:40])
        y = pd_attr.iloc[:,target_index]
        s = pd_attr.iloc[:,sensitive_index]
    
    elif args.dataset == "celeba":
        X, attr = load_celeba_data( path= "../datasets/celeba/raw", sensitive_attribute=args.sensitive_attr)

        X_np = np.stack( X["pixels"].to_list() )
        attr_np = attr.to_numpy()

        print( "X.shape: ", X.shape )
        print( "attr_np.shape: ", attr_np.shape )


        X_train, X_testvalid, attr_train, attr_testvalid = train_test_split(X_np, attr_np, test_size=0.2, stratify=attr_np, random_state=args.seed)
        X_test, X_val, attr_test, attr_val = train_test_split(X_testvalid, attr_testvalid, test_size=0.5, stratify=attr_testvalid, random_state=args.seed)

        print( "X_train.shape: ", X_train.shape )
        print( "X_val.shape: ", X_val.shape )
        print( "X_test.shape: ", X_test.shape )
        # dataset = TensorDataset(inps, tgts)
        X_train, attr_train = torch.from_numpy(X_train).float(), torch.from_numpy(attr_train).float()
        X_val, attr_val = torch.from_numpy(X_val).float(), torch.from_numpy(attr_val).float()
        X_test, attr_test = torch.from_numpy(X_test).float(), torch.from_numpy(attr_test).float()


        train_dataset = TensorDataset(X_train, attr_train)
        val_dataset = TensorDataset(X_val, attr_val)
        test_dataset = TensorDataset(X_test, attr_test)

        if args.target_attr == "Smiling":
            target_index = 0
            print( "target_attr is Smiling!" )
        elif args.target_attr == "Wavy_Hair":
            target_index = 1
            print( "target_attr is Wavy_Hair!" )
        elif args.target_attr == "Attractive":
            target_index = 2
            print( "target_attr is Attractive!" )
        else:
            NotImplementedError("target_attr is not implemented!")



        if args.sensitive_attr == "Gender":
            sensitive_index = 3
            print( "sensitive_attr is Gender!" )
        elif args.sensitive_attr == "Young":
            sensitive_index = 4
            print( "sensitive_attr is Young!" )
        else:
            NotImplementedError("sensitive_attr is not implemented!")
        y = attr.iloc[:,target_index]
        s = attr.iloc[:,sensitive_index]




    elif args.dataset == "utkface":
        X, attr = load_utkface_data( path= "../datasets/utkface/raw", sensitive_attribute=args.sensitive_attr)

        X_np = np.stack( X["pixels"].to_list() )
        attr_np = attr.to_numpy()

        print( "X.shape: ", X.shape )
        print( "attr_np.shape: ", attr_np.shape )


        X_train, X_testvalid, attr_train, attr_testvalid = train_test_split(X_np, attr_np, test_size=0.2, stratify=attr_np, random_state=args.seed)
        X_test, X_val, attr_test, attr_val = train_test_split(X_testvalid, attr_testvalid, test_size=0.5, stratify=attr_testvalid, random_state=args.seed)

        print( "X_train.shape: ", X_train.shape )
        print( "X_val.shape: ", X_val.shape )
        print( "X_test.shape: ", X_test.shape )
        # dataset = TensorDataset(inps, tgts)
        X_train, attr_train = torch.from_numpy(X_train).float(), torch.from_numpy(attr_train).float()
        X_val, attr_val = torch.from_numpy(X_val).float(), torch.from_numpy(attr_val).float()
        X_test, attr_test = torch.from_numpy(X_test).float(), torch.from_numpy(attr_test).float()


        train_dataset = TensorDataset(X_train, attr_train)
        val_dataset = TensorDataset(X_val, attr_val)
        test_dataset = TensorDataset(X_test, attr_test)

        # train_infinite_loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        # train_loader = DataLoader(train_dataset, batch_size=args.evaluation_batch_size, shuffle=True)
        # val_loader = DataLoader(val_dataset, batch_size=args.evaluation_batch_size, shuffle=False)
        # test_loader = DataLoader(test_dataset, batch_size=args.evaluation_batch_size, shuffle=False)
    

        target_index = 0

        if args.sensitive_attr == "Gender":
            sensitive_index = 1
            print( "sensitive_attr is Gender!" )
        elif args.sensitive_attr == "Race":
            sensitive_index = 2
            print( "sensitive_attr is Race!" )
        else:
            NotImplementedError("sensitive_attr is not implemented!")

        y = attr.iloc[:,target_index]
        s = attr.iloc[:,sensitive_index]



    dataset_stats = {
        "dataset": args.dataset,
        "num_classes": len(np.unique(y)),
        "num_sensitive": len(np.unique(s)),
        "num_samples": len(train_dataset)+len(val_dataset)+len(test_dataset),
        "num_train": len(train_dataset),
        "num_val": len(val_dataset),
        "num_test": len(test_dataset),
        "num_y1": (y.values == 1).sum(),
        "num_y0": (y.values == 0).sum(),
        "num_s1": (s.values == 1).sum(),
        "num_s0": (s.values == 0).sum(),
    }

    wandb.config.update(dataset_stats)

    # Create the table using the tabulate function
    table = tabulate([(k, v) for k, v in dataset_stats.items()], tablefmt='grid')
    print(table)


    train_infinite_loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=4, drop_last=False, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, batch_size=args.evaluation_batch_size, num_workers=4, drop_last=False, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=args.evaluation_batch_size, num_workers=4, drop_last=False, pin_memory=True)


    if args.architecture == "resnet18":
        net = resnet18_encoder(pretrained=True).to(device)
    elif args.architecture == "resnet34":
        net = resnet34_encoder(pretrained=True).to(device)
    elif args.architecture == "resnet50":
        net = resnet50_encoder(pretrained=True).to(device)
    elif args.architecture == "resnet101":
        net = resnet101_encoder(pretrained=True).to(device)
    elif args.architecture == "resnet152":
        net = resnet152_encoder(pretrained=True).to(device)
    elif args.architecture == "resnext50":
        net = resnext50_32x4d_encoder(pretrained=True).to(device)
    elif args.architecture == "resnext101":
        net = resnext101_32x8d_encoder(pretrained=True).to(device)
    elif args.architecture == "wide_resnet50":
        net = wide_resnet50_2_encoder(pretrained=True).to(device)
    elif args.architecture == "wide_resnet101":
        net = wide_resnet101_2_encoder(pretrained=True).to(device)
    elif args.architecture == "swin_t":
        net = swin_t_encoder(pretrained=True).to(device)
    else:
        NotImplementedError("archtecture is not implemented!")


  


    criterion = nn.BCELoss()
    fair_criterion = DiffDP()
    optimizer = optim.Adam(net.parameters(), lr = args.lr)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

    print(net)



    logs = []
    headers = ["Step(Val|Te)"] + args.evaluation_metrics.split(",")


    for step, (X, attr) in enumerate( train_infinite_loader ):
            if step >= args.num_training_steps:
                break

            attr = attr.type( torch.float32 )
            X = X.to( device )
            y = attr[:,target_index].to(  device )
            y = y.unsqueeze(1)
            s = attr[:,sensitive_index].to(  device )
            s = s.unsqueeze(1)

            net, loss, clf_loss, fair_loss = train_step(model=net, data=X, target=y, sensitive=s, optimizer=optimizer, scheduler=scheduler,  clf_criterion=criterion, fair_criterion=fair_criterion, lam=args.lam, device=device)

            if step % args.log_freq == 0 or step == 1 or step == args.num_training_steps:
            # if epoch % args.log_freq == 0:

                # train_metrics = test(model=net, test_loader=train_loader, criterion=criterion, device=device,  prefix="train")
                val_metrics   = test(model=net, test_loader=val_loader, clf_criterion=criterion, fair_criterion=fair_criterion, lam=args.lam, target_index = target_index, sensitive_index = sensitive_index, device=device, prefix="val")
                test_metrics  =  test(model=net, test_loader=test_loader, clf_criterion=criterion, fair_criterion=fair_criterion, lam=args.lam, target_index = target_index, sensitive_index = sensitive_index, device=device, prefix="test")
                # train_metrics = val_metrics
                res_dict = {}
                res_dict["training/step"] = step
                res_dict["training/loss"] = loss
                res_dict["training/clf_loss"] = clf_loss
                res_dict["training/fair_loss"] = fair_loss
                res_dict["training/lr"] = optimizer.param_groups[0]["lr"]
                # res_dict.update(train_metrics)
                res_dict.update(val_metrics)
                res_dict.update(test_metrics)

                # for wandb logging
                wandb.log(res_dict, step=step)

                # for printing
                if step % (args.log_freq*10) == 0:
                # if True:
                    res = print_metrics(res_dict, args.evaluation_metrics, train=False)
                    logs.append( [ step, *res] )
                    if  step > 3:
                        clear_lines(len(logs)*2 + 1)
                    table = tabulate(logs, headers=headers, tablefmt="grid", floatfmt="02.2f")
                    print(table)



    wandb.finish()


