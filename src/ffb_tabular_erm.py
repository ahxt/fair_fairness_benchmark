
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


from dataset import load_census_income_kdd_data, load_census_income_kdd_data, load_folktables_income, preprocess_adult_data
from dataset import load_census_income_kdd_data, load_census_income_kdd_data, load_folktables_income, load_folktables_employment, load_folktables_income_5year, load_folktables_employment_5year
from utils import seed_everything, PandasDataSet, metric_evaluation
from networks import MLP


logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')



def train(model, train_loader, optimizer, criterion, epoch, device, args=None):
    model.train()
    for batch_idx, (data, target, sensitive) in enumerate(train_loader):
        data, target, sensitive = data.to(device), target.to(device), sensitive.to(device)
        optimizer.zero_grad()
        h, output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                num_epochs, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
    return model



def test(model, test_loader, criterion, device, args=None):
    model.eval()
    test_loss = 0
    correct = 0
    target_hat_list = []
    target_list = []
    sensitive_list = []

    with torch.no_grad():
        for data, target, sensitive in test_loader:
            data, target, sensitive = data.to(device), target.to(device), sensitive.to(device)
            h, output = model(data)

            test_loss += criterion(output, target).item()  # sum up batch loss
            target_hat_list.append(output.cpu().numpy())
            target_list.append(target.cpu().numpy())
            sensitive_list.append(sensitive.cpu().numpy())

    target_hat_list = np.concatenate(target_hat_list, axis=0)
    target_list = np.concatenate(target_list, axis=0)
    sensitive_list = np.concatenate(sensitive_list, axis=0)
    metric = metric_evaluation( y_gt=target_list, y_pre= target_hat_list, s=sensitive_list, prefix="test_")

    test_loss /= len(test_loader.dataset)

    print(metric)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default="/data/han/data/fairness")
    parser.add_argument('--dataset', type=str, default="adult", choices=["adult", "kdd", "folktables_income", "folktables_employment", "folktables_income_5year", "folktables_employment_5year"] )
    parser.add_argument('--model', type=str, default="MLP")
    parser.add_argument('--target_attr', type=str, default="target")
    parser.add_argument('--sensitive_attr', type=str, default="sex")

    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--num_hidden', type=int, default=128)

    parser.add_argument('--lam', type=float, default=1)
    parser.add_argument('--gam', type=float, default=20)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--eps', type=float, default=0.05)

    parser.add_argument('--seed', type=int, default=1314)
    parser.add_argument('--exp_name', type=str, default="hhh_torch")
    parser.add_argument('--log_dir', type=str, default="")
    parser.add_argument('--log_screen', type=str, default="True")
    parser.add_argument('--round', type=int, default=0)

    parser.add_argument('--reg', type=str, default="gap_dp")


    args = parser.parse_args()
    data_path = args.data_path
    dataset_name = args.dataset
    seed = args.seed
    log_screen = eval(args.log_screen)
    log_dir = args.log_dir
    target_attr = args.target_attr
    sensitive_attr = args.sensitive_attr
    exp_name = args.exp_name

    model = args.model
    target_attr = args.target_attr
    num_hidden = args.num_hidden
    reg = args.reg

    num_epochs = args.epoch
    batch_size = args.batch_size
    learning_rate = args.lr
    momentum = args.momentum

    round = args.round

    lam = args.lam
    gam = args.gam
    beta = args.beta
    eps= args.eps


    seed_everything(seed=seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info("args:{}".format(args))


    if dataset_name == "folktables_income":
        logger.info(f'Dataset: folktables_income')
        X, y, s = load_folktables_income(sensitive_attributes=sensitive_attr)

    elif dataset_name == "folktables_employment":
        logger.info(f'Dataset: folktables_employment')
        X, y, s = load_folktables_employment(sensitive_attributes=sensitive_attr)

    elif dataset_name == "folktables_income_5year":
        logger.info(f'Dataset: folktables_income_5year')
        X, y, s = load_folktables_income_5year(sensitive_attributes=sensitive_attr)

    elif dataset_name == "folktables_employment_5year":
        logger.info(f'Dataset: folktables_employment_5year')
        X, y, s = load_folktables_employment_5year(sensitive_attributes=sensitive_attr)

    elif dataset_name == "adult":
        logger.info(f'Dataset: adult')
        X_train, X_val, X_test, y_train, y_val, y_test, A_train, A_val, A_test = preprocess_adult_data(seed=42, path = '/data/han/data/fairness/adult', sensitive_attributes=sensitive_attr)
        X = pd.DataFrame( np.concatenate( [X_train, X_val, X_test] ) )
        y = pd.DataFrame( np.concatenate( [y_train, y_val, y_test] ) )[0]
        s = pd.DataFrame( np.concatenate( [A_train, A_val, A_test] ) )


    elif dataset_name == "kdd":
        logger.info(f'Dataset: kdd')
        X, y, s = load_census_income_kdd_data('/data/han/data/fairness/census-income-mld', sensitive_attributes=sensitive_attr)

    else:
        logger.info(f'Wrong dataset_name')


    logger.info(f'X.shape: {X.shape}')
    logger.info(f'y.shape: {y.shape}')
    logger.info(f's.shape: {s.shape}')
    logger.info(f's.shape: {s.value_counts().to_dict()}')
    logger.info(f's.shape: {y.value_counts().to_dict()}')



    n_features = X.shape[1]
    n_classes = len(np.unique(y))

    # split into train/val/test set
    X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(X, y, s, test_size=0.2, stratify=y, random_state=seed)
    X_train, X_val, y_train, y_val, s_train, s_val = train_test_split(X_train, y_train, s_train, test_size=0.1 / 0.8, stratify=y_train, random_state=seed)


    logger.info(f'X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}, s_train.shape: {s_train.shape}')
    logger.info(f'X_val.shape: {X_val.shape}, y_val.shape: {y_val.shape}, s_val.shape: {s_val.shape}')
    logger.info(f'X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}, s_test.shape: {s_test.shape}')


    scaler = StandardScaler().fit(X_train)
    scale_df = lambda df, scaler: pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
    X_train = X_train.pipe(scale_df, scaler) 
    X_val = X_val.pipe(scale_df, scaler) 
    X_test = X_test.pipe(scale_df, scaler)

    train_data = PandasDataSet(X_train, y_train, s_train)
    val_data = PandasDataSet(X_val, y_val, s_val)
    test_data = PandasDataSet(X_test, y_test, s_test)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    train_loader_no_shuffle = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader( val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader( test_data, batch_size=batch_size, shuffle=False)



    clf = MLP(n_features=n_features, n_hidden=num_hidden).to(device)
    clf_criterion = nn.BCELoss()
    # clf_criterion = nn.BCEWithLogitsLoss()
    clf_optimizer = optim.Adam(clf.parameters(), lr=learning_rate)


    for epoch in range(num_epochs):
        clf = train(model=clf, train_loader=train_loader, optimizer=clf_optimizer, criterion=clf_criterion, epoch=epoch, device=device, args=args )

        test(model=clf, test_loader=train_loader, criterion=clf_criterion, device=device, args=None)
        test(model=clf, test_loader=val_loader, criterion=clf_criterion, device=device, args=None)
        test(model=clf, test_loader=test_loader, criterion=clf_criterion, device=device, args=None)


    logger.info(f"done experiment")





