# 

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import time
import argparse

import pandas as pd
import wandb
from src import dataset
from src import networks


def load_dataset(dataset_name, data_path, sensitive_attr, target_attr=None):
    if dataset_name == "adult":
        return dataset.load_adult_data()
    elif dataset_name == "german":
        return dataset.load_german_data()
    elif dataset_name == "bank_marketing":
        return dataset.load_bank_marketing_data()
    elif dataset_name == "compas":
        return dataset.load_compas_data()
    elif dataset_name == "acs":
        return dataset.load_acs_data()
    elif dataset_name == "census_income_kdd":
        return dataset.load_census_income_kdd_data()
    else:
        raise ValueError("dataset_name not found")
    

def load_model(model_name):
    if model_name == "diffdp":
        return networks.AdvFairNet
    else:
        raise ValueError("model_name not found")


def parse_args():
    parser = argparse.ArgumentParser()
    # dataset parameters
    parser.add_argument("--dataset_path", type=str, default="/data/han/data/fairness")
    parser.add_argument("--dataset_name", type=str, default="adult")
    parser.add_argument("--target_attr", type=str, default="income")
    parser.add_argument("--sensitive_attr", type=str, default="sex")
    
    # model parameters
    parser.add_argument("--deiasing_method", type=str, default="diffdp")
    parser.add_argument("--mlp_arch", type=str, default="MLP")


    # training parameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)


    parser.add_argument("--wandb_path", type=str, default="/data/han/data/fairness")
    parser.add_argument("--wandb_project", type=str, default="/data/han/data/fairness")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    wandb.init(project=args.wandb_project, config=args)

    X, y, s = load_dataset(args.dataset_name)
    model = load_model(args.deiasing_method)



