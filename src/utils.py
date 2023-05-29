import numpy as np
import pandas as pd
import numpy as np
from numpy import random
import os
import sys
import torch
import numpy as np
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader




def clear_lines(lines):
    for _ in range(lines):
        sys.stdout.write("\033[K")  # Clear to the end of line
        sys.stdout.write("\033[F")  # Move cursor up one line



def print_metrics(metrics_dict, metrics_print = "ap,dp", train=True):

    output = []
    for metric in metrics_print.split(","):
        if train == True:
            output.append( "{:0>2.2f}|{:0>2.2f}|{:0>2.2f}".format(metrics_dict["train/"+metric], metrics_dict["val/"+metric], metrics_dict["test/"+metric]) )
        else:
            output.append( "{:0>2.2f}|{:0>2.2f}".format(metrics_dict["val/"+metric], metrics_dict["test/"+metric]) )
    
    return tuple(output)




class PandasDataSet(TensorDataset):
    def __init__(self, *dataframes):
        tensors = (self._df_to_tensor(df) for df in dataframes)
        super(PandasDataSet, self).__init__(*tensors)

    def _df_to_tensor(self, df):
        if isinstance(df, pd.Series):
            df = df.to_frame("dummy")
        return torch.from_numpy(df.values).float()


def InfiniteDataLoader(dataset, batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=True):
    while True:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
        for data in data_loader:
            yield data



def seed_everything(seed=1314):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def save_checkpoint(dir, epoch, name="checkpoint", **kwargs):
    state = {"epoch": epoch}
    state.update(kwargs)
    filepath = os.path.join(dir, "%s-%d.pt" % (name, epoch))
    torch.save(state, filepath)


# def plot_distributions(y_pred, s, Z_pred=None, epoch=None):

#     fig, ax = plt.subplots(1, 1, figsize=(10, 4), sharey=True)

#     df = pd.DataFrame(s.copy())
#     sensitive_attribute = list(df.columns)
#     df["y_pred"] = y_pred


#     for label, df in df.groupby(sensitive_attribute):
#         sns.kdeplot(df["y_pred"], ax=ax, label=label, shade=True)
#     # sns.kdeplot(df['y_pred'], ax=ax, label=label, shade=True)

#     ax.set_xlim(0, 1)
#     fig.tight_layout()
#     return fig
