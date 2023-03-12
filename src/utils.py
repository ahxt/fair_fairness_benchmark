import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset


class PandasDataSet(TensorDataset):
    def __init__(self, *dataframes):
        tensors = (self._df_to_tensor(df) for df in dataframes)
        super(PandasDataSet, self).__init__(*tensors)

    def _df_to_tensor(self, df):
        if isinstance(df, pd.Series):
            df = df.to_frame("dummy")
        return torch.from_numpy(df.values).float()


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


def plot_distributions(y_pred, s, Z_pred=None, epoch=None):

    fig, ax = plt.subplots(1, 1, figsize=(10, 4), sharey=True)

    df = pd.DataFrame(s.copy())
    sensitive_attribute = list(df.columns)
    df["y_pred"] = y_pred


    for label, df in df.groupby(sensitive_attribute):
        sns.kdeplot(df["y_pred"], ax=ax, label=label, shade=True)
    # sns.kdeplot(df['y_pred'], ax=ax, label=label, shade=True)

    ax.set_xlim(0, 1)
    fig.tight_layout()
    return fig
