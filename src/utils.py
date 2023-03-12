import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
import os
import torch
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from statsmodels.distributions.empirical_distribution import ECDF
from torch.utils.data import TensorDataset


# def alpha_fairness( y_gt, y_pred, s_gt, s_pred, alpha=None, threshold=None ):

#     s_pre_alpha = s_pred <= alpha
#     s_pre_1_alpha = s_pred >= 1 - alpha

#     # mask = s_pre_50 + s_pre_100
#     mask = s_pre_alpha + s_pre_1_alpha

#     y_pred_tmp = y_pred[mask]
#     y_gt_tmp = y_gt[mask]
#     s_gt_tmp = s_gt[mask]
#     s_pred_tmp = s_pred[mask]

#     y_z_1 = y_pred_tmp[s_gt_tmp == 1] > threshold if threshold else y_pred_tmp[s_gt_tmp == 1]
#     y_z_0 = y_pred_tmp[s_gt_tmp == 0] > threshold if threshold else y_pred_tmp[s_gt_tmp == 0]

#     alpha_fairness_parity = abs(y_z_1.mean() - y_z_0.mean())

#     return alpha_fairness_parity


def fairness_metric_evaluation(y_gt, y_pre, s, s_pre, prefix=""):

    y_gt = y_gt.ravel()
    y_pre = y_pre.ravel()
    s = s.ravel()

    dp = demographic_parity(y_pre, s)
    dpe = demographic_parity(y_pre, s, threshold=None)
    dpauc = demographic_parity_auc(y_pre, y_gt, s)
    eo = equal_opportunity(y_pre, y_gt, s)
    eoe = equal_opportunity(y_pre, y_gt, s, threshold=None)
    prule = p_rule(y_pre, s)
    prulee = p_rule(y_pre, s, threshold=None)
    abpc = ABPC(y_pre, y_gt, s)
    abcc = ABCC(y_pre, y_gt, s)

    metric_name = ["dp", "dpe", "dpauc", "eo", "eoe", "prule", "prulee", "abpc", "abcc"]
    metric_name = [prefix + x for x in metric_name]
    metric_val = [dp, dpe, dpauc, eo, eoe, prule, prulee, abpc, abcc]

    return dict(zip(metric_name, metric_val))


def ABPC(y_pred, y_gt, z_values, bw_method="scott", sample_n=5000):
    y_pred = y_pred.ravel()
    y_gt = y_gt.ravel()
    z_values = z_values.ravel()

    y_pre_1 = y_pred[z_values == 1]
    y_pre_0 = y_pred[z_values == 0]

    kde0 = gaussian_kde(y_pre_0, bw_method=bw_method)
    kde1 = gaussian_kde(y_pre_1, bw_method=bw_method)

    x = np.linspace(0, 1, sample_n)
    kde1_x = kde1(x)
    kde0_x = kde0(x)
    # area under the lower kde, from the first leftmost point to the first intersection point
    abpc = np.trapz(np.abs(kde0_x - kde1_x), x)

    return abpc


def ABCC(y_pred, y_gt, z_values):

    y_pred = y_pred.ravel()
    y_gt = y_gt.ravel()
    z_values = z_values.ravel()

    y_pre_1 = y_pred[z_values == 1]
    y_pre_0 = y_pred[z_values == 0]

    ecdf0 = ECDF(y_pre_0)
    ecdf1 = ECDF(y_pre_1)

    x = np.linspace(0, 1, 10000)
    ecdf0_x = ecdf0(x)
    ecdf1_x = ecdf1(x)

    # area under the lower kde, from the first leftmost point to the first intersection point
    abcc = np.trapz(np.abs(ecdf0_x - ecdf1_x), x)

    return abcc


def demographic_parity_auc(y_pred, y_gt, z_values):
    y_pred = y_pred.ravel()
    y_gt = y_gt.ravel()
    z_values = z_values.ravel()
    # print( y_pred.shape, y_gt.shape, z_values.shape )

    y_pre_1 = y_pred[z_values == 1]
    y_pre_0 = y_pred[z_values == 0]

    y_gt_1 = y_gt[z_values == 1]
    y_gt_0 = y_gt[z_values == 0]

    auc1 = metrics.roc_auc_score(y_true=y_gt_1, y_score=y_pre_1)
    auc0 = metrics.roc_auc_score(y_true=y_gt_0, y_score=y_pre_0)

    auc_parity = abs(auc1 - auc0)
    return auc_parity
    # return 0


def demographic_parity(y_pred, z_values, threshold=0.5):
    y_z_1 = y_pred[z_values == 1] > threshold if threshold else y_pred[z_values == 1]
    y_z_0 = y_pred[z_values == 0] > threshold if threshold else y_pred[z_values == 0]
    parity = abs(y_z_1.mean() - y_z_0.mean())
    return parity


def equal_opportunity(y_pred, y_gt, z_values, threshold=0.5):
    y_pred = y_pred[y_gt == 1]  # y_gt == 1
    z_values = z_values[y_gt == 1]
    y_z_1 = y_pred[z_values == 1] > threshold if threshold else y_pred[z_values == 1]
    y_z_0 = y_pred[z_values == 0] > threshold if threshold else y_pred[z_values == 0]
    equality = abs(y_z_1.mean() - y_z_0.mean())
    return equality


def equalized_odds(y_pred, y_gt, z_values, threshold=0.5):
    y_z_1 = y_pred[z_values == 1] > threshold if threshold else y_pred[z_values == 1]
    y_z_0 = y_pred[z_values == 0] > threshold if threshold else y_pred[z_values == 0]
    y_gt_1 = y_gt[z_values == 1]
    y_gt_0 = y_gt[z_values == 0]
    equality = abs(y_z_1[y_gt_1 == 1].mean() - y_z_0[y_gt_0 == 1].mean())
    return equality


def p_rule(y_pred, z_values, threshold=0.5):
    y_z_1 = y_pred[z_values == 1] > threshold if threshold else y_pred[z_values == 1]
    y_z_0 = y_pred[z_values == 0] > threshold if threshold else y_pred[z_values == 0]
    odds = y_z_1.mean() / y_z_0.mean()
    return np.nanmin([odds, 1 / odds]) * 100


def plot_distributions(y_pred, s, Z_pred=None, epoch=None):

    fig, ax = plt.subplots(1, 1, figsize=(10, 4), sharey=True)

    df = pd.DataFrame(s.copy())
    sensitive_attribute = list(df.columns)
    df["y_pred"] = y_pred

    # print( df )
    # print(sensitive_attribute)

    for label, df in df.groupby(sensitive_attribute):
        sns.kdeplot(df["y_pred"], ax=ax, label=label, shade=True)
    # sns.kdeplot(df['y_pred'], ax=ax, label=label, shade=True)

    ax.set_xlim(0, 1)
    fig.tight_layout()
    return fig


def seed_everything(seed=1314):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def metric_evaluation(y_gt, y_pre, s, s_pre=None, prefix=""):

    y_gt = y_gt.ravel()
    y_pre = y_pre.ravel()
    s = s.ravel()

    accuracy = metrics.accuracy_score(y_gt, y_pre > 0.5) * 100
    ap = metrics.average_precision_score(y_gt, y_pre) * 100
    auc = metrics.roc_auc_score(y_gt, y_pre) * 100
    dp = demographic_parity(y_pre, s, threshold=0.5)
    dpe = demographic_parity(y_pre, s, threshold=None)
    dpauc = demographic_parity_auc(y_pre, y_gt, s)
    eo = equal_opportunity(y_pre, y_gt, s, threshold=0.5)
    eoe = equal_opportunity(y_pre, y_gt, s, threshold=None)
    prule = p_rule(y_pre, s)
    prulee = p_rule(y_pre, s, threshold=None)
    abpc = ABPC(y_pre, y_gt, s)
    abcc = ABCC(y_pre, y_gt, s)
    # avg_alpha, af_list = alpha_fairness_metric( y_pre, y_gt, s, s_pre, sample_n=10, threshold=None )

    metric_name = ["accuracy", "ap", "auc", "dp", "dpe", "dpauc", "eo", "eoe", "prule", "prulee", "abpc", "abcc"]
    metric_name = [prefix + x for x in metric_name]
    metric_val = [accuracy, ap, auc, dp, dpe, dpauc, eo, eoe, prule, prulee, abpc, abcc]

    return dict(zip(metric_name, metric_val))


def acc_metric_evaluation(y_gt, y_pre, prefix=""):

    y_gt = y_gt.ravel()
    y_pre = y_pre.ravel()

    accuracy = metrics.accuracy_score(y_gt, y_pre > 0.5) * 100
    ap = metrics.average_precision_score(y_gt, y_pre) * 100

    metric_name = [
        "accuracy",
        "ap",
    ]
    metric_name = [prefix + x for x in metric_name]
    metric_val = [accuracy, ap]

    return dict(zip(metric_name, metric_val))


# def fairness_metric_evaluation(y_gt, y_pre, s, s_pre = None, prefix=""):

#     y_gt = y_gt.ravel()
#     y_pre = y_pre.ravel()
#     s= s.ravel()

#     dp = demographic_parity(y_pre, s)
#     dpe = demographic_parity(y_pre, s, threshold=None)
#     dpauc = demographic_parity_auc(y_pre, y_gt, s)
#     eo = equal_opportunity(y_pre, y_gt, s)
#     eoe = equal_opportunity(y_pre, y_gt, s, threshold=None)
#     prule = p_rule(y_pre, s)
#     prulee = p_rule(y_pre, s, threshold=None)
#     abpc = ABPC( y_pre, y_gt, s)
#     abcc = ABCC( y_pre, y_gt, s)
#     avg_alpha, af_list = alpha_fairness_metric( y_gt, y_pre, s, s_pre, sample_n=11, threshold=None )

#     metric_name = [ "dp", "dpe", "dpauc", "eo", "eoe", "prule", "prulee", "abpc", "abcc", "avg_alpha", "af_list" ]
#     metric_name = [ prefix+x for x in metric_name]
#     metric_val = [ dp, dpe, dpauc, eo, eoe, prule, prulee, abpc, abcc, avg_alpha, af_list ]

#     return dict( zip( metric_name, metric_val))


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

    # print( df )
    # print(sensitive_attribute)

    for label, df in df.groupby(sensitive_attribute):
        sns.kdeplot(df["y_pred"], ax=ax, label=label, shade=True)
    # sns.kdeplot(df['y_pred'], ax=ax, label=label, shade=True)

    ax.set_xlim(0, 1)
    fig.tight_layout()
    return fig
