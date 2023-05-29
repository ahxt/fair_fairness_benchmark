import numpy as np
from sklearn import metrics
from scipy.stats import gaussian_kde
from statsmodels.distributions.empirical_distribution import ECDF

def CV():
    # CalderVerwer
    pass


def ABPC(y_pred, y_gt, sensitive_attribute, bw_method="scott", sample_n=5000):
    y_pred = y_pred.ravel()
    y_gt = y_gt.ravel()
    sensitive_attribute = sensitive_attribute.ravel()

    y_pre_1 = y_pred[sensitive_attribute == 1]
    y_pre_0 = y_pred[sensitive_attribute == 0]

    kde0 = gaussian_kde(y_pre_0, bw_method=bw_method)
    kde1 = gaussian_kde(y_pre_1, bw_method=bw_method)

    x = np.linspace(0, 1, sample_n)
    kde1_x = kde1(x)
    kde0_x = kde0(x)
    # area under the lower kde, from the first leftmost point to the first intersection point
    abpc = np.trapz(np.abs(kde0_x - kde1_x), x)

    abpc *= 100

    return abpc


def ABCC(y_pred, y_gt, sensitive_attribute):
    y_pred = y_pred.ravel()
    y_gt = y_gt.ravel()
    sensitive_attribute = sensitive_attribute.ravel()

    y_pre_1 = y_pred[sensitive_attribute == 1]
    y_pre_0 = y_pred[sensitive_attribute == 0]

    ecdf0 = ECDF(y_pre_0)
    ecdf1 = ECDF(y_pre_1)

    x = np.linspace(0, 1, 10000)
    ecdf0_x = ecdf0(x)
    ecdf1_x = ecdf1(x)

    # area under the lower kde, from the first leftmost point to the first intersection point
    abcc = np.trapz(np.abs(ecdf0_x - ecdf1_x), x)

    abcc *= 100

    return abcc


def auc_parity(y_pred: np.ndarray, y_gt: np.ndarray, sensitive_attribute: np.ndarray) -> float:
    """
    Calculates the AUC parity, which measures the difference in ROC AUC between different values of a
    binary sensitive attribute.

    Args:
        y_pred: A 1D array of predicted probabilities between 0 and 1 for each data point.
        y_gt: A 1D array of binary ground truth labels (0 or 1) for each data point.
        sensitive_attribute: A 1D array of binary values (0 or 1) indicating the sensitive attribute for each data point.

    Returns:
        A float value between 0 and 1 representing the absolute difference in ROC AUC between different
        values of the sensitive attribute.
    """
    # Flatten the 1D arrays to ensure they have the same shape.
    y_pred = y_pred.ravel()
    y_gt = y_gt.ravel()
    sensitive_attribute = sensitive_attribute.ravel()

    # Calculate the predicted values for the different values of the sensitive attribute.
    y_pre_1 = y_pred[sensitive_attribute == 1]
    y_pre_0 = y_pred[sensitive_attribute == 0]

    # Get the ground truth labels for the different values of the sensitive attribute.
    y_gt_1 = y_gt[sensitive_attribute == 1]
    y_gt_0 = y_gt[sensitive_attribute == 0]

    # Calculate the ROC AUC for the different values of the sensitive attribute.
    auc1 = metrics.roc_auc_score(y_true=y_gt_1, y_score=y_pre_1)
    auc0 = metrics.roc_auc_score(y_true=y_gt_0, y_score=y_pre_0)

    # Calculate the absolute difference in ROC AUC.
    parity = abs(auc1 - auc0)
    parity *= 100
    return parity


def demographic_parity(y_pred: np.ndarray, sensitive_attribute: np.ndarray, threshold: float = 0.5) -> float:
    """
    Calculates the demographic parity, which measures the difference in positive rate between different
    values of a binary sensitive attribute. The positive rate is defined as the proportion of data points
    that are predicted to be positive.

    Args:
        y_pred: A 1D array of predicted probabilities between 0 and 1 for each data point.
        sensitive_attribute: A 1D array of binary values (0 or 1) indicating the sensitive attribute for each data point.
        threshold: A float threshold value for converting predicted probabilities to binary predictions.

    Returns:
        A float value between 0 and 100 representing the percentage difference in positive rate
        between different values of the sensitive attribute.
    """
    # Convert predicted probabilities to binary predictions using the threshold.
    y_z_1 = y_pred[sensitive_attribute == 1] > threshold if threshold else y_pred[sensitive_attribute == 1]
    y_z_0 = y_pred[sensitive_attribute == 0] > threshold if threshold else y_pred[sensitive_attribute == 0]

    # If there are no data points in one of the sensitive attribute groups, return 0.
    if len(y_z_1) == 0 or len(y_z_0) == 0:
        return 0

    # Calculate the difference in positive rate.
    parity = abs(y_z_1.mean() - y_z_0.mean())
    parity *= 100
    return parity



def equal_opportunity(y_pred: np.ndarray, y_gt: np.ndarray, sensitive_attribute: np.ndarray, threshold: float = 0.5) -> float:
    """
    Calculates the equal opportunity, which measures the difference in true positive rate between different values
    of a binary sensitive attribute. The true positive rate is defined as the proportion of positive data points
    that are correctly predicted to be positive.

    Args:
        y_pred: A 1D array of predicted probabilities between 0 and 1 for each data point.
        y_gt: A 1D array of binary ground truth labels (0 or 1) for each data point.
        sensitive_attribute: A 1D array of binary values (0 or 1) indicating the sensitive attribute for each data point.
        threshold: A float threshold value for converting predicted probabilities to binary predictions.

    Returns:
        A float value between 0 and 100 representing the percentage difference in true positive rate
        between different values of the sensitive attribute.
    """
    # Select the predicted probabilities and sensitive attributes for the data points where the ground truth is positive.
    y_pred = y_pred[y_gt == 1]
    sensitive_attribute = sensitive_attribute[y_gt == 1]

    # Convert predicted probabilities to binary predictions using the threshold.
    y_z_1 = y_pred[sensitive_attribute == 1] > threshold if threshold else y_pred[sensitive_attribute == 1]
    y_z_0 = y_pred[sensitive_attribute == 0] > threshold if threshold else y_pred[sensitive_attribute == 0]

    # If there are no data points in one of the sensitive attribute groups, return 0.
    if len(y_z_1) == 0 or len(y_z_0) == 0:
        return 0

    # Calculate the difference in true positive rate.
    equality = abs(y_z_1.mean() - y_z_0.mean())
    equality *= 100
    return equality




def equalized_odds(y_pred: np.ndarray, y_gt: np.ndarray, sensitive_attribute: np.ndarray, threshold: float = 0.5) -> float:
    """
    Calculates the equalized odds, which measures the difference in true positive rate and false positive rate
    between different values of a binary sensitive attribute.

    Args:
        y_pred: A 1D array of predicted probabilities between 0 and 1 for each data point.
        y_gt: A 1D array of binary ground truth labels (0 or 1) for each data point.
        sensitive_attribute: A 1D array of binary values (0 or 1) indicating the sensitive attribute for each data point.
        threshold: A float threshold value for converting predicted probabilities to binary predictions.

    Returns:
        A float value between 0 and 100 representing the percentage difference in true positive rate and false positive rate
        between different values of the sensitive attribute.
    """
    # Make a copy of the predicted probabilities and sensitive attributes.
    y_pred_all = y_pred.copy()
    sensitive_attribute_all = sensitive_attribute.copy()

    # Select the predicted probabilities and sensitive attributes for the data points where the ground truth is positive.
    y_pred = y_pred_all[y_gt == 1]
    sensitive_attribute = sensitive_attribute_all[y_gt == 1]

    # Convert predicted probabilities to binary predictions using the threshold and calculate the difference in true positive rate.
    y_z_1 = y_pred[sensitive_attribute == 1] > threshold if threshold else y_pred[sensitive_attribute == 1]
    y_z_0 = y_pred[sensitive_attribute == 0] > threshold if threshold else y_pred[sensitive_attribute == 0]
    if len(y_z_1) == 0 or len(y_z_0) == 0:
        return 0
    equality = abs(y_z_1.mean() - y_z_0.mean())

    # Select the predicted probabilities and sensitive attributes for the data points where the ground truth is negative.
    y_pred = y_pred_all[y_gt == 0]
    sensitive_attribute = sensitive_attribute_all[y_gt == 0]

    # Convert predicted probabilities to binary predictions using the threshold and calculate the difference in false positive rate.
    y_z_1 = y_pred[sensitive_attribute == 1] > threshold if threshold else y_pred[sensitive_attribute == 1]
    y_z_0 = y_pred[sensitive_attribute == 0] > threshold if threshold else y_pred[sensitive_attribute == 0]
    if len(y_z_1) == 0 or len(y_z_0) == 0:
        return 0
    equality += abs(y_z_1.mean() - y_z_0.mean())

    # Multiply the difference in true positive rate and false positive rate by 100 to get a percentage difference and return it.
    equality *= 100
    return equality


def p_rule(y_pred: np.ndarray, sensitive_attribute: np.ndarray, threshold: float = 0.5) -> float:
    """
    Calculates the p-rule, which measures the ratio of the proportion of positive outcomes for the sensitive attribute
    group with the higher proportion to the proportion of positive outcomes for the sensitive attribute group with the
    lower proportion.

    Args:
        y_pred: A 1D array of predicted probabilities between 0 and 1 for each data point.
        sensitive_attribute: A 1D array of binary values (0 or 1) indicating the sensitive attribute for each data point.
        threshold: A float threshold value for converting predicted probabilities to binary predictions.

    Returns:
        A float value between 0 and 100 representing the p-rule. A value of 100 means perfect parity, while a value
        less than 100 means the model favors one sensitive attribute group over the other.
    """
    # Convert predicted probabilities to binary predictions using the threshold and select the binary predictions for each
    # sensitive attribute group.
    y_z_1 = y_pred[sensitive_attribute == 1] > threshold if threshold else y_pred[sensitive_attribute == 1]
    y_z_0 = y_pred[sensitive_attribute == 0] > threshold if threshold else y_pred[sensitive_attribute == 0]

    # Check that there are positive outcomes for both sensitive attribute groups.
    if len(y_z_1) == 0 or len(y_z_0) == 0:
        return 0

    # Calculate the odds ratio between the sensitive attribute groups and return the minimum of this ratio and its inverse,
    # multiplied by 100.
    odds = y_z_1.mean() / (y_z_0.mean() + 1e-8 )
    return np.nanmin([odds, 1 / (odds + 1e-8) ]) * 100


def predictive_parity_value(y_pred: np.ndarray, y_gt: np.ndarray, sensitive_attribute: np.ndarray, threshold: float = 0.5) -> float:
    """
    Calculates the predictive parity value, which measures the difference in prediction accuracy between
    different values of a sensitive attribute. The sensitive attribute must be binary.

    Args:
        y_pred: A 1D array of predicted probabilities between 0 and 1 for each data point.
        y_gt: A 1D array of ground truth labels (0 or 1) for each data point.
        sensitive_attribute: A 1D array of binary values (0 or 1) indicating the sensitive attribute for each data point.
        threshold: A float threshold value for converting predicted probabilities to binary predictions.

    Returns:
        A float value between 0 and 100 representing the percentage difference in prediction accuracy
        between different values of the sensitive attribute.
    """
    
    # Convert predicted probabilities to binary predictions based on threshold
    y_pred = (y_pred > threshold).astype(np.int32)

    # Filter ground truth and sensitive attribute based on binary predictions
    y_gt = y_gt[y_pred == 1]  # y_gt where y_pred == 1
    sensitive_attribute = sensitive_attribute[y_pred == 1]

    # Calculate the percentage difference in ground truth for each sensitive attribute value
    y_z_1 = y_gt[sensitive_attribute == 1] == 1  # y_gt where sensitive_attribute == 1 and y_gt == 1
    y_z_0 = y_gt[sensitive_attribute == 0] == 1  # y_gt where sensitive_attribute == 0 and y_gt == 1
    
    # Check if there is data for both sensitive attribute values
    if len(y_z_1) == 0 or len(y_z_0) == 0:
        return 0
    
    # Calculate predictive parity value as percentage difference between sensitive attribute values
    parity = abs(y_z_1.mean() - y_z_0.mean()) * 100
    return parity





def predictive_parity_value(y_pred: np.ndarray, y_gt: np.ndarray, sensitive_attribute: np.ndarray, threshold: float = 0.5) -> float:
    """
    Calculates the predictive parity value, which measures the difference in prediction accuracy between
    different values of a sensitive attribute. The sensitive attribute must be binary.

    Args:
        y_pred: A 1D array of predicted probabilities between 0 and 1 for each data point.
        y_gt: A 1D array of ground truth labels (0 or 1) for each data point.
        sensitive_attribute: A 1D array of binary values (0 or 1) indicating the sensitive attribute for each data point.
        threshold: A float threshold value for converting predicted probabilities to binary predictions.

    Returns:
        A float value between 0 and 100 representing the percentage difference in prediction accuracy
        between different values of the sensitive attribute.
    """
    
    # Convert predicted probabilities to binary predictions based on threshold
    y_pred = (y_pred > threshold).astype(np.int32)

    # Filter ground truth and sensitive attribute based on binary predictions
    y_gt = y_gt[y_pred == 1]  # y_gt where y_pred == 1
    sensitive_attribute = sensitive_attribute[y_pred == 1]

    # Calculate the percentage difference in ground truth for each sensitive attribute value
    y_z_1 = y_gt[sensitive_attribute == 1] == 1  # y_gt where sensitive_attribute == 1 and y_gt == 1
    y_z_0 = y_gt[sensitive_attribute == 0] == 1  # y_gt where sensitive_attribute == 0 and y_gt == 1
    
    # Check if there is data for both sensitive attribute values
    if len(y_z_1) == 0 or len(y_z_0) == 0:
        return 0
    
    # Calculate predictive parity value as percentage difference between sensitive attribute values
    parity = abs(y_z_1.mean() - y_z_0.mean()) * 100
    return parity


def balance_for_positive_class(y_pred: np.ndarray, y_gt: np.ndarray, sensitive_attribute: np.ndarray) -> float:
    """
    Calculates balance for negative class, which measures the difference in prediction accuracy between
    different values of a sensitive attribute for the negative class. The sensitive attribute must be binary.

    Args:
        y_pred: A 1D array of predicted probabilities for each data point. Values should be between 0 and 1.
        y_gt: A 1D array of ground truth labels for each data point. Values should be binary (0 or 1).
        sensitive_attribute: A 1D array of binary values indicating the sensitive attribute for each data point.

    Returns:
        A float value between 0 and 100 representing the percentage difference in prediction accuracy
        between different values of the sensitive attribute for the negative class.
    """

    # Filter predicted values, ground truth labels, and sensitive attribute based on ground truth labels where y_gt == 1
    y_pred = y_pred[y_gt == 1]
    sensitive_attribute = sensitive_attribute[y_gt == 1]

    # Filter predicted values for each sensitive attribute value where sensitive attribute == 1 and sensitive attribute == 1
    y_z_1 = y_pred[sensitive_attribute == 1]
    y_z_0 = y_pred[sensitive_attribute == 0]

    # Check if there is data for both sensitive attribute values
    if len(y_z_1) == 0 or len(y_z_0) == 0:
        return 0

    # Calculate balance for negative class as percentage difference between sensitive attribute values
    equality = abs(y_z_1.mean() - y_z_0.mean()) * 100
    return equality


def balance_for_negative_class(y_pred: np.ndarray, y_gt: np.ndarray, sensitive_attribute: np.ndarray) -> float:
    """
    Calculates balance for negative class, which measures the difference in prediction accuracy between
    different values of a sensitive attribute for the negative class. The sensitive attribute must be binary.

    Args:
        y_pred: A 1D array of predicted probabilities for each data point. Values should be between 0 and 1.
        y_gt: A 1D array of ground truth labels for each data point. Values should be binary (0 or 1).
        sensitive_attribute: A 1D array of binary values indicating the sensitive attribute for each data point.

    Returns:
        A float value between 0 and 100 representing the percentage difference in prediction accuracy
        between different values of the sensitive attribute for the negative class.
    """

    # Filter predicted values, ground truth labels, and sensitive attribute based on ground truth labels where y_gt == 0
    y_pred = y_pred[y_gt == 0]
    sensitive_attribute = sensitive_attribute[y_gt == 0]

    # Filter predicted values for each sensitive attribute value where sensitive attribute == 1 and sensitive attribute == 0
    y_z_1 = y_pred[sensitive_attribute == 1]
    y_z_0 = y_pred[sensitive_attribute == 0]

    # Check if there is data for both sensitive attribute values
    if len(y_z_1) == 0 or len(y_z_0) == 0:
        return 0

    # Calculate balance for negative class as percentage difference between sensitive attribute values
    equality = abs(y_z_1.mean() - y_z_0.mean()) * 100
    return equality




def accuracy_parity(y_pred, y_gt, sensitive_attribute):
    """
    Calculate the accuracy parity score between two groups defined by a sensitive attribute.

    Parameters:
    y_pred (numpy.ndarray): A one-dimensional array of predicted values between 0 and 1.
    y_gt (numpy.ndarray): A one-dimensional array of ground truth labels, where each label is either 0 or 1.
    sensitive_attribute (numpy.ndarray): A one-dimensional array of sensitive attributes, where each attribute is either 0 or 1.

    Returns:
    float: The absolute difference between the accuracy scores of the two groups defined by the sensitive attribute.
    """

    # Flatten the input arrays
    y_pred = y_pred.ravel()
    y_gt = y_gt.ravel()
    sensitive_attribute = sensitive_attribute.ravel()

    # Get predicted values and ground truth labels for data points where sensitive attribute is 1 or 0
    y_pre_1 = y_pred[sensitive_attribute == 1]
    y_pre_0 = y_pred[sensitive_attribute == 0]
    y_gt_1 = y_gt[sensitive_attribute == 1]
    y_gt_0 = y_gt[sensitive_attribute == 0]

    # Calculate accuracy scores for each group defined by sensitive attribute
    acc1 = metrics.accuracy_score(y_true=y_gt_1, y_pred=y_pre_1>0.5)
    acc0 = metrics.accuracy_score(y_true=y_gt_0, y_pred=y_pre_0>0.5)

    # Calculate the absolute difference between the two accuracy scores
    parity = abs(acc1 - acc0)
    parity *= 100

    # Return the accuracy parity value
    return parity


def metric_evaluation(y_gt, y_pre, s, s_pre=None, prefix=""):
    """
    Calculate various performance and fairness metrics based on ground truth labels, predicted values, and sensitive attributes.

    Parameters:
    y_gt (numpy.ndarray): A one-dimensional array of ground truth labels, where each label is either 0 or 1.
    y_pre (numpy.ndarray): A one-dimensional array of predicted values between 0 and 1.
    s (numpy.ndarray): A one-dimensional array of sensitive attributes, where each attribute is either 0 or 1.
    s_pre (numpy.ndarray): A one-dimensional array of predicted sensitive attributes, where each attribute is either 0 or 1. Default is None.
    prefix (str): A string to prefix all metric names with. Default is an empty string.

    Returns:
    dict: A dictionary that maps metric names to values.
    """

    # Flatten the input arrays
    y_gt = y_gt.ravel()
    y_pre = y_pre.ravel()
    s = s.ravel()

    # Calculate various performance metrics
    acc = metrics.accuracy_score(y_gt, y_pre > 0.5) * 100
    ap = metrics.average_precision_score(y_gt, y_pre) * 100
    precision = metrics.precision_score(y_gt, y_pre > 0.5, zero_division=0) * 100
    recall = metrics.recall_score(y_gt, y_pre > 0.5) * 100
    f1 = metrics.f1_score(y_gt, y_pre > 0.5) * 100
    auc = metrics.roc_auc_score(y_gt, y_pre) * 100

    # Calculate various fairness metrics
    dp = demographic_parity(y_pre, s, threshold=0.5)
    dpe = demographic_parity(y_pre, s, threshold=None)
    aucp = auc_parity(y_pre, y_gt, s)
    accp = accuracy_parity(y_pre, y_gt, s)
    ppv = predictive_parity_value(y_pre, y_gt, s)
    bfp = balance_for_positive_class(y_pre, y_gt, s)
    bfn = balance_for_negative_class(y_pre, y_gt, s)

    prule = p_rule(y_pre, s, threshold=0.5)
    prulee = p_rule(y_pre, s, threshold=None)

    eopp = equal_opportunity(y_pre, y_gt, s, threshold=0.5)
    eoppe = equal_opportunity(y_pre, y_gt, s, threshold=None)
    eodd = equalized_odds(y_pre, y_gt, s, threshold=0.5)
    eodde = equalized_odds(y_pre, y_gt, s, threshold=None)

    # abpc = ABPC(y_pre, y_gt, s)
    abcc = ABCC(y_pre, y_gt, s)
    # avg_alpha, af_list = alpha_fairness_metric( y_pre, y_gt, s, s_pre, sample_n=10, threshold=None )

    # Store metric names and values in a dictionary
    metric_name = ["acc", "ap", "precision", "recall", "f1", "auc", "dp", "dpe", "aucp", "eopp", "eoppe", "eodd", "eodde", "prule", "prulee", "abcc", "accp", "ppv", "bfp", "bfn"]
    metric_name = [prefix + "/" + x for x in metric_name]
    metric_val = [acc, ap, precision, recall, f1, auc, dp, dpe, aucp, eopp, eoppe, eodd, eodde, prule, prulee, abcc, accp, ppv, bfp, bfn]

    return dict(zip(metric_name, metric_val))













def demographic_parity_ref(y_logits, u):
    # y_ = (1.0 / (1.0 + np.exp(-y_logits)) > 0.5).astype(np.float32)
    y_ = y_logits > 0.5
    g, uc = np.zeros([2]), np.zeros([2])
    for i in range(u.shape[0]):
        if u[i] > 0:
            g[1] += y_[i]
            uc[1] += 1
        else:
            g[0] += y_[i]
            uc[0] += 1
    g = g / uc
    return np.abs(g[0] - g[1])



def acc_metric_evaluation(y_gt, y_pre, prefix=""):
    y_gt = y_gt.ravel()
    y_pre = y_pre.ravel()

    accuracy = metrics.accuracy_score(y_gt, y_pre > 0.5) * 100
    ap = metrics.average_precision_score(y_gt, y_pre) * 100

    metric_name = ["accuracy", "ap"]
    metric_name = [prefix + x for x in metric_name]
    metric_val = [accuracy, ap]

    return dict(zip(metric_name, metric_val))


def fairness_metric_evaluation(y_gt, y_pre, s, s_pre, prefix=""):
    """Compute all fairness metrics."""

    y_gt = y_gt.ravel()
    y_pre = y_pre.ravel()
    s = s.ravel()

    dp = demographic_parity(y_pre, s)
    dpe = demographic_parity(y_pre, s, threshold=None)
    dpauc = auc_parity(y_pre, y_gt, s)
    eopp = equal_opportunity(y_pre, y_gt, s)
    eoppe = equal_opportunity(y_pre, y_gt, s, threshold=None)
    eodd = equalized_odds(y_pre, y_gt, s)
    eodde = equalized_odds(y_pre, y_gt, s, threshold=None)
    prule = p_rule(y_pre, s)
    prulee = p_rule(y_pre, s, threshold=None)
    abpc = ABPC(y_pre, y_gt, s)
    abcc = ABCC(y_pre, y_gt, s)

    metric_name = ["dp", "dpe", "dpauc", "eopp", "eoppe", "eodd", "eodde" "prule", "prulee", "abpc", "abcc"]
    metric_name = [prefix + x for x in metric_name]
    metric_val = [dp, dpe, dpauc, eopp, eoppe, eodd, eodde, prule, prulee, abpc, abcc]

    return dict(zip(metric_name, metric_val))



def equalizied_opportunity_ref(y, y_logits, u):
    '''
    P(T = 1|Y = 1, S= s_i) = P(T = 1|Y = 1, S = s_j)
    '''
    # y_ = (1.0 / (1.0 + np.exp(-y_logits)) > 0.5).astype(np.float32)
    y_ = y_logits > 0.5
    g, uc = np.zeros([2]), np.zeros([2])
    for i in range(u.shape[0]):
        if y[i] < 0.999:
            continue
        if u[i] > 0:
            g[1] += y_[i]
            uc[1] += 1
        else:
            g[0] += y_[i]
            uc[0] += 1
    g = g / uc
    return np.abs(g[0] - g[1])


# def fairness_metric_evaluation(y_gt, y_pre, s, s_pre = None, prefix=""):

#     y_gt = y_gt.ravel()
#     y_pre = y_pre.ravel()
#     s= s.ravel()

#     dp = demographic_parity(y_pre, s)
#     dpe = demographic_parity(y_pre, s, threshold=None)
#     dpauc = auc_parity(y_pre, y_gt, s)
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

def predictive_parity_ref(target, pred, sensitive):
    """
    Check if P(T = 1|Y = 1, S= s_i) = P(T = 1|Y = 1, S = s_j)
    
    model: trained model
    dataset: test dataset
    sensitive: column name of the sensitive variable
    target: column name of the target of the prediction task
    """
    pred = (pred > 0.5).astype(int)
    p0 = target[ (sensitive == 0 ) & (pred == 1) ].mean()
    p1 = target[ (sensitive == 1 ) & (pred == 1) ].mean()
    
    return abs(p0-p1)


def equalized_odds_ref(y, y_logits, u):
    """Compute equalized odds of predictions."""
    # y_ = (1.0 / (1.0 + np.exp(-y_logits)) > 0.5).astype(np.float32)
    y_ = y_logits > 0.5
    g = np.zeros([2, 2])
    uc = np.zeros([2, 2])
    for i in range(u.shape[0]):
        if u[i] > 0:
            g[int(y[i])][1] += y_[i]
            uc[int(y[i])][1] += 1
        else:
            g[int(y[i])][0] += y_[i]
            uc[int(y[i])][0] += 1
    g = g / uc
    return np.abs(g[0, 1] - g[0, 0]) + np.abs(g[1, 1] - g[1, 0])


if __name__ == "__main__":
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([0.9, 0.4, 0.35, 0.8, 0.2, 0.6, 0.9, 0.99, 0.95, 0.3, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])
    s = np.array([0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

    print(demographic_parity(y_pred, s, threshold=0.5))
    print(demographic_parity_ref(y_pred, s))

    print(equal_opportunity(y_pred, y, s, threshold=0.5))
    print(equalizied_opportunity_ref(y, y_pred, s))

    print(equalized_odds(y_pred, y, s, threshold=0.5))
    print(equalized_odds_ref(y, y_pred, s))

    print(predictive_parity_value(y_pred, y, s, threshold=0.5))
    print(predictive_parity_ref(y, y_pred, s))    
