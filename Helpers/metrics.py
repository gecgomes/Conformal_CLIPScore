import torch
import numpy as np
import scipy.stats
from scipy.stats import ttest_ind, mannwhitneyu

def false_negative_rate(prediction_set, Y):
    n = len(prediction_set)
    losses = torch.zeros(n)
    for i in range(n):
        T = prediction_set[i]
        label_size = Y[i].float().sum()
        if label_size != 0:
            losses[i] = 1 - (T[Y[i].to(torch.bool)] == True).float().sum()/label_size
        else:
            losses[i] = 0
    return losses.mean().item()

def false_discovery_rate(prediction_set, Y):
    n = len(prediction_set)
    losses = torch.zeros(n)
    for i in range(n):
        T = prediction_set[i]
        set_size = T.float().sum()
        if set_size != 0:
            losses[i] = 1 - (T[Y[i].to(torch.bool)] == True).float().sum()/set_size
        else:
            losses[i] = 0
    return losses.mean().item()

def false_positive_rate(prediction_set, Y):
    n = len(prediction_set)
    losses = torch.zeros(n)
    for i in range(n):
        T = prediction_set[i]
        Y[i] = Y[i].to(torch.bool)
        negative_label_size = (~Y[i]).sum()
        if negative_label_size != 0:
            losses[i] = 1 - (T[~Y[i]] == False).float().sum()/negative_label_size
        else:
            losses[i] = 0
    return losses.mean().item()


def Upearson_risk(prediction_mean, prediction_std,Y):
    Ups = UPS(prediction_mean, prediction_std,Y)
    return 1 - torch.relu(torch.tensor(Ups)).numpy()

def UPS(prediction_mean, prediction_std,Y):
    abs_diff = np.abs(prediction_mean - Y)
    pearson = scipy.stats.pearsonr(abs_diff,prediction_std)[0]
    return pearson

def Pearson_risk(prediction_mean, prediction_std,Y):
    pearson = scipy.stats.pearsonr(prediction_mean,Y)[0]
    return 1 - torch.relu(torch.tensor(pearson)).numpy()

def print_metrics(human_scores, system_scores, system_std):
    kendalb = scipy.stats.kendalltau(human_scores,system_scores,variant = "b")[0]
    kendalc = scipy.stats.kendalltau(human_scores,system_scores,variant = "c")[0]
    pearson = scipy.stats.pearsonr(human_scores,system_scores)[0]
    spearman = scipy.stats.spearmanr(human_scores,system_scores)[0]
    ups = UPS(system_scores,system_std,human_scores)
    print("Kendal B: {}; Kendal C: {}; Pearson: {}, Spearman: {}, UPS: {}".format(kendalb,kendalc,pearson,spearman,ups))
    print("{},{},{},{},{}".format(ups, kendalb,kendalc,pearson,spearman))

def print_whoops_metrics(strange_std,natural_std, normal_std):
    mean_list = []
    pvalue_list = []
    print("AVG Strange Std: ", np.mean(strange_std))
    print("AVG Natural Std: ", np.mean(natural_std))
    print("AVG Normal Std: ", np.mean(normal_std))
    for ss_type,std_dist, std_orig in zip(["Strange Vs Natural","Strange Vs Normal"],[strange_std,strange_std],[natural_std,normal_std]):
        print(" === {} ===".format(ss_type))
        # 1. Calculate the mean difference between the two distributions
        mean_difference = np.mean(std_dist) - np.mean(std_orig)
        mean_list.append(mean_difference)
        # 2. Perform t-test assuming equal variances
        t_statistic, p_value_ttest = ttest_ind(std_dist, std_orig)

        # 3. Perform Welch’s t-test (does not assume equal variances)
        welch_statistic, p_value_welch = ttest_ind(std_dist, std_orig, equal_var=False)

        # 4. Perform Mann-Whitney U test
        mw_statistic, p_value_mannwhitney = mannwhitneyu(std_dist, std_orig, alternative='two-sided')
        pvalue_list.append(p_value_welch)

        # Report results
        print("Mean Difference between Variance Distributions:", mean_difference)
        print("\nT-Test Results:")
        print(f"Statistic: {t_statistic}, p-value: {p_value_ttest}")

        print("\nWelch’s T-Test Results:")
        print(f"Statistic: {welch_statistic}, p-value: {p_value_welch}")

        print("\nMann-Whitney U Test Results:")
        print(f"Statistic: {mw_statistic}, p-value: {p_value_mannwhitney}")
        print("\n")
    print("ACC_1 (Var_Natural < Var_Strange): ", np.mean(natural_std < strange_std))
    print("ACC_2 (Var_Normal < Var_Strange): ", np.mean(normal_std < strange_std))
    print("{},{},{},{},{},{}".format(np.mean(natural_std < strange_std), mean_list[0], pvalue_list[0], np.mean(normal_std < strange_std) , mean_list[1], pvalue_list[1]))

def foil_accuracy(prediction_set, Y):
    """
    Returns:
        accuracy (float): Overall fraction of correct predictions.
        recall (float):   Among actual foils (all-zero), fraction correctly predicted as foil.
        precision (float): Among predicted foils, fraction that are actually foil.
    """
    n = len(prediction_set)
    T = np.array([(prediction_set[i] == 1).any() for i in range(n)])
    label = np.array([(Y[i] == 1).any() for i in range(n)])
    accuracy = np.mean(T == label)
    foil = np.sum((T == label) & (label == 1)) / np.sum(label == 1)
    correct = np.sum((T == label) & (label == 0)) / np.sum(label == 0)
    
    return accuracy, foil, correct

def location_accuracy(prediction_set, Y, k=2):
    n = len(prediction_set)
    result = []
    for t,l in zip(prediction_set,Y):
        if (l.sum() > 0):
            if k == 1:
                result.append(((t*l).sum()).item())
            else:
                result.append(((t*l).sum()/l.sum()).item())
        elif t.sum() == 0:
            result.append(1)
        else:
            result.append(0)
    return np.mean(result)

def location_precision(prediction_set, Y):
    n = len(prediction_set)
    result = []
    for t, l in zip(prediction_set, Y):
        if t.sum() > 0:
            result.append(((t * l).sum() / t.sum()).item())  # Precision = TP / (TP + FP)
        else:
            result.append(0)  # Undefined precision, treat as 0
    return np.mean(result)

def location_recall(prediction_set, Y):
    n = len(prediction_set)
    result = []
    for t, l in zip(prediction_set, Y):
        if l.sum() > 0:
            result.append(((t * l).sum() / l.sum()).item())  # Recall = TP / (TP + FN)
        else:
            result.append(1)  # Perfect recall when there are no true positives to consider
    return np.mean(result)

def location_f1(prediction_set, Y):
    precision = location_precision(prediction_set, Y)
    recall = location_recall(prediction_set, Y)
    if precision + recall > 0:
        return 2 * (precision * recall) / (precision + recall)  # F1 = 2 * (Precision * Recall) / (Precision + Recall)
    else:
        return 0  # Undefined F1, treat as 0

def location_accuracy_foil_only(prediction_set, Y, k = 2):
    n = len(prediction_set)
    valid = np.array([bool((Y[i] == 1).any()) for i in range(n)])
    T = []
    for v,i in zip(prediction_set,valid):
        if i == 1:
            T.append(v)
    labels = []
    for v,i in zip(Y,valid):
        if i == 1:
            labels.append(v)
    return location_accuracy(T,labels, k=k)

def location_precision_foil_only(prediction_set, Y):
    n = len(prediction_set)
    valid = np.array([bool((Y[i] == 1).any()) for i in range(n)])
    T = []
    for v,i in zip(prediction_set,valid):
        if i == 1:
            T.append(v)
    labels = []
    for v,i in zip(Y,valid):
        if i == 1:
            labels.append(v)
    return location_precision(T,labels)

def location_recall_foil_only(prediction_set, Y):
    n = len(prediction_set)
    valid = np.array([bool((Y[i] == 1).any()) for i in range(n)])
    T = []
    for v,i in zip(prediction_set,valid):
        if i == 1:
            T.append(v)
    labels = []
    for v,i in zip(Y,valid):
        if i == 1:
            labels.append(v)
    return location_recall(T,labels)

def location_f1_foil_only(prediction_set, Y):
    n = len(prediction_set)
    valid = np.array([bool((Y[i] == 1).any()) for i in range(n)])
    T = []
    for v,i in zip(prediction_set,valid):
        if i == 1:
            T.append(v)
    labels = []
    for v,i in zip(Y,valid):
        if i == 1:
            labels.append(v)
    return location_f1(T,labels)

def top_location_accuracy(f_s, labels, lamhat, k=2):
    top_predictions = []
    for item_2 in f_s:
        # Get the top-k values and indices
        vals, ids = torch.topk(item_2, k=1)
        # Create a binary prediction tensor
        pred = torch.zeros_like(item_2)
        for val, idx in zip(vals, ids):
            if val > lamhat:
                pred[idx] = 1
        top_predictions.append(pred)
    return location_accuracy(top_predictions, labels, k=k)

def top_location_accuracy_foil_only(f_s, labels, lamhat, k = 2):
    top_predictions = []
    for item_2 in f_s:
        # Get the top-k values and indices
        vals, ids = torch.topk(item_2, k=1)
        # Create a binary prediction tensor
        pred = torch.zeros_like(item_2)
        for val, idx in zip(vals, ids):
            if val > lamhat:
                pred[idx] = 1
        top_predictions.append(pred)
    return location_accuracy_foil_only(top_predictions, labels, k= k)