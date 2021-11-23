import numpy as np


def metrics_report(y_test_, y_pred_):
    """
    Generate detailed metrics for testing model performance
    
    """

    y_test = np.ravel(y_test_)
    y_pred = np.ravel(y_pred_)

    true_positives = np.sum(y_test * y_pred)
    false_positives = np.sum(np.abs(y_test - 1) * y_pred)
    true_negatives = np.sum((y_test - 1) * (y_pred - 1))
    false_negatives = np.sum(y_test * np.abs(y_pred - 1))

    accuracy = round((true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives), 4)

    precision = round(true_positives / (true_positives + false_positives), 4)
    recall = round(true_positives / (true_positives + false_negatives), 4)
    specificity = round(true_negatives / (true_negatives + false_positives), 4)
    npv = round(true_negatives / (true_negatives + false_negatives), 4)

    f1_1 = round(2 * (precision * recall) / (precision + recall), 4)
    f1_0 = round(2 * (specificity * npv) / (specificity + npv), 4)
    f1_macro = round((f1_1 + f1_0) / 2, 4)

    positives_percentage = np.count_nonzero(y_test) / y_test.size
    negatives_percentage = np.count_nonzero(y_test == 0) / y_test.size
    f1_weighted = round((f1_1 * positives_percentage + f1_0 * negatives_percentage), 4)

    micro_precision = accuracy
    micro_recall = accuracy
    f1_micro = round(2 * (micro_precision * micro_recall) / (micro_precision + micro_recall), 4)
    
    metrics_dict = dict(
        accuracy = accuracy, f1_macro = f1_macro,
        f1_weighted = f1_weighted, f1_micro = f1_micro,
        f1_1 = f1_1, f1_0 = f1_0,
        precision = precision, recall = recall,
        specificity = specificity, npv = npv,
        TP = int(true_positives), FP = int(false_positives), FN = int(false_negatives),
        TN = int(true_negatives),
    )
    
    for metric_name in metrics_dict.keys():
        if str(metrics_dict[metric_name]) == 'nan':
            metrics_dict[metric_name] = 0
            
    return metrics_dict
