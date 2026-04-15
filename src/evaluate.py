from copy import deepcopy
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, precision_score


EMOTIONTALK_LABELS = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _sigmoid_numpy(x):
    return 1.0 / (1.0 + np.exp(-x))


def multiclass_acc(preds, truths):
    preds = _to_numpy(preds)
    truths = _to_numpy(truths)
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def weighted_acc(preds, truths, verbose=False):
    preds = _to_numpy(preds).reshape(-1)
    truths = _to_numpy(truths).reshape(-1)

    total = len(preds)
    tp = 0
    tn = 0
    p = 0
    n = 0
    for i in range(total):
        if truths[i] == 0:
            n += 1
            if preds[i] == 0:
                tn += 1
        elif truths[i] == 1:
            p += 1
            if preds[i] == 1:
                tp += 1

    w_acc = (tp * n / max(p, 1) + tn) / (2 * max(n, 1))

    if verbose:
        fp = n - tn
        fn = p - tp
        recall = tp / (tp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        f1 = 2 * recall * precision / (recall + precision + 1e-8)
        print('TP=', tp, 'TN=', tn, 'FP=', fp, 'FN=', fn, 'P=', p, 'N=', n, 'Recall=', recall, "f1=", f1)

    return w_acc


def eval_mosei_senti(results, truths, exclude_zero=False):
    test_preds = _to_numpy(results).reshape(-1)
    test_truth = _to_numpy(truths).reshape(-1)

    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0 or (not exclude_zero)])

    test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
    test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
    test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

    mae = np.mean(np.absolute(test_preds - test_truth))
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    acc7 = multiclass_acc(test_preds_a7, test_truth_a7)
    acc5 = multiclass_acc(test_preds_a5, test_truth_a5)
    f1 = f1_score((test_truth[non_zeros] > 0), (test_preds[non_zeros] > 0), average='weighted')
    binary_truth = (test_truth[non_zeros] > 0)
    binary_preds = (test_preds[non_zeros] > 0)
    acc2 = accuracy_score(binary_truth, binary_preds)

    return mae, acc2, acc5, acc7, f1, corr


def eval_mosei_emo(preds, truths, threshold, verbose=False):
    preds = _to_numpy(preds)
    truths = _to_numpy(truths)

    total = preds.shape[0]
    num_emo = preds.shape[1]

    preds = _sigmoid_numpy(preds)

    aucs = roc_auc_score(truths, preds, labels=list(range(num_emo)), average=None).tolist()
    aucs.append(float(np.average(aucs)))

    preds_bin = deepcopy(preds)
    preds_bin[preds_bin > threshold] = 1
    preds_bin[preds_bin <= threshold] = 0

    accs = []
    f1s = []
    for emo_ind in range(num_emo):
        preds_i = preds_bin[:, emo_ind]
        truths_i = truths[:, emo_ind]
        accs.append(weighted_acc(preds_i, truths_i, verbose=verbose))
        f1s.append(f1_score(truths_i, preds_i, average='weighted', zero_division=0))

    accs.append(float(np.average(accs)))
    f1s.append(float(np.average(f1s)))

    acc_strict = 0
    acc_intersect = 0
    acc_subset = 0
    for i in range(total):
        if np.all(preds_bin[i] == truths[i]):
            acc_strict += 1
            acc_intersect += 1
            acc_subset += 1
        else:
            is_loose = False
            is_subset = False
            for j in range(num_emo):
                if preds_bin[i, j] == 1 and truths[i, j] == 1:
                    is_subset = True
                    is_loose = True
                elif preds_bin[i, j] == 1 and truths[i, j] == 0:
                    is_subset = False
                    break
            if is_subset:
                acc_subset += 1
            if is_loose:
                acc_intersect += 1

    acc_strict /= total
    acc_intersect /= total
    acc_subset /= total

    return accs, f1s, aucs, [acc_strict, acc_subset, acc_intersect]


def _list_to_metric_dict(values, class_names):
    out = {}
    for i, c in enumerate(class_names):
        out[c] = float(values[i])
    out["average"] = float(values[-1])
    return out


def eval_iemocap(preds, truths, best_thresholds=None, class_names=None):
    preds = _to_numpy(preds)
    truths = _to_numpy(truths)

    if class_names is None:
        if preds.shape[1] == 7:
            class_names = EMOTIONTALK_LABELS
        else:
            class_names = [f"class_{i}" for i in range(preds.shape[1])]

    num_emo = preds.shape[1]
    probs = _sigmoid_numpy(preds)

    try:
        aucs = roc_auc_score(truths, probs, labels=list(range(num_emo)), average=None).tolist()
    except Exception:
        aucs = [0.0] * num_emo
    aucs.append(float(np.average(aucs)))

    if best_thresholds is None:
        thresholds = np.arange(0.05, 1.00, 0.05)
        f1_grid = []
        for t in thresholds:
            preds_bin = deepcopy(probs)
            preds_bin[preds_bin > t] = 1
            preds_bin[preds_bin <= t] = 0

            this_f1s = []
            for i in range(num_emo):
                pred_i = preds_bin[:, i]
                truth_i = truths[:, i]
                this_f1s.append(f1_score(truth_i, pred_i, zero_division=0))
            f1_grid.append(this_f1s)

        f1_grid = np.array(f1_grid)
        best_thresholds = (np.argmax(f1_grid, axis=0) + 1) * 0.05

    preds_bin = deepcopy(probs)
    for i in range(num_emo):
        pred = preds_bin[:, i]
        pred[pred > best_thresholds[i]] = 1
        pred[pred <= best_thresholds[i]] = 0
        preds_bin[:, i] = pred

    accs = []
    recalls = []
    precisions = []
    f1s = []
    for i in range(num_emo):
        pred_i = preds_bin[:, i]
        truth_i = truths[:, i]

        accs.append(accuracy_score(truth_i, pred_i))
        recalls.append(recall_score(truth_i, pred_i, zero_division=0))
        precisions.append(precision_score(truth_i, pred_i, zero_division=0))
        f1s.append(f1_score(truth_i, pred_i, zero_division=0))

    accs.append(float(np.average(accs)))
    recalls.append(float(np.average(recalls)))
    precisions.append(float(np.average(precisions)))
    f1s.append(float(np.average(f1s)))

    stats = {
        "acc": _list_to_metric_dict(accs, class_names),
        "recall": _list_to_metric_dict(recalls, class_names),
        "precision": _list_to_metric_dict(precisions, class_names),
        "f1": _list_to_metric_dict(f1s, class_names),
        "auc": _list_to_metric_dict(aucs, class_names),
    }

    return stats, np.array(best_thresholds)


def eval_iemocap_ce(preds, truths):
    preds = _to_numpy(preds)
    truths = _to_numpy(truths)

    preds = preds.argmax(-1)
    acc = accuracy_score(truths, preds)
    f1 = f1_score(truths, preds, average='macro', zero_division=0)
    r = recall_score(truths, preds, average='macro', zero_division=0)
    p = precision_score(truths, preds, average='macro', zero_division=0)
    return acc, r, p, f1
