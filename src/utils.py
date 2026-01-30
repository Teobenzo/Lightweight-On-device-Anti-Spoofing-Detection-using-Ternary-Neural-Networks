import os
import torch
import random
import GPUtil
import yaml
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, balanced_accuracy_score
import pandas as pd
import torch.nn as nn


# VARIOUS UTILITIES

def seed_everything(seed: int):
    """
    Set seed for everything.
    :param seed: seed value
    :type seed: int
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def set_gpu(id=-1):
    """
    Set GPU device or select the one with the lowest memory usage (None for CPU-only)

    :param id: if specified, corresponds to the GPU index desired.
    """
    if id is None:
        # CPU only
        print('GPU not selected')
        os.environ["CUDA_VISIBLE_DEVICES"] = str(-1)
    else:
        # -1 for automatic choice
        device = id if id != -1 else GPUtil.getFirstAvailable(order='memory')[0]
        try:
            name = GPUtil.getGPUs()[device].name
        except IndexError:
            print('The selected GPU does not exist. Switching to the most available one.')
            device = GPUtil.getFirstAvailable(order='memory')[0]
            name = GPUtil.getGPUs()[device].name
        print('GPU selected: %d - %s' % (device, name))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    # # Set memory growth
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)
    return device

def ensure_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Cartella creata: {folder_path}")
    else:
        print(f"Cartella già esistente: {folder_path}")

def read_yaml(config_path):
    """
    Read YAML file.

    :param config_path: path to the YAML config file.
    :type config_path: str
    :return: dictionary correspondent to YAML content
    :rtype dict
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def pad(x, max_len=64600):
    # Pad by repeating data
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

def _get_robust_suffix(cfg: dict) -> str:
    # Helper to get suffix for robustness tests configuration
    tags = []
    if cfg.get('apply_mulaw'): tags.append('mulaw')
    if cfg.get('apply_G722'): tags.append('g722')
    if cfg.get('apply_RIR'): tags.append('rir')
    if cfg.get('apply_noise_Inj_10db'): tags.append('noise10db')
    if cfg.get('apply_noise_Inj_20db'): tags.append('noise20db')
    if cfg.get('apply_opus'): tags.append('opus')
    if cfg.get('apply_vorbis'): tags.append('vorbis')
    if cfg.get('apply_env_noise_20db_SNR'): tags.append('env20')
    if cfg.get('apply_env_noise_15db_SNR'): tags.append('env15')
    if cfg.get('apply_white_noise_20db_SNR'): tags.append('wn20')
    if cfg.get('apply_white_noise_10db_SNR'): tags.append('wn10')
    return '_'.join(tags)


# PLOTS

def plot_roc_curve(labels, pred, save_dir, file_name, legend=None):
    """
    Plot ROC curve.

    :param labels: groundtruth labels
    :type labels: list
    :param pred: predicted score
    :type pred: list
    :param save_dir: directory where to save the plot
    :type save_dir: str
    :param file_name: name of the file
    :type file_name: str
    :param legend: if True, add legend to the plot
    :type legend: bool

    :return: optimal_thr: optimal threshold of decision (computed through EER)
    """
    # labels and pred bust be given in (N, ) shape

    labels = np.asarray(labels).ravel()
    pred = np.asarray(pred).ravel()

    # Case single-class: no ROC/EER
    if np.unique(labels).size < 2:
        print('Warning: only one class present in the labels. ROC/EER not computable.')
        print('Default threshold used = 0.5. AUC/EER = NaN.\n')
        return 0.5, float('nan'), float('nan')


    def tpr10(y_true, y_pred):
        '''
        Compute TPR at 10% of FPR
        TPR = True Positive Rate
        FPR = False Positive Rate
        '''
        fpr, tpr, thr = roc_curve(y_true, y_pred)  # thr = thresholds
        fp_sort = sorted(fpr)
        tp_sort = sorted(tpr)
        tpr_ind = [i for (i, val) in enumerate(fp_sort) if val >= 0.1][0]
        tpr01 = tp_sort[tpr_ind]
        return tpr01

    lw = 3

    fpr, tpr, thr = roc_curve(labels, pred)
    rocauc = auc(fpr, tpr)
    fnr = 1 - tpr
    optimal_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = fpr[optimal_idx]
    optimal_thr = thr[optimal_idx]  # EER corresponding thr

    # PRINTS AND PLOTS
    print('TPR10 = {:.3f}'.format(tpr10(labels, pred)))
    print('AUC = {:.3f}'.format(rocauc))
    print('EER = {:.3f}'.format(eer))
    print()

    # ROC PLOT
    plt.figure(figsize=(8, 6))
    if legend:
        plt.plot(fpr, tpr, lw=lw, label=legend + '\n - AUC = %0.2f\n - EER = %0.2f\n' % (rocauc, eer))
    else:
        plt.plot(fpr, tpr, lw=lw, label=' - AUC = %0.2f \n - EER = %0.2f\n' % (rocauc, eer))
    plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
    plt.title('ROC curve', fontsize=20)
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.03])
    plt.xlabel(r'$\mathrm{False\;Positive\;Rate}$', fontsize=18)
    plt.ylabel(r'$\mathrm{True\;Positive\;Rate}$', fontsize=18)
    plt.legend(loc="lower right", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True)
    # plt.show()

    # Save the plot
    save_path = os.path.join(save_dir, f'roc_curve_{file_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Roc Curve saved to {save_path}')
    # plt.clf()
    plt.close()

    # HISTOGRAM FOR METRICS
    metrics_hist = [rocauc, 1 - eer]
    labels_hist = ['AUC', '1 - EER']
    plt.figure(figsize=(8, 6))
    plt.bar(labels_hist, metrics_hist, color=['skyblue', 'salmon', 'lightgreen'])
    plt.ylim(0, 1)
    plt.title('Performance Metrics Comparison')
    plt.ylabel('Metric Value')
    save_path = os.path.join(save_dir, f'histogram_metrics_{file_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Histograms of the metrics saved to {save_path}')
    # plt.clf()
    plt.close()

    return optimal_thr, rocauc, eer


def plot_confusion_matrix(y_true, y_pred, save_dir, file_name, d_args, normalize=True, cmap=plt.cm.Blues, thr=0.5):
    """
    Plot confusion matrix.

    :param y_true: ground-truth labels
    :type y_true: list
    :param y_pred: predicted labels
    :type y_pred: list
    :param save_dir: directory where to save the plot
    :type save_dir: str
    :param file_name: name of the file
    :type file_name: str
    :param d_args: dictionary with label convention
    :param normalize: if set to True, normalise the confusion matrix. (% instead of absolute values)
    :type normalize: bool
    :param cmap: matplotlib cmap to be used for plot
    :type cmap: matplotlib colormap
    :param thr: threshold used for classification (for display purposes only)
    :type thr: float
    :return: bacc: balanced accuracy
    """
    cm = confusion_matrix(y_true, y_pred)

    # Single-class case
    try:
        bacc = balanced_accuracy_score(y_true, y_pred)
    except Exception:
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)
        bacc = float((y_true_arr == y_pred_arr).mean()) if y_true_arr.size else float('nan')
    print(f'Balanced Accuracy = {bacc:.3f}')

    if d_args['label_bonafide'] == 1:
        classes = ['$\it{Fake}$', '$\it{Real}$']
    else:
        classes = ['$\it{Real}$', '$\it{Fake}$']

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]

    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(f'Confusion matrix (BA={bacc:.3f})\nThreshold = {thr:.3f}')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    # Save the plot
    save_path = os.path.join(save_dir, f'confusion_matrix_{file_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f'Confusion matrix saved to {save_path}')
    plt.close()

    return bacc


def plot_results(pred_scores,
                 true_labels,
                 save_directory,
                 subfolder_name,
                 file_name,
                 d_args,
                 legend=None,
                 norm_conf_mat=True):
    """
    Plot results.
    :param pred_scores: list of predicted labels
    :param true_labels: list of true labels
    :param save_directory: directory to save the results
    :param subfolder_name: name of the subfolder (optional)
    :param file_name: name of the file to save the results
    :param d_args: dictionary with label convention
    :param legend: legend for the plot
    :param norm_conf_mat: normalize confusion matrix
    """

    print('PLOTTING RESULTS\n')

    # Folder management
    if subfolder_name:
        model_directory = os.path.join(save_directory, subfolder_name)
        ensure_folder_exists(model_directory)
    else:
        model_directory = save_directory

    # Ensure numpy arrays format
    pred_scores = np.asarray(pred_scores)

    # Compute ROC
    opt_thr, rocauc, eer  = plot_roc_curve(
        true_labels,
        pred_scores,
        model_directory,
        file_name,
        legend=legend
    ) # opt_thr: EER thr

    pred_labels_after_thr = pd.Series(
        np.where(
            pred_scores >= 0.5,
            d_args['label_spoof'],
            d_args['label_bonafide']
        )
    )

    # Confusion matrix
    bal_acc = plot_confusion_matrix(
        true_labels,
        pred_labels_after_thr,
        model_directory,
        file_name,
        d_args,
        norm_conf_mat,
        cmap=plt.cm.Blues,
        thr=0.5,
    )

    return {'roc': float(rocauc), 'eer': float(eer), 'bal acc': float(bal_acc)}
