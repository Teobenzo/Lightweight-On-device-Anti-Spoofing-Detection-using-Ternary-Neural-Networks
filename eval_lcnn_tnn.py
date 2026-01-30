import torch
import os
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from torch.utils.data import DataLoader
from src.models import LCNN_mel_spec_TNN, LCNN_sinc_conv_TNN
from src.nn_utils import LoadEvalData, visualizza_parametri_modello_ternary, print_eval_info_lcnn_tnn, set_robustness_params, load_model_name_lcnn_tnn, load_df_dataset
from src.utils import *
import copy
from tqdm import tqdm
import src.delta_regimes as delta_regimes
from glob import glob
import time
from src.utils import _get_robust_suffix
import matplotlib.pyplot as plt


def ternary_eval(model, df_eval, save_path, device, config):

    # EVAL DATALOADER
    file_eval = list(df_eval['path'])

    eval_set = LoadEvalData(list_IDs=file_eval, d_args=config)
    eval_loader = DataLoader(eval_set, batch_size=config['batch_size'], shuffle=False, drop_last=False, num_workers=config['num_workers'])

    model.eval()
    fname_list = []
    score_list = []

    # INITIALIZATION OF TIMING INFERENCE METRICS
    total_start = time.perf_counter()
    total_batches = 0
    total_samples = 0
    gpu_forward_ms = 0.0  # sum of forward time on GPU

    # FORWARD PASS
    with torch.no_grad():
        for batch_x, utt_id in tqdm(eval_loader, total=len(eval_loader)):

            batch_x = batch_x.to(device)

            total_batches += 1
            total_samples += batch_x.size(0)

            # Timing
            if device.type == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()

            batch_score = model(batch_x)  # Forward pass

            if device.type == 'cuda':
                end_event.record()
                torch.cuda.synchronize(device)
                gpu_forward_ms += start_event.elapsed_time(end_event)

            batch_score = batch_score.exp()  # From log-likelihood to probabilities

            # SCORES
            batch_score = batch_score[:, config['label_spoof']].detach().cpu().numpy().ravel()

            fname_list.extend(utt_id)
            score_list.extend(batch_score.tolist())

    # TIMING METRICS
    total_wall_s = time.perf_counter() - total_start
    throughput = (total_samples / total_wall_s) if total_wall_s > 0 else float('nan')
    avg_latency_ms = (total_wall_s * 1000.0 / total_samples) if total_samples > 0 else float('nan')
    avg_forward_ms_per_batch = (gpu_forward_ms / total_batches) if total_batches > 0 else float('nan')
    avg_forward_ms_per_sample = (gpu_forward_ms / total_samples) if total_samples > 0 else float('nan')


    # PRINTS AND LOGS
    with open(save_path, 'a+') as fh:
        for f, cm in zip(fname_list, score_list):
            fh.write('{} {}\n'.format(f, cm))
    fh.close()
    print('Scores saved to {}'.format(save_path))

    timing_path = os.path.join(
        config['model_subfolder'],
        'timing_{}__on-{}.txt'.format(config['model_name'], config['dataset_name'])
    )
    with open(timing_path, 'w') as tf:
        tf.write('Total files: {}\n'.format(total_samples))
        tf.write('Total batches: {}\n'.format(total_batches))
        tf.write('Total duration eval (s): {:.6f}\n'.format(total_wall_s))
        tf.write('Throughput (file/s): {:.3f}\n'.format(throughput))
        tf.write('Mean latency end-to-end (ms/file): {:.3f}\n'.format(avg_latency_ms))
        tf.write('Mean on-device batch forward (ms/batch): {:.3f}\n'.format(avg_forward_ms_per_batch))
        tf.write('Mean on-device file forward (ms/file): {:.3f}\n'.format(avg_forward_ms_per_sample))

    print('Total duration eval: {:.3f}s | file: {} | throughput: {:.2f} file/s'.format(
        total_wall_s, total_samples, throughput
    ))
    print('Mean latency end-to-end: {:.2f} ms/file'.format(avg_latency_ms))
    print('Mean on-device batch forward: {:.2f} ms/batch ({:.2f} ms/file)'.format(
        avg_forward_ms_per_batch, avg_forward_ms_per_sample
    ))


def init_eval(config):
    # SET GPU
    device = torch.device('cuda')

    # LOAD MODEL
    model_path = config['model_path']
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model file found: {model_path}")

    model_cfg = copy.deepcopy(config['model'])
    if config['feature'] == 'mel_spec':
        model = LCNN_mel_spec_TNN(device=device, d_args=model_cfg).eval().to(device)
    elif config['feature'] == 'sinc_conv':
        model = LCNN_sinc_conv_TNN(device=device, d_args=model_cfg).eval().to(device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        raise RuntimeError(f"Error during model loading: {e}")


    # PRINT MODEL PARAMETERS
    model_subfolder = config['model_subfolder']
    model_name = config['model_name']
    save_path = config['scores_file']
    df_eval = config['df_evaluation']

    param_path = os.path.join(model_subfolder, f"{model_name}_parameters.txt")
    visualizza_parametri_modello_ternary(model, file_path=param_path)


    # Delete previous scores file if exists
    if os.path.exists(save_path):
        print("Save path exists - Deleting file")
        os.remove(save_path)

    print(f"Loading model: {model_path}")
    print(f"Save path: {save_path}")
    print()

    # EVALUATION
    ternary_eval(
        model=model,
        df_eval=df_eval,
        save_path=save_path,
        device=device,
        config=config
    )

def run_test(config_yaml):
    # FOLDERS
    model_folder = config_yaml["model_folder"]
    ensure_folder_exists(model_folder)

    log_folder = config_yaml["log_folder"]
    ensure_folder_exists(log_folder)

    results_folder = config_yaml["results_folder"]
    ensure_folder_exists(results_folder)

    # DATASET (EVAL)
    df_evaluation = load_df_dataset(config_yaml)
    config_yaml['df_evaluation'] = df_evaluation

    # MODEL FILE
    model_file = load_model_name_lcnn_tnn(config_yaml)
    config_yaml['model_path'] = os.path.join(model_folder, model_file)

    model_name = os.path.splitext(os.path.basename(model_file))[0]

    # RESULTS SUBFOLDERS AND FILES
    base_subfolder = os.path.join(results_folder, model_name)
    ensure_folder_exists(base_subfolder)

    eval_dataset = config_yaml['dataset_name']
    robust_suffix = _get_robust_suffix(config_yaml)

    eval_folder_name = f"eval_{eval_dataset}" + (f"_{robust_suffix}" if robust_suffix else "")
    eval_subfolder = os.path.join(base_subfolder, eval_folder_name)
    ensure_folder_exists(eval_subfolder)

    results_file = f"{model_name}__on-{eval_dataset}.csv"
    save_path = os.path.join(eval_subfolder, results_file)

    config_yaml['model_subfolder'] = eval_subfolder
    config_yaml['scores_file'] = save_path
    config_yaml['model_name'] = model_name
    config_yaml['robust_suffix'] = robust_suffix

    # FOR ROBUSTNESS TESTING
    set_robustness_params(config_yaml)

    # PRINTS
    print_eval_info_lcnn_tnn(config_yaml)

    # RUN EVAL
    init_eval(config=config_yaml)

    # PLOT RESULTS
    df_results = pd.read_csv(save_path, sep=' ', header=None)
    metrics = plot_results(
        pred_scores=df_results[1],
        true_labels=df_evaluation['label'],
        save_directory=eval_subfolder,
        subfolder_name="plots",
        file_name=f"{model_name}__on-{eval_dataset}",
        d_args=config_yaml,
        legend="Test Results",
        norm_conf_mat=True
    )

    metrics_csv = os.path.join(base_subfolder, 'metrics.csv')  # results_folder/<model_name>/metrics.csv

    # Load existing metrics dataframe or create new dataframe
    if os.path.exists(metrics_csv):
        df_metrics = pd.read_csv(metrics_csv, index_col=0)
    else:
        df_metrics = pd.DataFrame(columns=['roc', 'eer', 'bal acc'])
        df_metrics.index.name = 'dataset'

    # Robustness suffix for dataset key
    robust_suffix = config_yaml.get('robust_suffix') or ''
    dataset_key = eval_dataset + (f"_{robust_suffix}" if robust_suffix else "")
    df_metrics.loc[dataset_key, ['roc', 'eer', 'bal acc']] = [
        metrics['roc'], metrics['eer'], metrics['bal acc']
    ]

    df_metrics.to_csv(metrics_csv)
    print(f"Metrics saved in: {metrics_csv}")


if __name__ == '__main__':

    # YAML
    config_path = 'config/cfg_lcnn_TNN.yaml'
    config_yaml = read_yaml(config_path)

    if config_yaml['model']['fs']==16000:
        config_yaml['model']['spect_fmax']=7600
    elif config_yaml['model']['fs']==8000:
        config_yaml['model']['spect_fmax']=4000

    # SEED
    seed_everything(1234)

    # GPU
    set_gpu(-1) # -1 = automatic selection

    # DATASES FOR TESTING
    # comment/uncomment to exclude/include datasets
    dict_datasets = [#'asvspoof2019',
                     'asvspoof2021_LA',
                     'asvspoof2021_DF',
                     'FakeOrReal',
                     'Purdue',
                     'InTheWild',
                     'TimitTts',
                     'llj+libri_real_only',
                     'llj_real_only',
                     'libri_real_only'
    ]

    # ROBUSTNESS TESTS
    # comment/uncomment to exclude/include
    config_yaml['no_robust_test'] = False
    dict_robust_tests = ['no_robust_test',
                         #'apply_mulaw',
                         #'apply_G722',
                         #'apply_RIR',
                         #'apply_noise_Inj_10db',
                         #'apply_noise_Inj_20db',
                         #'apply_opus',
                         #'apply_vorbis',
                         #'apply_env_noise_20db_SNR',
                         #'apply_env_noise_15db_SNR',
                         #'apply_white_noise_20db_SNR',
                         #'#apply_white_noise_10db_SNR',
    ]

    # LOOP OVER DATASETS AND ROBUSTNESS TESTS
    for key in dict_datasets:
        config_yaml[key] = True
        for other_key in (k for k in dict_datasets if k != key):
            config_yaml[other_key] = False

        if config_yaml['asvspoof2019']==True:
            for r_key in dict_robust_tests:
                config_yaml[r_key] = True
                print(f"\n\n\nEVALUATION ON DATASET {key} with robust test {r_key}\n\n\n")
                for other_r_key in (k for k in dict_robust_tests if k != r_key):
                    config_yaml[other_r_key] = False

                run_test(config_yaml)
        else:
            # reset robustness flag
            for r_key in dict_robust_tests:
                config_yaml[r_key] = False
            # no robustness test
            print(f"\n\n\nEVALUATION ON DATASET {key} with NO robust test\n\n\n")
            i=1
            run_test(config_yaml)

