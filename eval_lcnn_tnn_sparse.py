import torch
import os
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from torch.utils.data import DataLoader
from src.models import LCNN_mel_spec_TNN, LCNN_sinc_conv_TNN
from src.nn_utils import LoadEvalData, visualizza_parametri_modello_ternary, load_for_data
from src.utils import *
import copy
from tqdm import tqdm
import src.delta_regimes as delta_regimes
from glob import glob
import time
from src.sparse_utils import convert_model_to_sparse, save_sparsified_model, summarize_sparsification


def ternary_eval(model, df_eval, save_path, device, config):

    # EVAL DATALOADER
    file_eval = list(df_eval['path'])

    eval_set = LoadEvalData(list_IDs=file_eval, d_args=config)
    eval_loader = DataLoader(eval_set, batch_size=config['batch_size'], shuffle=False, drop_last=False, num_workers=config['num_workers'])

    model.eval()
    fname_list = []
    score_list = []

    # EVALUATION EFFETTIVA (FORWARD PASS)
    with torch.no_grad():
        for batch_x, utt_id in tqdm(eval_loader, total=len(eval_loader)):

            batch_x = batch_x.to(device)

            batch_score = model(batch_x) # Forward pass

            batch_score = batch_score.exp()  # From log-likelihoods to probabilities
            batch_score = batch_score[:, config['label_spoof']].detach().cpu().numpy().ravel()

            # ID AND SCORES LIST
            fname_list.extend(utt_id)
            score_list.extend(batch_score.tolist())


    # WRITE SCORES TO FILE
    with open(save_path, 'a+') as fh:
        for f, cm in zip(fname_list, score_list):
            fh.write('{} {}\n'.format(f, cm))
    fh.close()
    print('Scores saved to {}'.format(save_path))


def init_eval(config):

    # LOAD CONFIGURATION
    model_path = config['model_path']
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Il file del modello non esiste: {model_path}")

    # SET DEVICE
    device = torch.device('cpu') # Test sparse works on cpu-only

    # LOAD MODEL
    model_cfg = copy.deepcopy(config['model'])
    if config['feature'] == 'mel_spec':
        model = LCNN_mel_spec_TNN(device=device, d_args=model_cfg).eval().to(device)
    elif config['feature'] == 'sinc_conv':
        model = LCNN_sinc_conv_TNN(device=device, d_args=model_cfg).eval().to(device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        raise RuntimeError(f"Errore during model loading: {e}")

    # ---  SPARSE CONVERSION ---
    if config['enable_sparse'] == True:
        print("[Sparse] Start conversion:",
              config.get('sparse_min_sparsity', 0.8))
        model = convert_model_to_sparse(
            model,
            min_sparsity=config['sparse_min_sparsity'],
            convert_1x1_and_linear_only=config['sparse_convert_1x1_and_linear_only'],
            force_convert_3x3=config['sparse_force_convert_3x3'],
        )
        print("[Sparse] Conversione completata")

    # PRINT REPORT AND SAVE SPARSE MODEL
    if config['enable_sparse'] == True:
        summarize_sparsification(model, config, save_report=True)
        _, _ = save_sparsified_model(model, config)

    # PRINT MODEL PARAMETERS
    model_subfolder = config['model_subfolder']
    model_name = config['model_name']
    save_path = config['scores_file']
    df_eval = config['df_evaluation']

    param_path = os.path.join(model_subfolder, f"{model_name}_parameters.txt")
    visualizza_parametri_modello_ternary(model, file_path=param_path)


    # If save_path exists, delete it
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
    set_gpu(-1)

    # FOLDERS
    model_folder = config_yaml["model_folder"]
    ensure_folder_exists(model_folder)

    log_folder = config_yaml["log_folder"]
    ensure_folder_exists(log_folder)

    # DATASETS
    if config_yaml['asvspoof2019']:
        config_yaml['dataset_name'] = 'ASVSPOOF19'
        data_dir = config_yaml['data_dir']
        eval_file = os.path.join(data_dir, config_yaml['path_df_eval'])

        df_evaluation = pd.read_csv(eval_file, sep=' ', header=None)
        df_evaluation = df_evaluation.replace({'bonafide': config_yaml['label_bonafide'], 'spoof': config_yaml['label_spoof']})
        df_evaluation['path'] = df_evaluation[1].apply(lambda x: os.path.join(data_dir, config_yaml['path_files_eval'], str(x) + '.flac'))
        df_evaluation['algo'] = df_evaluation[3]
        df_evaluation['label'] = df_evaluation[4]

    elif config_yaml['asvspoof2021_LA']:
        config_yaml['dataset_name'] = 'ASVSPOOF21_LA'
        data_dir = config_yaml['data_dir_asvspoof21_la']
        eval_file = os.path.join(data_dir, config_yaml['path_df_eval_asvspoof21_la'])

        df_evaluation = pd.read_csv(eval_file, sep=' ', header=None)
        df_evaluation = df_evaluation.replace({'bonafide': config_yaml['label_bonafide'], 'spoof': config_yaml['label_spoof']})
        df_evaluation['path'] = df_evaluation[1].apply(lambda x: os.path.join(data_dir, config_yaml['path_files_eval_asvspoof21_la'], str(x) + '.flac'))
        df_evaluation['label'] = df_evaluation[5]

    elif config_yaml['asvspoof2021_DF']:
        config_yaml['dataset_name'] = 'ASVSPOOF21_DF'
        data_dir = config_yaml['data_dir_asvspoof21_df']
        eval_file = os.path.join(data_dir, config_yaml['path_df_eval_asvspoof21_df'])

        df_evaluation = pd.read_csv(eval_file, sep=' ', header=None)

        exclude_ids = {'DF_E_2101080'}
        df_evaluation = df_evaluation[~df_evaluation[1].isin(exclude_ids)]

        df_evaluation = df_evaluation.replace({'bonafide': config_yaml['label_bonafide'], 'spoof': config_yaml['label_spoof']})
        df_evaluation['path'] = df_evaluation[1].apply(lambda x: os.path.join(data_dir, config_yaml['path_files_eval_asvspoof21_df'], str(x) + '.flac'))
        df_evaluation['label'] = df_evaluation[5]

    elif config_yaml['FakeOrReal']:
        config_yaml['dataset_name'] = 'FakeOrReal'
        base_dir_for = '/nas/home/dsalvi/fake_or_real_FOR/for-original-wav'
        df_evaluation = load_for_data(base_dir_for, 'testing', label_bonafide=config_yaml['label_bonafide'], label_spoof=config_yaml['label_spoof'])

    elif config_yaml['InTheWild']:
        config_yaml['dataset_name'] = 'InTheWild'
        df_itw = pd.read_csv('/nas/home/dsalvi/AISEC_audio_deepfake/meta.csv')
        df_itw['path'] = df_itw['file'].apply(lambda x: '/nas/home/dsalvi/AISEC_audio_deepfake/wavs/' + x)
        df_itw['label'] = df_itw['label'].map({'spoof': config_yaml['label_spoof'], 'bona-fide': config_yaml['label_bonafide']})
        df_evaluation = df_itw[['path', 'label']]

    else:
        raise ValueError("Error loading dataset.")

    config_yaml['df_evaluation'] = df_evaluation


    # MODEL FILE
    model_date = config_yaml["model_date_4eval"]
    if config_yaml['feature'] == 'mel_spec':
        model_file_template = config_yaml["model_file_template"]
        model_file = model_file_template.format(
            feature = config_yaml['feature'],
            dataset_name='ASVSPOOF19',
            win_len=config_yaml['model']["win_len"],
            fs=config_yaml['model']["fs"],
            trim=config_yaml['trim'],
            lr=config_yaml['lr'],
            min_lr=config_yaml['min_lr'],
            delta_min=config_yaml['model']['delta_regime_min'],
            delta_max=config_yaml['model']['delta_regime_max'],
            delta_epoch=config_yaml['model']['delta_regime_max_epoch'],
            delta_regime=config_yaml['model']['delta_regime_type'],
            f32_act=config_yaml['model']['f32_activations'],
            full_fc=config_yaml['model']['full_fc'],
            weight_decay=config_yaml['model']['weight_decay'],
            date=model_date
        )
    elif config_yaml['feature'] == 'sinc_conv':
        model_file_template = config_yaml["model_file_template_sinc_conv"]
        model_file = model_file_template.format(
            feature=config_yaml['feature'],
            dataset_name='ASVSPOOF19',
            win_len=config_yaml['model']["win_len"],
            fs=config_yaml['model']["fs"],
            lr=config_yaml['lr'],
            min_lr=config_yaml['min_lr'],
            delta_min=config_yaml['model']['delta_regime_min'],
            delta_max=config_yaml['model']['delta_regime_max'],
            delta_epoch=config_yaml['model']['delta_regime_max_epoch'],
            delta_regime=config_yaml['model']['delta_regime_type'],
            f32_act=config_yaml['model']['f32_activations'],
            full_fc=config_yaml['model']['full_fc'],
            weight_decay=config_yaml['model']['weight_decay'],
            sinc_out_channels=config_yaml['model']['sinc_out_channels'],
            sinc_kernel_size=config_yaml['model']['sinc_kernel_size'],
            sinc_stride=config_yaml['model']['sinc_stride'],
            date=model_date
        )
    config_yaml['model_path'] = os.path.join(model_folder, model_file)

    # RESULTS FOLDER
    results_folder_base = config_yaml["results_folder"]
    if config_yaml.get("enable_sparse", False):
        sparsity_tag = (
            f"sparse_min{config_yaml['sparse_min_sparsity']}_"
            f"1x1only{int(config_yaml['sparse_convert_1x1_and_linear_only'])}_"
            f"force3x3{int(config_yaml['sparse_force_convert_3x3'])}"
            f"fmt{config_yaml.get('sparse_format', 'csr')}"
        )
        results_folder = os.path.join(results_folder_base, sparsity_tag)
    else:
        sparsity_tag = "tnn_dense"
        results_folder = os.path.join(results_folder_base, sparsity_tag)
    ensure_folder_exists(results_folder)

    model_name = os.path.splitext(os.path.basename(model_file))[0]

    base_subfolder = os.path.join(results_folder, model_name)
    ensure_folder_exists(base_subfolder)

    eval_dataset = config_yaml['dataset_name']
    eval_subfolder = os.path.join(base_subfolder, f"eval_{eval_dataset}")
    ensure_folder_exists(eval_subfolder)

    results_file = f"{model_name}__on-{eval_dataset}.csv"
    save_path = os.path.join(eval_subfolder, results_file)

    config_yaml['model_subfolder'] = eval_subfolder
    config_yaml['scores_file'] = save_path
    config_yaml['model_name'] = model_name

    # PRINT CONFIGURATION
    print("\n" + "="*50)
    print("STARTING TESTING\n")
    print("="*50)
    print(f"DATE OF MODEL'S TRAINING : {config_yaml['model_date_4eval']}")
    print(f"MODEL FILE TEMPLATE: {config_yaml['model_file_template']}")
    print(f"DATASET: {config_yaml['dataset_name']}")
    print(f"LEARNING RATE: {config_yaml['lr']}")
    print(f"MIN LR: {config_yaml['min_lr']}")
    print(f"WIN LENGTH: {config_yaml['model']['win_len']} SEC")
    print("\nDELTA REGIME SETTINGS:")
    print(f"- Regime: {config_yaml['model']['delta_regime_type']}")
    print(f"- Delta min: {config_yaml['model']['delta_regime_min']}")
    print(f"- Delta max: {config_yaml['model']['delta_regime_max']} (epoch target: {config_yaml['model']['delta_regime_max_epoch']})")
    print("\nTERNARIZATION SETTINGS:")
    print(f"- Full precision activations: {config_yaml['model']['f32_activations']}")
    print(f"- Full precision FC layers: {config_yaml['model']['full_fc']}")
    print("\nOPTIMIZER SETTINGS:")
    print(f"- Weight decay: {config_yaml['model']['weight_decay']}")
    print("\nSPARSITY SETTINGS:")
    print(f"- Enable sparse: {config_yaml.get('enable_sparse', False)}")
    print(f"- Sparse format: {config_yaml.get('sparse_format', 'csr')}")
    print(f"- Sparse min sparsity: {config_yaml.get('sparse_min_sparsity', 'N/A')}")
    print(f"- Sparse convert 1x1 and linear only: {config_yaml.get('sparse_convert_1x1_and_linear_only', 'N/A')}")
    print(f"- Sparse force convert 3x3: {config_yaml.get('sparse_force_convert_3x3', 'N/A')}")
    print("="*50 + "\n")

    # EVALUATION
    init_eval(config=config_yaml)

    # PLOT RESULTS
    df_results = pd.read_csv(save_path, sep=' ', header=None)
    plot_results(
        pred_scores = df_results[1],
        true_labels = df_evaluation['label'],
        save_directory = eval_subfolder,
        subfolder_name="plots",
        file_name=f"{model_name}__on-{eval_dataset}",
        d_args = config_yaml,
        legend = "Test Results",
        norm_conf_mat = True
    )
