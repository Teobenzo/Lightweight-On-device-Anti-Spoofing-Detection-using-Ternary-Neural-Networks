from glob import glob
import os
import pandas as pd
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from .audio_utils import *
from .utils import *
from torchaudio.utils import download_asset
import librosa
import torchaudio
import numpy
from . import audio_utils as AU


# LOAD DATASET

def load_for_data(base_dir, subset, label_bonafide, label_spoof):
    # FakeOrReal dataset
    real_files = glob(os.path.join(base_dir, subset, 'real', '*.wav'))
    fake_files = glob(os.path.join(base_dir, subset, 'fake', '*.wav'))

    df_real = pd.DataFrame(real_files, columns=['path'])
    df_real['label'] = label_bonafide

    df_fake = pd.DataFrame(fake_files, columns=['path'])
    df_fake['label'] = label_spoof

    return pd.concat([df_real, df_fake])

def load_df_dataset(config_yaml):
    if config_yaml['asvspoof2019']:
        config_yaml['dataset_name'] = 'ASVSPOOF19'
        data_dir = config_yaml['data_dir']
        eval_file = os.path.join(data_dir, config_yaml['path_df_eval'])

        df_evaluation = pd.read_csv(eval_file, sep=' ', header=None)
        df_evaluation = df_evaluation.replace({'bonafide': config_yaml['label_bonafide'],
                                               'spoof': config_yaml['label_spoof']})
        df_evaluation['path'] = df_evaluation[1].apply(
            lambda x: os.path.join(data_dir, config_yaml['path_files_eval'], str(x) + '.flac'))
        df_evaluation['algo'] = df_evaluation[3]
        df_evaluation['label'] = df_evaluation[4]

    elif config_yaml['asvspoof2021_LA']:
        config_yaml['dataset_name'] = 'ASVSPOOF21_LA'
        data_dir = config_yaml['data_dir_asvspoof21_la']
        eval_file = os.path.join(data_dir, config_yaml['path_df_eval_asvspoof21_la'])

        df_evaluation = pd.read_csv(eval_file, sep=' ', header=None)
        df_evaluation = df_evaluation.replace(
            {'bonafide': config_yaml['label_bonafide'], 'spoof': config_yaml['label_spoof']})
        df_evaluation['path'] = df_evaluation[1].apply(
            lambda x: os.path.join(data_dir, config_yaml['path_files_eval_asvspoof21_la'], str(x) + '.flac'))
        df_evaluation['label'] = df_evaluation[5]

    elif config_yaml['asvspoof2021_DF']:
        config_yaml['dataset_name'] = 'ASVSPOOF21_DF'
        data_dir = config_yaml['data_dir_asvspoof21_df']
        eval_file = os.path.join(data_dir, config_yaml['path_df_eval_asvspoof21_df'])

        df_evaluation = pd.read_csv(eval_file, sep=' ', header=None)

        exclude_ids = {'DF_E_2101080'}
        df_evaluation = df_evaluation[~df_evaluation[1].isin(exclude_ids)]

        df_evaluation = df_evaluation.replace(
            {'bonafide': config_yaml['label_bonafide'], 'spoof': config_yaml['label_spoof']})
        df_evaluation['path'] = df_evaluation[1].apply(
            lambda x: os.path.join(data_dir, config_yaml['path_files_eval_asvspoof21_df'], str(x) + '.flac'))
        df_evaluation['label'] = df_evaluation[5]

    elif config_yaml['FakeOrReal']:
        config_yaml['dataset_name'] = 'FakeOrReal'
        base_dir_for = '/nas/home/dsalvi/fake_or_real_FOR/for-original-wav'
        df_evaluation = load_for_data(base_dir_for, 'testing', label_bonafide=config_yaml['label_bonafide'],
                                      label_spoof=config_yaml['label_spoof'])

    elif config_yaml['Purdue']:
        config_yaml['dataset_name'] = 'Purdue'
        # FAKE
        fake_list_libri = glob(
            '/nas/home/dsalvi/background_detection/SAFE_challenge_data/DiffSSD_dataset/generated_speech/*/speaker_*')
        fake_list_lj = glob(
            '/nas/home/dsalvi/background_detection/SAFE_challenge_data/DiffSSD_dataset/generated_speech/*/*.wav')
        fake_samples = fake_list_lj
        for folder in fake_list_libri:
            fake_samples.extend(glob(folder + '/*.mp3'))
        for folder in fake_list_libri:
            fake_samples.extend(glob(folder + '/*.wav'))

        df_fake = pd.DataFrame(fake_samples, columns=['path'])
        df_fake['label'] = config_yaml['label_spoof']

        # REAL
        speaker_list = [f.split('/')[-1] for f in fake_list_libri]
        speaker_list = list(set(speaker_list))
        lj_real = glob('/nas/public/dataset/lj_speech/*')
        libri_real = []
        for speaker in speaker_list:
            speaker_num = speaker[8:]
            libri_sp_list = glob(f'/nas/public/dataset/LibriSpeech/train-clean-360/{speaker_num}/*/*.flac')
            libri_real.extend(libri_sp_list)

        real_lj_libri = lj_real + libri_real

        df_real = pd.DataFrame(real_lj_libri, columns=['path'])
        df_real['label'] = config_yaml['label_bonafide']

        df_evaluation = pd.concat([df_fake, df_real], ignore_index=True)

    elif config_yaml['InTheWild']:
        config_yaml['dataset_name'] = 'InTheWild'
        df_itw = pd.read_csv('/nas/home/dsalvi/AISEC_audio_deepfake/meta.csv')
        df_itw['path'] = df_itw['file'].apply(
            lambda x: '/nas/home/dsalvi/AISEC_audio_deepfake/wavs/' + x)
        df_itw['label'] = df_itw['label'].map({'spoof': config_yaml['label_spoof'], 'bona-fide': config_yaml[
            'label_bonafide']})
        df_evaluation = df_itw[['path', 'label']]

    return df_evaluation

def set_robustness_params(config_yaml):
    if config_yaml['apply_noise_Inj_10db'] or config_yaml['apply_noise_Inj_20db']:
        SAMPLE_NOISE = download_asset("tutorial-assets/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo-8000hz.wav")
        noise, fs_noise = torchaudio.load(SAMPLE_NOISE)
        AU.noise = noise
        AU.fs_noise = fs_noise

    if config_yaml['apply_RIR']:
        rir_directory = '/nas/public/exchange/dataset_smartphones/POLIPHONE/IRs_devices'
        rir_files_raw = [os.path.join(rir_directory, f) for f in os.listdir(rir_directory) if f.endswith('.npy')]
        rir_files = []
        for rir_file in rir_files_raw:
            rir_file = numpy.load(rir_file)
            rir_file = librosa.resample(rir_file, orig_sr=44100, target_sr=config_yaml['model']['fs'])
            rir_files.append(rir_file)
        AU.rir_files = rir_files

    if config_yaml['apply_env_noise_20db_SNR'] or config_yaml['apply_env_noise_15db_SNR']:
        df_wham_tt = pd.read_csv('/nas/home/mbenzo/tesi/dataset/wham_noise/metadata/noise_meta_tt.csv')
        AU.wham_files = df_wham_tt['utterance_id']

class LoadTrainData(Dataset):
    '''
        Return:
        - x: audio sample
        - y: label
    '''
    def __init__(self, list_IDs, labels, d_args):
        '''self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)'''

        self.list_IDs = list_IDs
        self.labels = labels
        self.win_len = d_args['model']['win_len']
        self.fs = d_args['model']['fs']
        self.d_args = d_args
        self.win_len_samples = int(self.win_len * self.fs)

        self.balance_batch = True

        df = pd.DataFrame(labels.items(), columns=['path', 'label'])
        self.real_list = list(df[df['label'] == d_args['label_bonafide']]['path'])
        self.fake_list = list(df[df['label'] == d_args['label_spoof']]['path'])

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):

        # BALANCIING REAL AND FAKE
        if self.balance_batch:
            if index % 2 == 0:
                track_name = random.choice(self.real_list)
            else:
                track_name = random.choice(self.fake_list)
        else:
            track_name = self.list_IDs[index]

        # READING AUDIO
        x, fs = read_audio(track_name, fs=self.fs, norm=True,  # trim=False,
                           trim=self.d_args['trim'],
                           mulaw=False,
                           g722=False,
                           rir=False,
                           noise_Inj_10db=False,
                           noise_Inj_20db=False,
                           )
        y = self.labels[track_name] # label
        audio_len = len(x)
        x = torch.from_numpy(x).float()

        # PADDING
        if audio_len < self.win_len_samples:
            x = pad(x, self.win_len_samples)
            audio_len = len(x)

        # RANDOM WINDOW
        last_valid_start_sample = audio_len - self.win_len_samples
        if not last_valid_start_sample == 0:
            start_sample = random.randrange(start=0, stop=last_valid_start_sample)
        else:
            start_sample = 0
        x_win = x[start_sample : start_sample + self.win_len_samples]

        # For pycharm debugging
        if not isinstance(x_win, torch.Tensor):
            x_win = torch.from_numpy(x_win)
        x_win = x_win.unsqueeze(0)  # shape [1, win_len_samples]

        x_win = torch.Tensor(x_win)

        return x_win, y

class LoadEvalData(Dataset):
    def __init__(self, list_IDs, d_args):
        self.list_IDs = list_IDs
        self.win_len = d_args['model']['win_len']
        self.fs = d_args['model']['fs']
        self.d_args = d_args
        self.win_len_samples = int(self.win_len * self.fs)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):

        # AUDIO READING
        track = self.list_IDs[index]
        x, fs = read_audio(track, fs=self.fs, norm=True, #trim=False,
                            trim=self.d_args['trim'],
                            mulaw=self.d_args['apply_mulaw'],
                            g722=self.d_args['apply_G722'],
                            rir=self.d_args['apply_RIR'],
                            noise_Inj_10db=self.d_args['apply_noise_Inj_10db'],
                            noise_Inj_20db=self.d_args['apply_noise_Inj_20db'],
                            opus=self.d_args['apply_opus'],
                            vorbis=self.d_args['apply_vorbis'],
                            envNoise_20db=self.d_args['apply_env_noise_20db_SNR'],
                            envNoise_15db=self.d_args['apply_env_noise_15db_SNR'],
                            whiteNoise_inj_20db=self.d_args['apply_white_noise_20db_SNR'],
                            whiteNoise_inj_10db=self.d_args['apply_white_noise_10db_SNR'],
                           )
        audio_len = len(x)
        x = torch.from_numpy(x).float()

        # PADDING
        if audio_len < self.win_len_samples:
            x = pad(x, self.win_len_samples)
            audio_len = len(x)

        # TEST ON THE MIDDLE WINDOW
        start_sample = int(0.5*(len(x) - self.win_len_samples))
        x_win = x[start_sample: start_sample + self.win_len_samples]

        # For pycharm debugging
        if not isinstance(x_win, torch.Tensor):
            x_win = torch.from_numpy(x_win)
        x_win = x_win.unsqueeze(0)  # shape [1, win_len_samples]

        x_win = torch.Tensor(x_win)

        return x_win, track


# TRAINING AND VALIDATION FUNCTIONS

def train_epoch(train_loader, model, optim, device, criterion):

    # INITIALIZATION
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0

    model.train()


    # TRAINING
    for batch_x, batch_y in tqdm(train_loader, total=len(train_loader)):
        # batch_x: samples
        # batch_y: labels

        batch_size = batch_x.size(0)
        num_total += batch_size

        # BATCH TO GPU
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)

        # FORWARD PASS
        batch_out = model(x = batch_x)
        batch_loss = criterion(batch_out, batch_y)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        running_loss += (batch_loss.item() * batch_size)

        # BACKWARD PASS + OPTIMIZATION
        optim.zero_grad()
        batch_loss.backward()
        optim.step()

    # STATS TRAINING
    running_loss /= num_total
    train_accuracy = (num_correct / num_total) * 100
    return running_loss, train_accuracy

def valid_model(dev_loader, model, device, criterion):

    # INITIALIZATION
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0

    model.eval()

    for batch_x, batch_y in tqdm(dev_loader, total=len(dev_loader)):
        batch_size = batch_x.size(0)
        num_total += batch_size

        # BATCH TO GPU
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)

        # FORWARD PASS
        batch_out = model(batch_x)

        # STATS
        batch_loss = criterion(batch_out, batch_y)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        running_loss += (batch_loss.item() * batch_size)

    running_loss /= num_total
    valid_acc = 100 * (num_correct / num_total)
    return running_loss, valid_acc

def ternary_train_epoch(train_loader, model, optim, device, criterion):

    # INITIALIZATION
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0

    model.train()

    # TRAINING
    for batch_x, batch_y in tqdm(train_loader, total=len(train_loader)):
        # batch_x: samples
        # batch_y: labels

        batch_size = batch_x.size(0)
        num_total += batch_size

        # BATCH TO GPU
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)

        # FORWARD PASS
        batch_out = model(x = batch_x)
        batch_loss = criterion(batch_out, batch_y)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        running_loss += (batch_loss.item() * batch_size)

        # BACKWARD PASS + OPTIMIZATION
        # forward pass (quantization) ->
        # ->  loss -> compute gradient on ternarized weights ->
        # -> de-quantization
        # -> weights update ->
        # -> da capo

        optim.zero_grad()
        batch_loss.backward()

        # STE
        for p in list(model.parameters()):
            if hasattr(p, 'org'):
                p.data.copy_(p.org)

        optim.step()

        # clamping
        for p in list(model.parameters()):
            if hasattr(p, 'org'):
                p.org.copy_(p.data.clamp_(-1, 1))

    # STATS TRAINING
    running_loss /= num_total
    train_accuracy = (num_correct / num_total) * 100
    return running_loss, train_accuracy


# LOAD MODEL NAME

''' Questa non è necessaria se mettiamo solo modello ternario'''
def load_model_name_lcnn(config_yaml):
    if config_yaml['feature'] in ('mel_spec', 'mfcc'):
        model_file = config_yaml["model_file_template"].format(
            feature=config_yaml["feature"],
            dataset_name='ASVSPOOF19',
            win_len=config_yaml['model']["win_len"],
            fs=config_yaml['model']["fs"],
            trim=config_yaml["trim"],
            lr=config_yaml['lr'],
            min_lr=config_yaml['min_lr'],
            weight_decay=config_yaml["model"]["weight_decay"],
            bias=config_yaml["model"]["bias"],
            date=config_yaml["model_date_4eval"],
        )
    elif config_yaml['feature'] == 'sinc_conv':
        model_file = config_yaml["model_file_template_sinc_conv"].format(
            feature=config_yaml["feature"],
            dataset_name='ASVSPOOF19',
            win_len=config_yaml['model']["win_len"],
            fs=config_yaml['model']["fs"],
            trim=config_yaml["trim"],
            lr=config_yaml['lr'],
            min_lr=config_yaml['min_lr'],
            weight_decay=config_yaml["model"]["weight_decay"],
            bias=config_yaml["model"]["bias"],
            sinc_out_channels=config_yaml['model']['sinc_out_channels'],
            sinc_kernel_size=config_yaml['model']['sinc_kernel_size'],
            sinc_stride=config_yaml['model']['sinc_stride'],
            date=config_yaml["model_date_4eval"]
        )
    else:
        raise ValueError("feature non supportata")
    return model_file

def load_model_name_resnet(config_yaml):
    model_file_template = config_yaml["model_file_template"]
    if config_yaml['feature'] == 'mel_spec':
        model_file = model_file_template.format(
            feature=config_yaml["feature"],
            dataset_name='ASVSPOOF19',
            win_len=config_yaml['model']["win_len"],
            fs=config_yaml['model']["fs"],
            trim=config_yaml["trim"],
            lr=config_yaml['lr'],
            min_lr=config_yaml['min_lr'],
            weight_decay=config_yaml['model']['weight_decay'],
            date=config_yaml["model_date_4eval"]
        )
    else:
        raise ValueError("feature non supportata")
    return model_file

def load_model_name_lcnn_tnn(config_yaml):
    if config_yaml['feature'] == 'mel_spec':
        model_file_template = config_yaml["model_file_template"]
        model_file = model_file_template.format(
            feature=config_yaml['feature'],
            dataset_name='ASVSPOOF19',
            win_len=config_yaml['model']["win_len"],
            fs=config_yaml['model']['fs'],
            trim=config_yaml["trim"],
            lr=config_yaml['lr'],
            min_lr=config_yaml['min_lr'],
            delta_min=config_yaml['model']['delta_regime_min'],
            delta_max=config_yaml['model']['delta_regime_max'],
            delta_epoch=config_yaml['model']['delta_regime_max_epoch'],
            delta_regime=config_yaml['model']['delta_regime_type'],
            f32_act=config_yaml['model']['f32_activations'],
            full_fc=config_yaml['model']['full_fc'],
            weight_decay=config_yaml['model']['weight_decay'],
            date=config_yaml["model_date_4eval"]
        )
    elif config_yaml['feature'] == 'sinc_conv':
        model_file_template = config_yaml["model_file_template_sinc_conv"]
        model_file = model_file_template.format(
            feature=config_yaml['feature'],
            dataset_name='ASVSPOOF19',
            win_len=config_yaml['model']["win_len"],
            fs=config_yaml['model']['fs'],
            trim=config_yaml["trim"],
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
            date=config_yaml["model_date_4eval"]
        )

    return model_file

def load_model_name_resnet_tnn(config_yaml):
    model_file_template = config_yaml["model_file_template"]
    if config_yaml['feature'] == 'mel_spec':
        model_file = model_file_template.format(
            feature=config_yaml["feature"],
            dataset_name='ASVSPOOF19',
            win_len=config_yaml['model']["win_len"],
            fs=config_yaml['model']["fs"],
            trim=config_yaml["trim"],
            lr=config_yaml['lr'],
            min_lr=config_yaml['min_lr'],
            delta_min=config_yaml['model']['delta_regime_min'],
            delta_max=config_yaml['model']['delta_regime_max'],
            delta_epoch=config_yaml['model']['delta_regime_max_epoch'],
            delta_regime=config_yaml['model']['delta_regime_type'],
            f32_act=config_yaml['model']['f32_activations'],
            full_fc=config_yaml['model']['full_fc'],
            weight_decay=config_yaml['model']['weight_decay'],
            date=config_yaml["model_date_4eval"]
        )
    else:
        raise ValueError("feature non supportata")
    return model_file

# PRINTING INFO FOR EVALUATION

def visualizza_parametri_modello_ternary(model, file_path=None):
    output_lines = []

    output_lines.append("\n" + "=" * 60)
    output_lines.append("MODEL PARAMETERS")
    output_lines.append("=" * 60)

    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        param_size = param.numel()
        total_params += param_size
        if param.requires_grad:
            trainable_params += param_size

        output_lines.append(f"Layer: {name}")
        output_lines.append(f"  Shape: {param.shape}")
        output_lines.append(f"  Parameters: {param_size:,}")
        output_lines.append(f"  Device: {param.device}")
        output_lines.append(f"  Type: {param.dtype}")

        if param.numel() > 0:
            try:
                output_lines.append(f"  Min: {param.min().item():.4f}, Max: {param.max().item():.4f}")
                output_lines.append(f"  Mean: {param.mean().item():.4f}, Std: {param.std().item():.4f}")
            except:
                pass
        output_lines.append("-" * 40)

    output_lines.append("\nSummary:")
    output_lines.append(f"Total parameters: {total_params:,}")
    output_lines.append(f"Trainable parameters: {trainable_params:,}")

    if hasattr(model, 'weight_count'):
        zeros, plus_ones, minus_ones, num_params = model.weight_count()
        z_perc = zeros / num_params * 100
        p1_perc = plus_ones / num_params * 100
        m1_perc = minus_ones / num_params * 100
        non_zero = plus_ones + minus_ones
        non_zero_perc = non_zero / num_params * 100

        output_lines.append("\n" + "=" * 60)
        output_lines.append("TERNARIZED WEIGHTS DISTRIBUTION")
        output_lines.append("=" * 60)
        output_lines.append(f"Total parameters: {num_params:,}")
        output_lines.append(f"Zeroed parameters: {zeros:,} ({z_perc:.2f}%)")
        output_lines.append(f"Non-zero parameters: {non_zero:,} ({non_zero_perc:.2f}%)")
        output_lines.append(f"  - Value +1: {plus_ones:,} ({p1_perc:.2f}%)")
        output_lines.append(f"  - Value -1: {minus_ones:,} ({m1_perc:.2f}%)")

    output_lines.append("=" * 60)

    if file_path:
        with open(file_path, 'w') as f:
            for line in output_lines:
                f.write(line + "\n")

def print_eval_info_full_precision(config_yaml):
    print("\n" + "=" * 50)
    print("STARTING TESTING\n")
    print("=" * 50)
    print(f"DATE OF MODEL'S TRAINING : {config_yaml['model_date_4eval']}")
    print(f"MODEL FILE TEMPLATE: {config_yaml['model_file_template']}")
    print(f"DATASET: {config_yaml['dataset_name']}")
    print(f"LEARNING RATE: {config_yaml['lr']}")
    print(f"MIN LEARNING RATE: {config_yaml['min_lr']}")
    print(f"WIN LENGTH: {config_yaml['model']['win_len']} SEC")
    print("\nOPTIMIZER SETTINGS:")
    print(f"- Weight decay: {config_yaml['model']['weight_decay']}")
    print("=" * 50 + "\n")

def print_eval_info_tnn(config_yaml):
    print("\n" + "=" * 50)
    print("STARTING TESTING\n")
    print("=" * 50)
    print(f"DATE OF MODEL'S TRAINING : {config_yaml['model_date_4eval']}")
    if config_yaml['feature'] == 'mel_spec':
        print(f"MODEL FILE TEMPLATE: {config_yaml['model_file_template']}")
    elif config_yaml['feature'] == 'sinc_conv':
        print(f"MODEL FILE TEMPLATE: {config_yaml['model_file_template_sinc_conv']}")
    print(f"DATASET: {config_yaml['dataset_name']}")
    print(f"LEARNING RATE: {config_yaml['lr']}")
    print(f"MIN LR: {config_yaml['min_lr']}")
    print(f"WIN LENGTH: {config_yaml['model']['win_len']} SEC")
    print("\nDELTA REGIME SETTINGS:")
    print(f"- Regime: {config_yaml['model']['delta_regime_type']}")
    print(f"- Delta min: {config_yaml['model']['delta_regime_min']}")
    print(
        f"- Delta max: {config_yaml['model']['delta_regime_max']} (epoch target: {config_yaml['model']['delta_regime_max_epoch']})")
    print("\nTERNARIZATION SETTINGS:")
    print(f"- Full precision activations: {config_yaml['model']['f32_activations']}")
    print(f"- Full precision FC layers: {config_yaml['model']['full_fc']}")
    print("\nOPTIMIZER SETTINGS:")
    print(f"- Weight decay: {config_yaml['model']['weight_decay']}")
    print("=" * 50 + "\n")

def print_eval_info_rawTFnet_small(config_yaml):
    print("\n" + "=" * 50)
    print("STARTING TESTING\n")
    print("=" * 50)
    print(f"DATE OF MODEL'S TRAINING : {config_yaml['model_date_4eval']}")
    print(f"MODEL FILE TEMPLATE: {config_yaml['model_file_template']}")
    print(f"DATASET: {config_yaml['dataset_name']}")
    print(f"LEARNING RATE: {config_yaml['lr']}")
    print(f"MIN LEARNING RATE: {config_yaml['min_lr']}")
    print(f"WIN LENGTH: {config_yaml['model']['win_len']} SEC")
    print("\nOPTIMIZER SETTINGS:")
    print(f"- Weight decay: {config_yaml['model']['weight_decay']}")
    print("=" * 50 + "\n")