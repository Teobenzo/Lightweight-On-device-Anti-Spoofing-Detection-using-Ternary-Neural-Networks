import torch
import os
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from torch.utils.data import DataLoader
from src.nn_utils import ternary_train_epoch, valid_model, LoadTrainData
from src.utils import *
from src.models import LCNN_mel_spec_TNN, LCNN_sinc_conv_TNN, init_weights
import src.delta_regimes as delta_regimes
import datetime
import matplotlib.pyplot as plt


def main(config):

    device = torch.device('cuda')

    # DATAFRAME
    df_train = pd.read_csv(train_file, sep=' ', header=None)
    df_dev = pd.read_csv(dev_file, sep=' ', header=None)

    df_train = df_train.replace({'bonafide': config['label_bonafide'], 'spoof': config['label_spoof']})
    df_dev = df_dev.replace({'bonafide': config['label_bonafide'], 'spoof': config['label_spoof']})

    df_train['path'] = df_train[1].apply(lambda x: os.path.join(data_dir, config['path_files_train'], str(x) + '.flac'))
    df_dev['path'] = df_dev[1].apply(lambda x: os.path.join(data_dir, config['path_files_dev'], str(x) + '.flac'))

    df_train['label'] = df_train[4]
    df_dev['label'] = df_dev[4]

    # TRAIN DATALOADER
    d_label_trn = dict(zip(df_train['path'], df_train['label']))
    file_train = list(df_train['path'])

    train_set = LoadTrainData(list_IDs=file_train, labels=d_label_trn, d_args=config)
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, drop_last=True,
                              num_workers=config['num_workers'])
    del train_set, d_label_trn

    # DEV DATALOADER
    d_label_dev = dict(zip(df_dev['path'], df_dev['label']))
    file_dev = list(df_dev['path'])

    dev_set = LoadTrainData(list_IDs=file_dev, labels=d_label_dev, d_args=config)
    dev_loader = DataLoader(dev_set, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
    del dev_set, d_label_dev

    # DELTA REGIMES
    DeltaRegimeClass = delta_regimes.by_name(config['model']['delta_regime_type'])
    delta_regime = DeltaRegimeClass(config['model']['delta_regime_min'], config['model']['delta_regime_max'], max_at_epoch=config['model']['delta_regime_max_epoch'])

    # LOAD MODEL
    if config['feature'] == 'mel_spec':
        model = LCNN_mel_spec_TNN(device=device, d_args=config['model'], delta=delta_regime.get(0))
    elif config['feature'] == 'sinc_conv':
        model = LCNN_sinc_conv_TNN(device=device, d_args=config['model'], delta=delta_regime.get(0))

    model = (model).to(device)
    init_weights(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['model']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', min_lr=config['min_lr'])
    cost = nn.NLLLoss()

    # INITIALIZATION
    writer = SummaryWriter(config['log_folder'])
    best_acc = 0
    best_loss = 100
    early_stopping = 0
    best_epoch = 0
    flag_earlystop = False
    best_zeros_perc = None
    print(f"Device: {device}")
    delta = float('-inf')

    sparsity_logs = {
        "epoch": [],
        "zeros_perc": [],
        "plus1_perc": [],
        "minus1_perc": [],
        "train_accuracy": [],
        "valid_accuracy": []
    }

    # CHECKPOINT LOADING
    if os.path.exists(config['model_path']):
        model.load_state_dict(torch.load(config['model_path'], map_location=device))
        print('Model loaded : {}'.format(config['model_path']))

        model.eval()
        with torch.no_grad():
            init_loss, init_acc = valid_model(
                dev_loader=dev_loader,
                model=model,
                device=device,
                criterion=cost
            )
        best_loss = init_loss
        best_acc = init_acc

        print(f"Resuming from saved model: initial valid_loss={init_loss:.4f}, valid_acc={init_acc:.2f}")
    else:
        print('No model found at {}'.format(config['model_path']))

    # TRAINING LOOP
    for epoch in range(config['num_epochs']):

        # SET DELTA
        prev_delta = delta
        delta = delta_regime.get(epoch) # delta change with epochs
        writer.add_scalar("DELTA", delta, epoch)
        model.set_delta(delta)

        print(f"Epoch {epoch}, learning rate: {optimizer.param_groups[0]['lr']}, delta: {delta:.5f}")

        if early_stopping < config['early_stopping']:
            # TRAINING
            running_loss, train_accuracy = ternary_train_epoch(train_loader=train_loader, model=model, optim=optimizer,
                                                       device=device, criterion=cost)

            weights_entropy = model.get_weights_entropy()

            # VALIDATION
            writer.add_scalar("WEIGHTS ENTROPY", weights_entropy, epoch)
            with torch.no_grad():
                valid_loss, valid_accuracy = valid_model(dev_loader=dev_loader, model=model, device=device,
                                                         criterion=cost)

            scheduler.step(valid_loss)

            # WRITER
            writer.add_scalar('valid_accuracy', valid_accuracy, epoch)
            writer.add_scalar('valid_loss', valid_loss, epoch)
            print('\nEpoch: {} - Train Loss: {:.5f} - Val Loss: {:.5f} - Train Acc: {:.2f} - Val Acc: {:.2f}'.format(
                epoch, running_loss, valid_loss, train_accuracy, valid_accuracy))

            # SAVE BEST MODEL
            if valid_loss < best_loss or delta > prev_delta:
                print('Best model found at epoch ', epoch)
                if valid_loss < best_loss:
                    print(f'Valid loss improved from {best_loss:.5f} to {valid_loss:.5f}')
                else:
                    print(f'Delta increased from {prev_delta:.5f} to {delta:.5f}')

                print(f'Current delta value: {delta:.5f}')
                config['model']['delta_best_model'] = delta
                best_epoch = epoch

                torch.save(model.state_dict(), config['model_path'])
                early_stopping = 0

                best_loss = min(valid_loss, best_loss)
                best_acc = valid_accuracy
            else:
                early_stopping += 1

        else:
            # EARLY STOPPING PRINTS
            flag_earlystop = True
            print("\n" + "=" * 50)
            print("Training stopped after {} epochs".format(epoch))
            print('Best model found at epoch {} - Best Val Acc {:.2f}'.format(best_epoch, best_acc))
            print(f"Delta best model: {config['model']['delta_best_model']:.5f}")

            if config['feature'] == 'mel_spec':
                best_model = LCNN_mel_spec_TNN(device=device, d_args=config['model'], delta=config['model']['delta_best_model'])
            elif config['feature'] == 'sinc_conv':
                best_model = LCNN_sinc_conv_TNN(device=device, d_args=config['model'], delta=config['model']['delta_best_model'])

            best_model = best_model.to(device)
            best_model.load_state_dict(torch.load(config['model_path'], map_location=device))

            # Weights distribution of the best model
            z, p1, m1, num = best_model.weight_count()
            z_perc, p1_perc, m1_perc = (z / num * 100, p1 / num * 100, m1 / num * 100)
            print(f"Best model weights distribution - Zeros: {z_perc:.2f}% - Plus one: {p1_perc:.2f}% - Minus one: {m1_perc:.2f}%")

            best_zeros_perc = z_perc

            print("=" * 50)
            break


        # SPARSITY PRINTS AND LOGS
        z, p1, m1, num = model.weight_count()
        z_perc, p1_perc, m1_perc = (z/num*100, p1/num*100, m1/num*100)
        print(f"Epoch {epoch} - Zeros: {z_perc:.2f}% - Plus one: {p1_perc:.2f}% - Minus one: {m1_perc:.2f}%")
        writer.add_scalar("weights/zeros", z_perc, epoch)
        writer.add_scalar("weights/plus-one", p1_perc, epoch)
        writer.add_scalar("weights/minus-one", m1_perc, epoch)
        writer.add_scalar("sparsity/zeros_perc", z_perc, epoch)

        sparsity_logs["epoch"].append(epoch)
        sparsity_logs["zeros_perc"].append(z_perc)
        sparsity_logs["plus1_perc"].append(p1_perc)
        sparsity_logs["minus1_perc"].append(m1_perc)
        sparsity_logs["train_accuracy"].append(train_accuracy)
        sparsity_logs["valid_accuracy"].append(valid_accuracy)

    # NON EARLY STOPPING PRINTS
    if flag_earlystop == False:
        print("\n" + "=" * 50)
        print(f"TRAINING COMPLETED")
        print(f"Training stopped after {config['num_epochs']} epochs")
        print('Best model found at epoch {} - Best Val Acc {:.2f}'.format(best_epoch, best_acc))
        print(f"Delta best model: {config['model']['delta_best_model']:.5f}")

        if config['feature'] == 'mel_spec':
            best_model = LCNN_mel_spec_TNN(device=device, d_args=config['model'], delta=config['model']['delta_best_model'])
        elif config['feature'] == 'sinc_conv':
            best_model = LCNN_sinc_conv_TNN(device=device, d_args=config['model'], delta=config['model']['delta_best_model'])

        best_model = best_model.to(device)
        best_model.load_state_dict(torch.load(config['model_path'], map_location=device))

        # Weights distribution of the best model
        z, p1, m1, num = best_model.weight_count()
        z_perc, p1_perc, m1_perc = (z / num * 100, p1 / num * 100, m1 / num * 100)
        print(f"Best model weights distribution - Zeros: {z_perc:.2f}% - Plus one: {p1_perc:.2f}% - Minus one: {m1_perc:.2f}%")
        best_zeros_perc = z_perc
        print("=" * 50)

    # PLOT SPARSITY
    target_dir = config.get('model_subfolder',
                            os.path.join(config.get("results_folder", "results"),
                            os.path.splitext(os.path.basename(config['model_path']))[0]))
    os.makedirs(target_dir, exist_ok=True)

    sparsity_df = pd.DataFrame(sparsity_logs)
    model_tag = os.path.splitext(os.path.basename(config['model_path']))[0]
    csv_path = os.path.join(target_dir, f"sparsity_{model_tag}.csv")
    png_path = os.path.join(target_dir, f"sparsity_{model_tag}.png")

    sparsity_df.to_csv(csv_path, index=False)

    plt.figure(figsize=(9, 5))
    plt.plot(sparsity_df["epoch"], sparsity_df["zeros_perc"], label="Zero (%)", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Percentage of zeros")
    plt.title("Sparsity rate over epochs")
    plt.grid(alpha=0.3)

    if best_zeros_perc is not None:
        plt.axvline(best_epoch, color="crimson", linestyle="--", linewidth=1.5, alpha=0.8,
                    label=f"Best model (epoch {best_epoch})")
        plt.scatter([best_epoch], [best_zeros_perc], color="crimson", s=40, zorder=5)
        plt.annotate(
            f"delta={config['model']['delta_best_model']:.5f}\nzeros={best_zeros_perc:.2f}%\nval_acc={best_acc:.2f}",
            xy=(best_epoch, best_zeros_perc),
            xytext=(10, 12),
            textcoords="offset points",
            fontsize=9,
            color="crimson",
            arrowprops=dict(arrowstyle="->", color="crimson", lw=1),
        )

    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    print(f"Sparsity salvata in: {csv_path}")
    print(f"Grafico salvato in: {png_path}")


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

    # FOLDERS
    model_folder = config_yaml["model_folder"]
    ensure_folder_exists(model_folder)

    log_folder = config_yaml["log_folder"]
    ensure_folder_exists(log_folder)

    # MODEL FILE NAME
    config_yaml['dataset_name'] = "ASVSPOOF19"
    today_str = datetime.datetime.now().strftime('%d %B')
    model_file_template = config_yaml["model_file_template"]
    if config_yaml['feature'] == 'mel_spec':
        model_file = model_file_template.format(
            feature=config_yaml["feature"],
            dataset_name=config_yaml['dataset_name'],
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
            date=today_str
        )
    elif config_yaml['feature'] == 'sinc_conv':
        model_file = model_file_template.format(
            feature=config_yaml["feature"],
            dataset_name=config_yaml['dataset_name'],
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
            sinc_out_channels=config_yaml['model']['sinc_out_channels'],
            sinc_kernel_size=config_yaml['model']['sinc_kernel_size'],
            sinc_stride=config_yaml['model']['sinc_stride'],
            date=today_str
        )
    config_yaml['model_path'] = os.path.join(model_folder, model_file)

    model_name = os.path.splitext(model_file)[0]
    config_yaml['model_subfolder'] = os.path.join(config_yaml.get('results_folder', 'results'), model_name)
    ensure_folder_exists(config_yaml['model_subfolder'])

    # results_folder = config_yaml.get("results_folder", "results")

    # DATASET PATH (TRAIN E DEV)
    data_dir = config_yaml['data_dir']
    train_file = os.path.join(data_dir, config_yaml['path_df_train'])
    dev_file = os.path.join(data_dir, config_yaml['path_df_dev'])

    # PRINTS
    print("\n" + "="*50)
    print("STARTING TRAINING\n")
    print("="*50)
    print(f"DATE OF TODAY: {today_str}")
    print(f"MODEL FILE TEMPLATE: {config_yaml['model_file_template']}")
    print(f"DATASET: {config_yaml['dataset_name']}")
    print(f"LEARNING RATE: {config_yaml['lr']}")
    print(f"MIN LEARNING RATE: {config_yaml['min_lr']}")
    print(f"WIN LENGTH: {config_yaml['model']['win_len']} SEC")
    print("\nDELTA REGIME SETTINGS:")
    print(f"- Type: {config_yaml['model']['delta_regime_type']}")
    print(f"- Min value: {config_yaml['model']['delta_regime_min']}")
    print(f"- Max value: {config_yaml['model']['delta_regime_max']}")
    print(f"- Max epoch: {config_yaml['model']['delta_regime_max_epoch']}")
    print("\nTERNARIZATION SETTINGS:")
    print(f"- Full precision activations: {config_yaml['model']['f32_activations']}")
    print(f"- Full precision FC layers: {config_yaml['model']['full_fc']}")
    print("\nOPTIMIZER SETTINGS:")
    print(f"- Weight decay: {config_yaml['model']['weight_decay']}")
    print("="*50 + "\n")

    # TRAINING
    main(config_yaml)
