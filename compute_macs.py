import torch
import os
import torch.nn as nn
from collections import OrderedDict

from src.models import *
from src.utils import *


'''
Compute MACs (Multiply-Accumulate Operations) 

MACs are computed by loading a TNN model and performing a forward pass with a dummy input.

Actually, MAC refers to multiplications: here the term is used improperly since if the hardware were to be
optimized for ternary networks, operations with weight 1 or -1 would not require multiplications,
but would become either pass-through (weight +1) or flip-sign (weight −1)
'''

def macs_conv2d(layer, out):
    # MAC = Cout * (Cin / groups) * kH * kW * Hout * Wout
    Cin = layer.in_channels
    Cout = layer.out_channels
    kH, kW = layer.kernel_size
    groups = layer.groups
    Hout, Wout = out.shape[2], out.shape[3]

    mac = Cout * (Cin // groups) * kH * kW * Hout * Wout
    return mac

def macs_conv2d_ternary(layer, out):
    # MAC = number of non-zero weights * Hout * Wout
    # Since ach non-zero weight is multiplied Hout*Wout times during the convolution

    weight = layer.weight.data
    total_w = weight.numel()

    Hout, Wout = out.shape[2], out.shape[3]

    pos1_w = (weight == 1).sum().item()
    neg1_w = (weight == -1).sum().item()
    nnz_w = pos1_w + neg1_w
    zeros_w = total_w - nnz_w

    return {
        "type": "ternary_conv2d",
        "add_ops": nnz_w * Hout * Wout,
        "mult_ops": 0,
        "pass_through": pos1_w * Hout * Wout,
        "sign_flip": neg1_w * Hout * Wout,
        "nonzero_params": nnz_w,
        "total_params": total_w,
        "sparsity": zeros_w / total_w
    }

def macs_linear_ternary(layer):
    # MACs = number of non-zero weights

    weight = layer.weight.data
    total_w = weight.numel()

    pos1_w = (weight == 1).sum().item()
    neg1_w = (weight == -1).sum().item()
    nnz_w = pos1_w + neg1_w
    zeros_w = total_w - nnz_w

    return {
        "type": "ternary_linear",
        "add_ops": nnz_w,   # mac
        "mult_ops": 0,
        "pass_through": pos1_w,
        "sign_flip": neg1_w,
        "nonzero_params": nnz_w,
        "total_params": total_w,
        "sparsity": zeros_w / total_w
    }

def macs_conv1d(layer, out):
    # MAC = Cout * (Cin / groups) * k * Lout
    Cin = layer.in_channels
    Cout = layer.out_channels
    k = layer.kernel_size[0]
    groups = layer.groups
    Lout = out.shape[2]

    mac = Cout * (Cin // groups) * k * Lout
    return mac

def macs_sincconv(module, out):
    # MACs as in a conv1d: Cout * (Cin / groups) * K * Lout (groups=1 in SincConv)
    Cout = module.out_channels
    Cin = module.in_channels
    k = module.kernel_size
    Lout = out.shape[2]

    mac = Cout * Cin * k * Lout
    return mac

def macs_linear(layer):
    # MACs = in_features * out_features
    mac = layer.in_features * layer.out_features
    return mac

def profile_macs(model, waveform):
    # Compute MACs

    macs = OrderedDict()
    handles = []

    # HOOKS FUNCTIONS
    def hook_conv2d_ternary(m, inp, out):
        macs[m] = macs_conv2d_ternary(m, out)

    def hook_linear_ternary(m, inp, out):
        macs[m] = macs_linear_ternary(m)

    def hook_conv2d(m, inp, out):
        macs[m] = {"type": "conv2d", "mac": macs_conv2d(m, out)}

    def hook_conv1d(m, inp, out):
        macs[m] = {"type": "conv1d", "mac": macs_conv1d(m, out)}

    def hook_linear(m, inp, out):
        macs[m] = {"type": "linear", "mac": macs_linear(m)}

    def hook_sinc(m, inp, out):
        macs[m] = {"type": "sincconv", "mac": macs_sincconv(m, out)}

    for m in model.modules():
        if isinstance(m, TernarizeConv2d):
            handles.append(m.register_forward_hook(hook_conv2d_ternary))
        elif isinstance(m, TernarizeLinear):
            handles.append(m.register_forward_hook(hook_linear_ternary))
        elif isinstance(m, torch.nn.Conv2d):
            handles.append(m.register_forward_hook(hook_conv2d))
        elif isinstance(m, torch.nn.Conv1d):
            handles.append(m.register_forward_hook(hook_conv1d))
        elif isinstance(m, torch.nn.Linear):
            handles.append(m.register_forward_hook(hook_linear))
        elif m.__class__.__name__ == "SincConv":
            handles.append(m.register_forward_hook(hook_sinc))

    # Forward pass
    model.eval()
    with torch.no_grad():
        _ = model(waveform)

    # Remove hook
    for h in handles:
        h.remove()

    # Total sum
    total_standard_mac = sum(v["mac"] for v in macs.values() if "mac" in v)
    return macs, total_standard_mac


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # YAML
    config_path = 'config/cfg_lcnn_TNN.yaml'
    config_yaml = read_yaml(config_path)
    config_yaml['is_tnn'] = True

    if config_yaml['model']['fs']==16000:
        config_yaml['model']['spect_fmax']=7600
    elif config_yaml['model']['fs']==8000:
        config_yaml['model']['spect_fmax']=4000

    # FOLDERS
    model_folder = config_yaml["model_folder"]
    ensure_folder_exists(model_folder)
    results_folder = config_yaml["results_folder"]
    ensure_folder_exists(results_folder)

    # MODEL FILE PATH
    config_yaml['dataset_name'] = 'ASVSPOOF19'
    if config_path == 'config/cfg_lcnn_TNN.yaml':
        if config_yaml['feature'] == 'sinc_conv':
            model_file_template = config_yaml["model_file_template_sinc_conv"]
            model_file = model_file_template.format(
                feature=config_yaml['feature'],
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
                weight_decay=config_yaml["model"]["weight_decay"],
                sinc_out_channels=config_yaml['model']['sinc_out_channels'],
                sinc_kernel_size=config_yaml['model']['sinc_kernel_size'],
                sinc_stride=config_yaml['model']['sinc_stride'],
                date=config_yaml["model_date_4eval"]
            )
        elif config_yaml['feature'] == 'mel_spec':
            model_file_template = config_yaml["model_file_template"]
            model_file = model_file_template.format(
                feature=config_yaml['feature'],
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
                weight_decay=config_yaml["model"]["weight_decay"],
                date=config_yaml["model_date_4eval"]
            )
    else:
        raise ValueError("feature non supportata")

    config_yaml['model_path'] = os.path.join(model_folder, model_file)

    # LOAD MODEL
    if config_path == 'config/cfg_lcnn_TNN.yaml':
        if config_yaml['feature'] == 'sinc_conv':
            config_yaml['model']['mode'] = 'eval'
            model = LCNN_sinc_conv_TNN(device=device, d_args=config_yaml['model']).to(device)
        elif config_yaml['feature'] == 'mel_spec':
            config_yaml['model']['mode'] = 'train'
            model = LCNN_mel_spec_TNN(device=device, d_args=config_yaml['model']).to(device)

    try:
        state = torch.load(config_yaml['model_path'], map_location=device)
        model.load_state_dict(state)
    except Exception as e:
        raise RuntimeError(f"Error during model loading: {e}")

    # RESULTS FOLDER
    model_name = os.path.splitext(os.path.basename(model_file))[0]
    base_subfolder = os.path.join(results_folder, model_name)
    ensure_folder_exists(base_subfolder)

    # DUMMY INPUT
    fs = config_yaml['model']['fs']
    win_len = config_yaml['model']['win_len']
    dummy = torch.randn(1, 1, fs * win_len, device=device)

    # COMPUTE MACs
    macs_dict, total_std = profile_macs(model, dummy) # total_std = sum non-ternary MACs

    # COUNT TERNARY OPERATIONS
    if config_yaml.get('is_tnn', True):
        total_add = sum(v["add_ops"] for v in macs_dict.values() if v["type"].startswith("ternary"))
        total_pass = sum(v["pass_through"] for v in macs_dict.values() if v["type"].startswith("ternary"))
        total_flip = sum(v["sign_flip"] for v in macs_dict.values() if v["type"].startswith("ternary"))
        total_nnz_params = sum(v["nonzero_params"] for v in macs_dict.values() if v["type"].startswith("ternary"))
        total_ternary_params = sum(v["total_params"] for v in macs_dict.values() if v["type"].startswith("ternary"))
        tot_z, tot_p1, tot_m1, tot_param = model.weight_count()
        global_sparsity = tot_z / tot_param if tot_param > 0 else 0.0 # Sparsity includes bn and pooling params

    # PRINT AND SAVE REPORT
    print("Operations (standard layer: MAC, ternary layer: add/pass/flip):")
    for layer, info in macs_dict.items():
        name = layer.__class__.__name__
        t = info["type"]
        if t.startswith("ternary"):
            print(
                f"{name:<18} add={info['add_ops'] / 1e6:7.2f}M pass={info['pass_through'] / 1e6:7.2f}M flip={info['sign_flip'] / 1e6:7.2f}M spars={info['sparsity']:.2%}")
        else:
            print(f"{name:<18} MAC={info['mac'] / 1e6:7.2f}M")

    if config_yaml.get('is_tnn', True):
        if total_ternary_params > 0:
            print(
                f"\nTernary Operation Total: add={total_add / 1e6:.2f}M pass={total_pass / 1e6:.2f}M flip={total_flip / 1e6:.2f}M")
            print(f"Non zero ternary param: {total_nnz_params / 1e6:.3f}M (global sparsity: {global_sparsity:.2%})")
    else:
        print("\nNo ternary layer found.")

    print(f"\nTotal MACs (only ternary layers): {total_std / 1e6:.2f} M  (~{total_std / 1e9:.3f} G)")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params / 1e6:.2f} M")

    report_path = os.path.join(base_subfolder, "macs_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Report MACs model: {model_name}\n")
        f.write(f"Path model: {config_yaml['model_path']}\n")
        f.write(f"Feature: {config_yaml['feature']}\n")
        f.write(f"Win len (s): {win_len}\n")
        f.write(f"Sampling rate: {fs}\n")
        f.write("-" * 70 + "\n")
        f.write("Layer:\n")
        for layer, info in macs_dict.items():
            name = layer.__class__.__name__
            t = info["type"]
            if t.startswith("ternary"):
                f.write(f"{name:<18} add={info['add_ops'] / 1e6:7.2f}M pass={info['pass_through'] / 1e6:7.2f}M flip={info['sign_flip'] / 1e6:7.2f}M spars={info['sparsity']:.2%}\n")
            else:
                f.write(f"{name:<18} MAC={info['mac'] / 1e6:7.2f}M\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total MACs (standard): {total_std / 1e6:.3f} M  (~{total_std / 1e9:.3f} G)\n")
        if config_yaml.get('is_tnn', True):
            if total_ternary_params > 0:
                f.write(f"Ternary Operation Total: add={total_add / 1e6:.3f}M pass={total_pass / 1e6:.3f}M flip={total_flip / 1e6:.3f}M\n")
                f.write(f"Non zero ternary param: {total_nnz_params / 1e6:.3f}M su {total_ternary_params / 1e6:.3f}M (global sparsity: {global_sparsity:.2%})\n")
            else:
                f.write("No ternary layer found.\n")
        f.write(f"Total parameters: {n_params / 1e6:.3f} M\n")
    print(f"MACs file saved: {report_path}")