import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Optional, Tuple, Dict, List

SUPPORTED_SPARSE_FMTS = ('coo', 'csr')

'''
    Info torch.sparse: https://docs.pytorch.org/docs/stable/sparse.html
'''


# ---------- helper: check sparse format support ----------

def assert_sparse_format_supported(fmt: str, device: torch.device) -> None:
    """
    Assert sparse@dense operation is supported for the requested format.
    - coo: usa torch.sparse.mm
    - csr: usa torch.matmul
    """
    fmt = fmt.lower()
    if fmt not in ("coo", "csr"):
        raise ValueError(f"Format '{fmt}' not valid. Expected: 'coo' o 'csr'.")

    try:
        if fmt == "coo":
            # A(2x2) coo @ X(2x3) -> torch.sparse.mm
            indices = torch.tensor([[0, 1], [1, 0]], device=device)
            values = torch.ones(2, device=device)
            A = torch.sparse_coo_tensor(indices, values, (2, 2), device=device).coalesce()
            X = torch.randn(2, 3, device=device)
            _ = torch.sparse.mm(A, X)
        else:  # csr
            # A(2x2) csr @ X(2x3) -> torch.matmul
            crow_indices = torch.tensor([0, 1, 2], device=device)  # 2 righe
            col_indices = torch.tensor([0, 1], device=device)  # 2 colonne non zero
            values = torch.ones(2, device=device)
            A = torch.sparse_csr_tensor(crow_indices, col_indices, values, size=(2, 2), device=device)
            X = torch.randn(2, 3, device=device)
            _ = torch.matmul(A, X)
    except Exception as e:
        raise RuntimeError(
            f"Sparse format '{fmt}' not supported on '{device.type}'. "
            f"Error: {e}"
        )


# ---------- helper: sparseify weight matrices ----------

def conv_weight_to_sparse_2d(W: torch.Tensor):
    # Convert conv weight (out_c, in_c, kH, kW) to sparse 2D matrix (out_c, in_c*kH*kW)
    out_c, in_c, kH, kW = W.shape
    W2 = W.view(out_c, -1).contiguous()  # (out_c, in_c*kH*kW)
    W_sp = W2.to_sparse().coalesce()  # sparse COO 2D
    return W_sp


def linear_weight_to_sparse_2d(W: torch.Tensor):
    # Convert linear weight (out_f, in_f) to sparse 2D matrix (out_f, in_f)
    W_sp = W.to_sparse().coalesce()  # sparse COO 2D
    return W_sp


# ---------- wrappers ----------

# CPU-only functions
class SparseConv2d(nn.Module):
    def __init__(self, orig_conv: nn.Conv2d, weight_sparse: torch.sparse_coo_tensor, bias: torch.Tensor = None):
        super().__init__()
        assert orig_conv.groups == 1
        self.register_buffer("weight_sparse", weight_sparse.coalesce())
        self.in_channels = orig_conv.in_channels
        self.out_channels = orig_conv.out_channels
        self.kernel_size = orig_conv.kernel_size
        self.stride = orig_conv.stride
        self.padding = orig_conv.padding
        self.dilation = orig_conv.dilation
        self.groups = orig_conv.groups
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spW = self.weight_sparse  # (out_c, in_c*kH*kW)
        kH, kW = self.kernel_size
        B, C_in, H, W = x.shape
        x_unf = F.unfold(
            x,
            kernel_size=(kH, kW),
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride
        )  # da (B, C, H, W) a (B, K, L)
        B, K, L = x_unf.shape


        X_cat = x_unf.permute(1, 0, 2).contiguous().view(K, B * L)
        # sparse matrix multiplication (convolution substituted by sp matmul)
        Y_cat = torch.sparse.mm(spW, X_cat)  # (out_c, B*L)
        Y = Y_cat.view(self.out_channels, B, L).permute(1, 0, 2).contiguous()
        # out dimensions
        H_out = (H + 2 * self.padding[0] - self.dilation[0] * (kH - 1) - 1) // self.stride[0] + 1
        W_out = (W + 2 * self.padding[1] - self.dilation[1] * (kW - 1) - 1) // self.stride[1] + 1
        # from (B, out_c, L) to (B, out_c, H_out, W_out)
        out = F.fold(Y.view(B, self.out_channels, L), (H_out, W_out), kernel_size=(1, 1))

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)
        return out


class SparseLinear(nn.Module):
    def __init__(self, orig_linear: nn.Linear, weight_sparse: torch.sparse_coo_tensor, bias: torch.Tensor = None):
        super().__init__()
        self.register_buffer("weight_sparse", weight_sparse.coalesce())  # (out, in)
        self.out_features = orig_linear.out_features
        self.in_features = orig_linear.in_features
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spW = self.weight_sparse
        # corresponds to y = xW^T + b
        Y_t = torch.sparse.mm(spW, x.t().contiguous())  # (out, B)
        Y = Y_t.t()  # (B, out)
        if self.bias is not None:
            Y = Y + self.bias.view(1, -1)
        return Y


# ---------- model traversal & replacement ----------
def convert_model_to_sparse(model: nn.Module,
                            min_sparsity: float = 0.8,
                            convert_1x1_and_linear_only: bool = True,
                            force_convert_3x3: bool = False) -> nn.Module:
    """
    Convert conv1x1 and Linear layers (by default) to sparse wrappers when their sparsity >= min_sparsity.
    convert_1x1_and_linear_only: if True, convert only 1x1 convs and linears (default).
    if False, convert all convs (1x1, 3x3, 5x5 etc...).

    if convert_1x1_and_linear_only is True and force_convert_3x3 is False, converts only 1x1 convs and linears.
    if convert_1x1_and_linear_only is True and force_convert_3x3 is True, converts 1x1 and 3x3 convs and linears.
    """

    def recurse(mod: nn.Module, prefix=''):
        for name, child in list(mod.named_children()):
            full_name = f"{prefix}.{name}" if prefix else name
            # Conv2d
            if isinstance(child, nn.Conv2d):
                if child.groups != 1:
                    # skip grouped conv
                    # keep module as is
                    pass
                else:
                    kH, kW = child.kernel_size
                    convert_this = False
                    if convert_1x1_and_linear_only:
                        if (kH, kW) == (1, 1):
                            convert_this = True
                        elif force_convert_3x3 and (kH, kW) == (3, 3):
                            convert_this = True
                    else:
                        # convert all convs
                        convert_this = True

                    # Conversion
                    if convert_this:
                        # Convert only if sparsity  >= 80%
                        # Otherwise overhead of sparse.mm is too high (better keep dense conv)
                        W = child.weight.detach().cpu()
                        out_c, in_c, kh, kw = W.shape
                        N = out_c * in_c * kh * kw
                        zeros = (W == 0.0).sum().item()
                        sparsity = zeros / N
                        if sparsity >= min_sparsity:
                            sp = conv_weight_to_sparse_2d(W)  # weights sparse
                            bias = child.bias.detach().cpu() if child.bias is not None else None
                            wrapper = SparseConv2d(child, sp, bias=bias)
                            # copy other attributes if needed
                            setattr(mod, name, wrapper)  # substitute module with wrapper (sparse module)
                        else:
                            # not sparse enough; leave as is
                            pass

            # Linear
            elif isinstance(child, nn.Linear):
                # Convert only if sparsity  >= 80%
                # Otherwise overhead of sparse.mm is too high (better keep dense conv)
                W = child.weight.detach().cpu()
                out_f, in_f = W.shape
                N = out_f * in_f
                zeros = (W == 0.0).sum().item()
                sparsity = zeros / N
                if sparsity >= min_sparsity:
                    sp = linear_weight_to_sparse_2d(W)  # weights sparse
                    bias = child.bias.detach().cpu() if child.bias is not None else None
                    wrapper = SparseLinear(child, sp, bias=bias)  # substitute module with wrapper (sparse module)
                    setattr(mod, name, wrapper)
                else:
                    pass
            else:
                # recurse into child
                recurse(child, full_name)

    recurse(model)
    return model


# ---------- save sparse model ----------

def save_sparsified_model(model: nn.Module,
                          config: dict,
                          overwrite: bool = False,
                          save_meta: bool = True) -> Optional[Tuple[str, Optional[str]]]:
    """
    Save the sparse model with sparsity tags.
    Doesn't overwrite original model file.
    Returns (saved_model_path, meta_file_path) or None if sparse is disabled.
    """

    def _to_bool_str(val, default=False) -> str:
        # convert in true/false string
        if val is None:
            return str(bool(default)).lower()
        if isinstance(val, bool):
            return str(val).lower()
        if isinstance(val, str):
            return str(val.strip().lower() in ('1', 'true', 'yes', 'y', 'on')).lower()
        return str(bool(val)).lower()

    if not config.get('enable_sparse', False):
        return None

    # Sparsity tag
    sparsity_tag = (
        f"SPARSE_min{config.get('sparse_min_sparsity', 0.8)}_"
        f"1x1only{_to_bool_str(config.get('sparse_convert_1x1_and_linear_only', True), True)}_"
        f"force3x3{_to_bool_str(config.get('sparse_force_convert_3x3', False), False)}"
    )

    orig_model_path = config['model_path']
    orig_dir = os.path.dirname(orig_model_path)
    orig_name = os.path.splitext(os.path.basename(orig_model_path))[0]

    base_new_name = f"{orig_name}_{sparsity_tag}.pth"
    new_path = os.path.join(orig_dir, base_new_name)

    # Avoid overwrite
    if not overwrite and os.path.exists(new_path):
        v = 1
        while True:
            candidate = os.path.join(orig_dir, f"{orig_name}_{sparsity_tag}_v{v}.pth")
            if not os.path.exists(candidate):
                new_path = candidate
                break
            v += 1

    # Save
    torch.save(model.state_dict(), new_path)

    meta_path = None
    if save_meta:
        # Compute sparsity stats
        total_elems = 0
        zero_elems = 0
        for m in model.modules():
            cls = m.__class__.__name__
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                W = m.weight.detach().cpu()
                total_elems += W.numel()
                zero_elems += (W == 0).sum().item()
            elif cls in ('SparseConv2d', 'SparseLinear'):
                spW = m.weight_sparse  # sparse COO 2D
                shape = spW.size()
                layer_total = 1
                for d in shape:
                    layer_total *= d
                nnz = spW._values().numel()
                total_elems += layer_total
                zero_elems += (layer_total - nnz)

        global_sparsity = (zero_elems / total_elems) if total_elems > 0 else 0.0

        meta_path = new_path.replace('.pth', '_meta.txt')
        with open(meta_path, 'w') as mf:
            mf.write(f"original_model: {orig_model_path}\n")
            mf.write(f"sparsified_model: {new_path}\n")
            mf.write(f"sparse_min_sparsity: {config.get('sparse_min_sparsity')}\n")
            mf.write(f"sparse_convert_1x1_and_linear_only: {config.get('sparse_convert_1x1_and_linear_only')}\n")
            mf.write(f"sparse_force_convert_3x3: {config.get('sparse_force_convert_3x3')}\n")
            mf.write(f"global_sparsity: {global_sparsity:.6f}\n")

    print(f"[Sparse] Modello salvato: {new_path}")
    if meta_path:
        print(f"[Sparse] Metadati: {meta_path}")
    return new_path, meta_path


def summarize_sparsification(model: nn.Module,
                             config: dict,
                             save_report: bool = True) -> List[Dict]:
    """
    Return list of dict with info about conv/linear layers:
      name, type, status, kernel, shape, sparsity, reason
    status: 'sparse', 'kept_dense_low_sparsity', 'kept_dense_not_candidate'
    """
    min_sparsity = config.get('sparse_min_sparsity', 0.8)
    only_1x1 = config.get('sparse_convert_1x1_and_linear_only', True)
    force_3x3 = config.get('sparse_force_convert_3x3', False)

    rows = []
    for name, m in model.named_modules():
        cls = m.__class__.__name__

        # Sparse wrappers
        if cls == 'SparseConv2d':
            spW = m.weight_sparse
            total = spW.size()[0] * spW.size()[1]
            nnz = spW._values().numel()
            sparsity = (total - nnz) / total if total > 0 else 0.0
            rows.append(dict(
                name=name,
                type='Conv2d',
                kernel=str(m.kernel_size),
                shape=f"{spW.size()} (flattened conv)",
                sparsity=sparsity,
                status='sparse',
                reason='converted'
            ))

        elif cls == 'SparseLinear':
            spW = m.weight_sparse
            total = spW.size()[0] * spW.size()[1]
            nnz = spW._values().numel()
            sparsity = (total - nnz) / total if total > 0 else 0.0
            rows.append(dict(
                name=name,
                type='Linear',
                kernel='-',
                shape=str(spW.size()),
                sparsity=sparsity,
                status='sparse',
                reason='converted'
            ))

        # Dense Conv2d
        elif isinstance(m, nn.Conv2d):
            if m.groups != 1:
                candidate = False
                reason = 'groups!=1'
            else:
                kH, kW = m.kernel_size
                if only_1x1:
                    if (kH, kW) == (1, 1):
                        candidate = True
                    elif force_3x3 and (kH, kW) == (3, 3):
                        candidate = True
                    else:
                        candidate = False
                else:
                    candidate = True
                reason = 'rule_excluded' if not candidate else ''
            W = m.weight.detach().cpu()
            zeros = (W == 0).sum().item()
            total = W.numel()
            sparsity = zeros / total if total > 0 else 0.0

            if candidate:
                if sparsity >= min_sparsity:
                    status = 'WARNING_expected_sparse'
                    reason = f"s>=thr({min_sparsity})"
                else:
                    status = 'kept_dense_low_sparsity'
                    reason = f"s({sparsity:.3f})<thr({min_sparsity})"
            else:
                status = 'kept_dense_not_candidate'

            rows.append(dict(
                name=name,
                type='Conv2d',
                kernel=str(m.kernel_size),
                shape=str(tuple(m.weight.shape)),
                sparsity=sparsity,
                status=status,
                reason=reason
            ))

        # Dense Linear
        elif isinstance(m, nn.Linear):
            W = m.weight.detach().cpu()
            zeros = (W == 0).sum().item()
            total = W.numel()
            sparsity = zeros / total if total > 0 else 0.0
            # All linears are candidates
            if sparsity >= min_sparsity:
                status = 'WARNING_expected_sparse'
                reason = f"s>=thr({min_sparsity})"
            else:
                status = 'kept_dense_low_sparsity'
                reason = f"s({sparsity:.3f})<thr({min_sparsity})"

            rows.append(dict(
                name=name,
                type='Linear',
                kernel='-',
                shape=str(tuple(m.weight.shape)),
                sparsity=sparsity,
                status=status,
                reason=reason
            ))

    # Prints
    def fmt_pct(x):
        return f"{x * 100:6.2f}%"

    header = f"{'NAME':40} {'TYPE':8} {'KERNEL':10} {'SPARSITY':10} {'STATUS':24} REASON"
    print("\n[Sparse] Riepilogo dettagliato layer")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['name'][:40]:40} {r['type']:8} {r['kernel']:10} {fmt_pct(r['sparsity']):10} {r['status'][:24]:24} {r['reason']}")

    conv_sparse = sum(1 for r in rows if r['type'] == 'Conv2d' and r['status'] == 'sparse')
    lin_sparse = sum(1 for r in rows if r['type'] == 'Linear' and r['status'] == 'sparse')
    print(f"\n[Sparse] Totale conv sparse: {conv_sparse} | linear sparse: {lin_sparse}")

    if save_report:
        report_path = os.path.join(config['model_subfolder'], "sparsification_report.csv")
        try:
            import csv
            with open(report_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['name', 'type', 'kernel', 'shape', 'sparsity', 'status', 'reason'])
                for r in rows:
                    writer.writerow([r['name'], r['type'], r['kernel'], r['shape'],
                                     f"{r['sparsity']:.6f}", r['status'], r['reason']])
            print(f"[Sparse] Report salvato: {report_path}")
        except Exception as e:
            print(f"[Sparse] Errore salvataggio report: {e}")

    return rows


