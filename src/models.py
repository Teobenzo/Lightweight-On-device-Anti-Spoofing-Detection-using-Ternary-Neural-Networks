import torch
import torch.nn as nn
import torchaudio.transforms as T
import torchaudio.functional as F
import torch.nn.functional as Func
import torchaudio
import numpy as np
import random

def init_weights(model):
    print("initializing weights...")
    for m in model.modules():
        if isinstance(m, (TernarizeConv2d, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight)  # 0.993
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.001)
        elif isinstance(m, (TernarizeLinear, nn.Linear)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.001)

class PreEmphasis(torch.nn.Module):
    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        self.register_buffer(
            'flipped_filter', torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, input: torch.tensor) -> torch.tensor:
        assert len(input.size()) == 2, 'The number of dimensions of input tensor must be 2!'
        input = input.unsqueeze(1)
        input = Func.pad(input, (1, 0), 'reflect')
        return Func.conv1d(input, self.flipped_filter).squeeze(1)  # filtering: y[n] = x[n] - coef * x[n-1]

class SincConv(nn.Module):
    """_summary_
    This code is from https://github.com/clovaai/aasist/blob/a04c9863f63d44471dde8a6abcb3b082b07cd1d1/models/AASIST.py#L325C8-L325C8
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self,
                 out_channels,
                 kernel_size,
                 sample_rate=16000,
                 in_channels=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=False,
                 groups=1,
                 mask=False):
        super().__init__()
        if in_channels != 1:
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (
                in_channels)
            raise ValueError(msg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.mask = mask
        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        NFFT = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)
        fmel = self.to_mel(f)
        fmelmax = np.max(fmel)
        fmelmin = np.min(fmel)
        filbandwidthsmel = np.linspace(fmelmin, fmelmax, self.out_channels + 1)
        filbandwidthsf = self.to_hz(filbandwidthsmel)

        self.mel = filbandwidthsf
        self.hsupp = torch.arange(-(self.kernel_size - 1) / 2,
                                  (self.kernel_size - 1) / 2 + 1)
        self.band_pass = torch.zeros(self.out_channels, self.kernel_size)
        for i in range(len(self.mel) - 1):
            fmin = self.mel[i]
            fmax = self.mel[i + 1]
            hHigh = (2 * fmax / self.sample_rate) * \
                    np.sinc(2 * fmax * self.hsupp / self.sample_rate)
            hLow = (2 * fmin / self.sample_rate) * \
                   np.sinc(2 * fmin * self.hsupp / self.sample_rate)
            hideal = hHigh - hLow

            self.band_pass[i, :] = torch.Tensor(np.hamming(
                self.kernel_size)) * torch.Tensor(hideal)

    def forward(self, x, mask=False):
        band_pass_filter = self.band_pass.clone().to(x.device)
        if mask:
            A = np.random.uniform(0, 20)
            A = int(A)
            A0 = random.randint(0, band_pass_filter.shape[0] - A)
            band_pass_filter[A0:A0 + A, :] = 0
        else:
            band_pass_filter = band_pass_filter

        self.filters = (band_pass_filter).view(self.out_channels, 1,
                                               self.kernel_size)

        return Func.conv1d(x,
                        self.filters,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        bias=None,
                        groups=1)

class ResNetBlock(nn.Module):
    '''
        Residual Blocks can be represented this way:
        in-->[Layer]-->[Layer]-->etc...---+
         |                                |
         |                                [Add]--->[LReLU]---> out
         |                                |
         +---------------->---------------+
    '''
    def __init__(self, in_depth, depth, first=False):
        super(ResNetBlock, self).__init__()
        self.first = first

        self.conv1 = nn.Conv2d(in_depth, depth, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(depth)
        self.lrelu = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(depth, depth, kernel_size=3, stride=3, padding=1)
        self.conv11 = nn.Conv2d(in_depth, depth, kernel_size=3, stride=3, padding=1)
        if not self.first:
            self.pre_bn = nn.BatchNorm2d(in_depth)

    def forward(self, x):
        prev_mp = self.conv11(x)

        out = x
        if not self.first:
            out = self.pre_bn(out)
            out = self.lrelu(out)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.lrelu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        out = out + prev_mp
        out = self.lrelu(out)

        return out

class TernarizeConv2d(nn.Conv2d):
    '''
    Code from: Ternary Neural Networks for Gait Identification in Wearable Devices - Agnetti et al.
    '''
    def __init__(self, *kargs, delta=0.1, is_first=False, f32_activations=False, **kwargs):
        super(TernarizeConv2d, self).__init__(*kargs, **kwargs)
        self.delta = delta
        self.is_first = is_first
        self.f32_activations = f32_activations

    def forward(self, input):
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()

        max_w = self.weight.org.abs().max()
        d = (self.delta * max_w).item()

        # Input ternarization
        if not (self.is_first or self.f32_activations):
            input.data = Ternarize(input.data, d)

        # Weight ternarization
        self.weight.data = Ternarize(self.weight.org, d)

        out = nn.functional.conv2d(input, self.weight, None, self.stride, self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            self.bias.data = Ternarize(self.bias.org, d)  # ternarize bias
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out

class TernarizeLinear(nn.Linear):
    '''
    Code from: Ternary Neural Networks for Gait Identification in Wearable Devices - Agnetti et al.
    '''
    def __init__(self, *kargs, delta=0.1, is_first=False, f32_activations=False, **kwargs):
        super(TernarizeLinear, self).__init__(*kargs, **kwargs)
        self.delta = delta
        self.is_first = is_first
        self.f32_activations = f32_activations

    def forward(self, input):
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()

        max_w = self.weight.org.abs().max()
        d = (self.delta * max_w).item()

        # Input ternarization
        if not (self.is_first or self.f32_activations):
            input.data = Ternarize(input.data, d)

        # TERNARIZE
        self.weight.data = Ternarize(self.weight.org, d)

        out = nn.functional.linear(input, self.weight)

        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            self.bias.data = Ternarize(self.bias.org, d)  # ternarize bias
            out += self.bias.view(1, -1).expand_as(out)

        return out

class ResNetBlock_TNN(nn.Module):
    '''
        Residual Blocks can be represented this way:
        in-->[Layer]-->[Layer]-->etc...---+
         |                                |
         |                                [Add]--->[LReLU]---> out
         |                                |
         +---------------->---------------+
    '''
    def __init__(self, in_depth, depth, device, d_args, delta, first=False):
        super(ResNetBlock_TNN, self).__init__()
        self.first = first
        self.device = device
        self.d_args = d_args

        self.conv1 = TernarizeConv2d(in_depth, depth, kernel_size=3, stride=1, padding=1, delta=delta, f32_activations=d_args['f32_activations'])
        self.bn1 = nn.BatchNorm2d(depth)
        self.lrelu = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(0.5)
        self.conv2 = TernarizeConv2d(depth, depth, kernel_size=3, stride=3, padding=1, delta=delta, f32_activations=d_args['f32_activations'])
        self.conv11 = TernarizeConv2d(in_depth, depth, kernel_size=3, stride=3, padding=1, delta=delta, f32_activations=d_args['f32_activations'])
        if not self.first:
            self.pre_bn = nn.BatchNorm2d(in_depth)

    def forward(self, x):
        prev_mp = self.conv11(x)

        out = x
        if not self.first:
            out = self.pre_bn(out)
            out = self.lrelu(out)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.lrelu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        out = out + prev_mp
        out = self.lrelu(out)

        return out

def Ternarize(tensor, delta):
    # Delta is the threshold
    cond1 = torch.abs(tensor) < delta
    cond2 = tensor >= delta
    cond3 = tensor <= -delta
    t1 = torch.where(cond1, 0.0, tensor)
    t2 = torch.where(cond2, 1.0, t1)
    t3 = torch.where(cond3, -1.0, t2)
    return t3

class LCNN_mel_spec_TNN(nn.Module):
    def __init__(self, device, d_args, delta=0.1, return_emb=False, num_class=2):
        super(LCNN_mel_spec_TNN, self).__init__()

        self.device = device
        self.d_args = d_args

        self.return_emb = return_emb

        # Feature Extraction Part
        self.conv1 = TernarizeConv2d(in_channels=1, out_channels=64, kernel_size=(5, 5),
                                     padding=(2, 2), stride=(1, 1),
                                     delta=delta, is_first=True,
                                     f32_activations=d_args['f32_activations'])
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv4 = TernarizeConv2d(in_channels=32, out_channels=64, kernel_size=(1, 1),
                                     padding=(0, 0), stride=(1, 1),
                                     delta=delta, is_first=False,
                                     f32_activations=d_args['f32_activations'])
        self.batchnorm6 = nn.BatchNorm2d(32)
        self.conv7 = TernarizeConv2d(in_channels=32, out_channels=96, kernel_size=(3, 3),
                                     padding=(1, 1), stride=(1, 1),
                                     delta=delta, is_first=False,
                                     f32_activations=d_args['f32_activations'])
        self.maxpool9 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.batchnorm10 = nn.BatchNorm2d(48)
        self.conv11 = TernarizeConv2d(in_channels=48, out_channels=96, kernel_size=(1, 1),
                                     padding=(0, 0), stride=(1, 1),
                                     delta=delta, is_first=False,
                                     f32_activations=d_args['f32_activations'])
        self.batchnorm13 = nn.BatchNorm2d(48)
        self.conv14 = TernarizeConv2d(in_channels=48, out_channels=128, kernel_size=(3, 3),
                                     padding=(1, 1), stride=(1, 1),
                                     delta=delta, is_first=False,
                                     f32_activations=d_args['f32_activations'])
        self.maxpool16 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv17 = TernarizeConv2d(in_channels=64, out_channels=128, kernel_size=(1, 1),
                                     padding=(0, 0), stride=(1, 1),
                                     delta=delta, is_first=False,
                                     f32_activations=d_args['f32_activations'])
        self.batchnorm19 = nn.BatchNorm2d(64)
        self.conv20 = TernarizeConv2d(in_channels=64, out_channels=64, kernel_size=(3, 3),
                                     padding=(1, 1), stride=(1, 1),
                                     delta=delta, is_first=False,
                                     f32_activations=d_args['f32_activations'])
        self.batchnorm22 = nn.BatchNorm2d(32)
        self.conv23 = TernarizeConv2d(in_channels=32, out_channels=64, kernel_size=(1, 1),
                                     padding=(0, 0), stride=(1, 1),
                                     delta=delta, is_first=False,
                                     f32_activations=d_args['f32_activations'])
        self.batchnorm25 = nn.BatchNorm2d(32)
        self.conv26 = TernarizeConv2d(in_channels=32, out_channels=64, kernel_size=(3, 3),
                                     padding=(1, 1), stride=(1, 1),
                                     delta=delta, is_first=False,
                                     f32_activations=d_args['f32_activations'])
        self.maxpool28 = nn.AdaptiveMaxPool2d((16, 8))

        # Classification Part
        self.fc29 = TernarizeLinear(32 *16 * 8, 128, delta=delta, f32_activations=d_args['f32_activations']) if not d_args['full_fc'] else nn.Linear(32 *16 * 8, 128)
        self.batchnorm31 = nn.BatchNorm1d(64)
        self.fc32 = TernarizeLinear(64, num_class, delta=delta, f32_activations=d_args['f32_activations']) if not d_args['full_fc'] else nn.Linear(64, num_class)

        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(sample_rate=d_args['fs'], n_fft=d_args['spect_nfft'], win_length=d_args['spect_win_length'], hop_length=d_args['spect_hop_length'],
                                                 f_min=d_args['spect_fmin'], f_max=d_args['spect_fmax'], window_fn=torch.hamming_window, n_mels=d_args['spect_n_mels']),
        )
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def mfm2(self, x):
        out1, out2 = torch.chunk(x, 2,1)
        return torch.max(out1, out2)

    def forward(self, x):

        x = x.to(self.device)
        x = x.squeeze(1)  # [B, 1, T] -> [B, T]

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                x = self.torchfbank(x) + 1e-6  # pre-enphasis and melspectrogram
                x = x.log()
                x = x - torch.mean(x, dim=-1, keepdim=True)

        x = self.conv1(x.unsqueeze(1))
        x = self.mfm2(x)
        x = self.maxpool3(x)
        x = self.conv4(x)
        x = self.mfm2(x)
        x = self.batchnorm6(x)
        x = self.conv7(x)
        x = self.mfm2(x)
        x = self.maxpool9(x)
        x = self.batchnorm10(x)
        x = self.conv11(x)
        x = self.mfm2(x)
        x = self.batchnorm13(x)
        x = self.conv14(x)
        x = self.mfm2(x)
        x = self.maxpool16(x)
        x = self.conv17(x)
        x = self.mfm2(x)
        x = self.batchnorm19(x)
        x = self.conv20(x)
        x = self.mfm2(x)
        x = self.batchnorm22(x)
        x = self.conv23(x)
        x = self.mfm2(x)
        x = self.batchnorm25(x)
        x = self.conv26(x)
        x = self.mfm2(x)
        x = self.maxpool28(x)

        x = x.view(x.size(0), -1)
        emb = self.mfm2((self.fc29(x)))
        x = self.batchnorm31(emb)
        logits = self.fc32(x)

        output = self.logsoftmax(logits)

        if self.return_emb:
            return emb
        else:
            return output


    def set_delta(self, delta: float):
        self._delta = delta
        for m in self.modules():
            if isinstance(m, TernarizeConv2d):
                m.delta = delta
            elif isinstance(m, TernarizeLinear):
                m.delta = delta

    def weight_count(self):
        zeros = 0
        plus_ones = 0
        minus_ones = 0

        for p in self.parameters():
            if p is not None:
                zeros += torch.sum(torch.eq(p, 0)).item()
                plus_ones += torch.sum(torch.eq(p, 1)).item()
                minus_ones += torch.sum(torch.eq(p, -1)).item()

        num_params = sum([p.nelement() for p in self.parameters()])
        return (zeros, plus_ones, minus_ones, num_params)

    def get_weights_entropy(self):
        zeros, plus_ones, minus_ones, num_params = self.weight_count()
        ps = torch.tensor([zeros, plus_ones, minus_ones]) / num_params
        return -1 * (ps * torch.log2(ps + 1e-9)).sum().item()

class LCNN_sinc_conv_TNN(nn.Module):
    def __init__(self, device, d_args, delta=0.1, return_emb=False, num_class=2):
        super(LCNN_sinc_conv_TNN, self).__init__()

        self.device = device
        self.d_args = d_args

        self.return_emb = return_emb

        self.pre_emphasis = PreEmphasis()

        self.sinc = SincConv(
            in_channels=1,
            out_channels=self.d_args['sinc_out_channels'],
            kernel_size=self.d_args['sinc_kernel_size'],
            sample_rate=self.d_args['fs'],
            stride=self.d_args['sinc_stride'],
        )

        self.sc_pool = nn.MaxPool2d((3, 3))
        self.sc_bn = nn.BatchNorm2d(num_features=1)
        self.sc_selu = nn.SELU(inplace=True)

        # Feature Extraction Part
        self.conv1 = TernarizeConv2d(in_channels=1, out_channels=64, kernel_size=(5, 5),
                                     padding=(2, 2), stride=(1, 1),
                                     delta=delta, is_first=True,
                                     f32_activations=d_args['f32_activations'])
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv4 = TernarizeConv2d(in_channels=32, out_channels=64, kernel_size=(1, 1),
                                     padding=(0, 0), stride=(1, 1),
                                     delta=delta, is_first=False,
                                     f32_activations=d_args['f32_activations'])
        self.batchnorm6 = nn.BatchNorm2d(32)
        self.conv7 = TernarizeConv2d(in_channels=32, out_channels=96, kernel_size=(3, 3),
                                     padding=(1, 1), stride=(1, 1),
                                     delta=delta, is_first=False,
                                     f32_activations=d_args['f32_activations'])
        self.maxpool9 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.batchnorm10 = nn.BatchNorm2d(48)
        self.conv11 = TernarizeConv2d(in_channels=48, out_channels=96, kernel_size=(1, 1),
                                      padding=(0, 0), stride=(1, 1),
                                      delta=delta, is_first=False,
                                      f32_activations=d_args['f32_activations'])
        self.batchnorm13 = nn.BatchNorm2d(48)
        self.conv14 = TernarizeConv2d(in_channels=48, out_channels=128, kernel_size=(3, 3),
                                      padding=(1, 1), stride=(1, 1),
                                      delta=delta, is_first=False,
                                      f32_activations=d_args['f32_activations'])
        self.maxpool16 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv17 = TernarizeConv2d(in_channels=64, out_channels=128, kernel_size=(1, 1),
                                      padding=(0, 0), stride=(1, 1),
                                      delta=delta, is_first=False,
                                      f32_activations=d_args['f32_activations'])
        self.batchnorm19 = nn.BatchNorm2d(64)
        self.conv20 = TernarizeConv2d(in_channels=64, out_channels=64, kernel_size=(3, 3),
                                      padding=(1, 1), stride=(1, 1),
                                      delta=delta, is_first=False,
                                      f32_activations=d_args['f32_activations'])
        self.batchnorm22 = nn.BatchNorm2d(32)
        self.conv23 = TernarizeConv2d(in_channels=32, out_channels=64, kernel_size=(1, 1),
                                      padding=(0, 0), stride=(1, 1),
                                      delta=delta, is_first=False,
                                      f32_activations=d_args['f32_activations'])
        self.batchnorm25 = nn.BatchNorm2d(32)
        self.conv26 = TernarizeConv2d(in_channels=32, out_channels=64, kernel_size=(3, 3),
                                      padding=(1, 1), stride=(1, 1),
                                      delta=delta, is_first=False,
                                      f32_activations=d_args['f32_activations'])
        self.maxpool28 = nn.AdaptiveMaxPool2d((16, 8))

        # Classification Part
        self.fc29 = TernarizeLinear(32 * 16 * 8, 128, delta=delta, f32_activations=d_args['f32_activations']) if not d_args['full_fc'] else nn.Linear(
            32 * 16 * 8, 128)
        self.batchnorm31 = nn.BatchNorm1d(64)
        self.fc32 = TernarizeLinear(64, num_class, delta=delta, f32_activations=d_args['f32_activations']) if not d_args['full_fc'] else nn.Linear(64, num_class)

        self.logsoftmax = nn.LogSoftmax(dim=1)

    def mfm2(self, x):
        # chunk divide il tensor in N altri tensori (chunks).
        out1, out2 = torch.chunk(x, 2, 1) # Attempts to split a tensor into the specified number of chunks. Each chunk is a view of the input tensor.
        return torch.max(out1, out2)

    def forward(self, x):

        x = x.to(self.device)
        x = x.squeeze(1)  # [B, 1, T] -> [B, T]

        # Pre-emphasis
        x = self.pre_emphasis(x)
        x = x.unsqueeze(1)

        # SincConv
        x = self.sinc(x)
        x = x.unsqueeze(1)
        x = self.sc_pool(x)
        x = self.sc_bn(x)
        x = self.sc_selu(x)

        x = self.conv1(x)
        x = self.mfm2(x)
        x = self.maxpool3(x)
        x = self.conv4(x)
        x = self.mfm2(x)
        x = self.batchnorm6(x)
        x = self.conv7(x)
        x = self.mfm2(x)
        x = self.maxpool9(x)
        x = self.batchnorm10(x)
        x = self.conv11(x)
        x = self.mfm2(x)
        x = self.batchnorm13(x)
        x = self.conv14(x)
        x = self.mfm2(x)
        x = self.maxpool16(x)
        x = self.conv17(x)
        x = self.mfm2(x)
        x = self.batchnorm19(x)
        x = self.conv20(x)
        x = self.mfm2(x)
        x = self.batchnorm22(x)
        x = self.conv23(x)
        x = self.mfm2(x)
        x = self.batchnorm25(x)
        x = self.conv26(x)
        x = self.mfm2(x)
        x = self.maxpool28(x)

        x = x.view(-1, 32 * 16 * 8)
        emb = self.mfm2((self.fc29(x)))
        x = self.batchnorm31(emb)
        logits = self.fc32(x)

        output = self.logsoftmax(logits)

        if self.return_emb:
            return emb
        else:
            return output

    def set_delta(self, delta: float):
        self._delta = delta
        for m in self.modules():
            if isinstance(m, TernarizeConv2d):
                m.delta = delta
            elif isinstance(m, TernarizeLinear):
                m.delta = delta

    def weight_count(self):
        zeros = 0
        plus_ones = 0
        minus_ones = 0

        for p in self.parameters():
            if p is not None:
                zeros += torch.sum(torch.eq(p, 0)).item()
                plus_ones += torch.sum(torch.eq(p, 1)).item()
                minus_ones += torch.sum(torch.eq(p, -1)).item()

        num_params = sum([p.nelement() for p in self.parameters()])
        return (zeros, plus_ones, minus_ones, num_params)

    def get_weights_entropy(self):
        zeros, plus_ones, minus_ones, num_params = self.weight_count()
        ps = torch.tensor([zeros, plus_ones, minus_ones]) / num_params
        return -1 * (ps * torch.log2(ps + 1e-9)).sum().item()

class Resnet_mel_spec(nn.Module):
    def __init__(self, device, d_args):
        super(Resnet_mel_spec, self).__init__()

        self.device = device
        self.d_args = d_args

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.block1 = ResNetBlock(32, 32, True)
        # self.mp = nn.MaxPool2d(3, stride=3, padding=1)
        # self.block2 = ResNetBlock(32, 32, False)
        self.block3 = ResNetBlock(32, 32, False)
        # self.block4 = ResNetBlock(32, 32, False)
        self.block5 = ResNetBlock(32, 32, False)
        # self.block6 = ResNetBlock(32, 32, False)
        self.block7 = ResNetBlock(32, 32, False)
        # self.block8 = ResNetBlock(32, 32, False)
        self.block9 = ResNetBlock(32, 32, False)
        # self.block10 = ResNetBlock(32, 32, False)
        self.block11 = ResNetBlock(32, 32, False)
        self.lrelu = nn.LeakyReLU(0.01)
        self.bn = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.5)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        # self.fc1 = nn.Linear(64, 128)
        self.fc1 = nn.Linear(32, 16)
        # self.fc2 = nn.Linear(128, 2)
        self.fc2 = nn.Linear(16, 2)

        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),
            T.MelSpectrogram(sample_rate=self.d_args['fs'],
                             n_fft=self.d_args['spect_nfft'],
                             win_length=self.d_args['spect_win_length'],
                             hop_length=self.d_args['spect_hop_length'],
                             f_min=self.d_args['spect_fmin'],
                             f_max=self.d_args['spect_fmax'],
                             window_fn=torch.hamming_window,
                             n_mels=self.d_args['spect_n_mels'], ),
        )

    def forward(self, x):
        batch_size = x.size(0)

        x = x.to(self.device)
        x = x.squeeze(1)  # [B, 1, T] -> [B, T]
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                x = self.torchfbank(x) + 1e-6 # pre-enphasis and melspectrogram
                x = x.log()
                x = x - torch.mean(x, dim=-1, keepdim=True)

        x = x.unsqueeze(dim=1)
        out = self.conv1(x)
        out = self.block1(out)
        # out = self.block2(out)
        # out = self.mp(out)
        out = self.block3(out)
        # out = self.block4(out)
        # out = self.mp(out)
        out = self.block5(out)
        # out = self.block6(out)
        # out = self.mp(out)
        out = self.block7(out)
        # out = self.block8(out)
        # out = self.mp(out)
        out = self.block9(out)
        # out = self.block10(out)
        # out = self.mp(out)
        out = self.block11(out)
        out = self.bn(out)
        out = self.lrelu(out)
        # out = self.mp(out)
        out = out.view(batch_size, -1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.lrelu(out)
        out = self.fc2(out)
        out = self.logsoftmax(out)
        return out

class Resnet_mel_spec_TNN(nn.Module):
    def __init__(self, device, d_args, delta=0.1):
        super(Resnet_mel_spec_TNN, self).__init__()

        self.device = device
        self.d_args = d_args

        # self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv1 = TernarizeConv2d(1, 32, kernel_size=3, stride=1, padding=1, is_first=True, delta=delta, f32_activations=d_args['f32_activations'])
        self.block1 = ResNetBlock_TNN(32, 32, self.device, self.d_args, delta, first=True)
        #self.mp = nn.MaxPool2d(3, stride=3, padding=1)
        # self.block2 = ResNetBlock_TNN(32, 32, self.device, self.d_args, delta, first=False)
        self.block3 = ResNetBlock_TNN(32, 32, self.device, self.d_args, delta, first=False)
        # self.block4 = ResNetBlock_TNN(32, 32, self.device, self.d_args, delta, first=False)
        self.block5 = ResNetBlock_TNN(32, 32, self.device, self.d_args, delta, first=False)
        # self.block6 = ResNetBlock_TNN(32, 32, self.device, self.d_args, delta, first=False)
        self.block7 = ResNetBlock_TNN(32, 32, self.device, self.d_args, delta, first=False)
        # self.block8 = ResNetBlock_TNN(32, 32, self.device, self.d_args, delta, first=False)
        self.block9 = ResNetBlock_TNN(32, 32, self.device, self.d_args, delta, first=False)
        # self.block10 = ResNetBlock_TNN(32, 32, self.device, self.d_args, delta, first=False)
        self.block11 = ResNetBlock_TNN(32, 32, self.device, self.d_args, delta, first=False)
        self.lrelu = nn.LeakyReLU(0.01)
        self.bn = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.5)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc1 = TernarizeLinear(32, 16, delta=delta, f32_activations=d_args['f32_activations']) if not d_args['full_fc'] else nn.Linear(32, 16)
        self.fc2 = TernarizeLinear(16, 2, delta=delta, f32_activations=d_args['f32_activations']) if not d_args['full_fc'] else nn.Linear(16, 2)

        self.torchfbank = torch.nn.Sequential(
            PreEmphasis(),
            T.MelSpectrogram(sample_rate=self.d_args['fs'],
                             n_fft=self.d_args['spect_nfft'],
                             win_length=self.d_args['spect_win_length'],
                             hop_length=self.d_args['spect_hop_length'],
                             f_min=self.d_args['spect_fmin'],
                             f_max=self.d_args['spect_fmax'],
                             window_fn=torch.hamming_window,
                             n_mels=self.d_args['spect_n_mels'], ),
        )

    def forward(self, x):
        batch_size = x.size(0)

        x = x.to(self.device)
        x = x.squeeze(1)  # [B, 1, T] -> [B, T]
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                x = self.torchfbank(x) + 1e-6 # pre-emphasis and melspectrogram
                x = x.log()
                x = x - torch.mean(x, dim=-1, keepdim=True)

        x = x.unsqueeze(dim=1)
        out = self.conv1(x)
        out = self.block1(out)
        # out = self.block2(out)
        # out = self.mp(out)
        out = self.block3(out)
        # out = self.block4(out)
        # out = self.mp(out)
        out = self.block5(out)
        # out = self.block6(out)
        # out = self.mp(out)
        out = self.block7(out)
        # out = self.block8(out)
        # out = self.mp(out)
        out = self.block9(out)
        # out = self.block10(out)
        # out = self.mp(out)
        out = self.block11(out)
        out = self.bn(out)
        out = self.lrelu(out)
        # out = self.mp(out)
        out = out.view(batch_size, -1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.lrelu(out)
        out = self.fc2(out)
        out = self.logsoftmax(out)
        return out

    def set_delta(self, delta: float):
        self._delta = delta
        for m in self.modules():
            if isinstance(m, TernarizeConv2d):
                m.delta = delta
            elif isinstance(m, TernarizeLinear):
                m.delta = delta

    def weight_count(self):
        zeros = 0
        plus_ones = 0
        minus_ones = 0

        for p in self.parameters():
            if p is not None:
                zeros += torch.sum(torch.eq(p, 0)).item()
                plus_ones += torch.sum(torch.eq(p, 1)).item()
                minus_ones += torch.sum(torch.eq(p, -1)).item()

        num_params = sum([p.nelement() for p in self.parameters()])
        return (zeros, plus_ones, minus_ones, num_params)

    def get_weights_entropy(self):
        zeros, plus_ones, minus_ones, num_params = self.weight_count()
        ps = torch.tensor([zeros, plus_ones, minus_ones]) / num_params
        return -1 * (ps * torch.log2(ps + 1e-9)).sum().item()


