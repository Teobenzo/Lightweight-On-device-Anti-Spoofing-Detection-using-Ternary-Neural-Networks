import librosa
import numpy as np
import warnings
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as F
import random


warnings.filterwarnings("ignore")

def read_audio(audio_path, dur=180, fs=16000, norm=False, trim=False, int_type=False, windowing=False,
               mulaw=False, g722=False, rir=False, noise_Inj_10db=False, noise_Inj_20db=False, opus=False,
               vorbis=False, envNoise_20db=False, envNoise_15db=False, whiteNoise_inj_20db=False, whiteNoise_inj_10db=False):
    try:
        X, fs_orig = librosa.load(audio_path, sr=None, duration=dur)
    except Exception as e:
        raise ValueError(f"[AUDIO LOAD ERROR] path={audio_path} -> {type(e).__name__}: {e}")

    # RESAMPLING
    if not fs:
        fs = fs_orig
    if fs_orig != fs:
        X = librosa.resample(X, orig_sr=fs_orig, target_sr=fs)

    if norm:
        peak = np.max(np.abs(X))
        if peak > 0:
            X = X / peak

    if trim:
        X, _ = librosa.effects.trim(X, top_db=40)

    if int_type:
        X = (X * 32768).astype(np.int32)
    if windowing:
        win_len = 3 # in seconds
        mask = np.zeros(dur*fs).astype(bool)
        for ii in range(mask.shape[0]//(win_len*fs)):
            mask[ii*win_len*fs:ii*win_len*fs+fs] = True
            mask = mask[:X.shape[0]]
        X = X[mask]

        sf.write(audio_path, X, fs)

    if mulaw:
        X = apply_codec(X, fs, "wav", encoder="pcm_mulaw")

    if g722:
        X = apply_codec(X, fs, "wav", encoder="g722")

    global noise
    global fs_noise
    if noise_Inj_10db:
        assert fs == 8000, "fs rumore diverso da fs segnale"

        # snr as tensor
        snr_db = 10
        snr_db = torch.tensor(snr_db).float()
        snr_db = snr_db.unsqueeze(0)

        # X to tensor
        X = torch.from_numpy(X).float()
        X = X.unsqueeze(0)

        noise_inj = noise
        # Noise padding
        if X.shape[1] > noise.shape[1]:
            num_repeats = int(X.shape[1] / noise.shape[1]) + 1
            noise_inj = noise.repeat(1, num_repeats)

        noise_inj = noise_inj[:, :X.shape[1]]

        X = F.add_noise(X, noise_inj, snr_db)

        X = X.squeeze().numpy()

    if noise_Inj_20db:
        assert fs == 8000, "fs rumore diverso da fs segnale"

        # snr as tensor
        snr_db = 20
        snr_db = torch.tensor(snr_db).float()
        snr_db = snr_db.unsqueeze(0)

        # X to tensor
        X = torch.from_numpy(X).float()
        X = X.unsqueeze(0)

        noise_inj = noise
        # noise padding
        if X.shape[1] > noise.shape[1]:
            num_repeats = int(X.shape[1] / noise.shape[1]) + 1
            noise_inj = noise.repeat(1, num_repeats)

        noise_inj = noise_inj[:, :X.shape[1]]

        X = F.add_noise(X, noise_inj, snr_db)

        X = X.squeeze().numpy()

    global rir_files
    if rir:
        assert fs == 8000, "fs sample RIR a 8000"

        # X to tensor
        X = torch.from_numpy(X).float()
        X = X.unsqueeze(0)

        # random rir choice
        rir_file = random.choice(rir_files)

        rir_file = torch.from_numpy(rir_file).float()
        rir_file = rir_file.unsqueeze(0)
        rir_file = rir_file / torch.linalg.vector_norm(rir_file, ord=2)

        X = F.fftconvolve(X, rir_file)

        X = X.squeeze().numpy()

    if opus:
        X = apply_codec(X, fs, format="ogg", encoder="opus")

    if vorbis:
        X = apply_codec(X, fs, format="ogg", encoder="vorbis")

    global wham_files
    if envNoise_20db:
        assert fs == 8000, "fs rumore diverso da fs segnale"

        # snr as tensor
        snr_db = 20
        snr_db = torch.tensor(snr_db).float()
        snr_db = snr_db.unsqueeze(0)

        # X to tensor
        X = torch.from_numpy(X).float()
        X = X.unsqueeze(0)

        # SELECT NOISE
        wham_file_path = random.choice(wham_files)
        wham_file_path = "/nas/home/mbenzo/tesi/dataset/wham_noise/tt/" + wham_file_path
        noise_inj, fs_wham = librosa.load(wham_file_path, sr=16000, mono=True)  # WHAM! sample rate 16kHZ
        noise_inj = librosa.resample(noise_inj, orig_sr=16000, target_sr=8000)  # resample to 8kHz

        noise_inj = torch.from_numpy(noise_inj).float()
        noise_inj = noise_inj.unsqueeze(0)

        # noise padding
        if X.shape[1] > noise_inj.shape[1]:
            num_repeats = int(X.shape[1] / noise_inj.shape[1]) + 1
            noise_inj = noise_inj.repeat(1, num_repeats)

        noise_inj = noise_inj[:, :X.shape[1]]

        X = F.add_noise(X, noise_inj, snr_db)

        X = X.squeeze().numpy()

    if envNoise_15db:
        assert fs == 8000, "fs rumore diverso da fs segnale"

        # snr as tensor
        snr_db = 15
        snr_db = torch.tensor(snr_db).float()
        snr_db = snr_db.unsqueeze(0)

        # X to tensor
        X = torch.from_numpy(X).float()
        X = X.unsqueeze(0)

        # SELECT NOISE
        wham_file_path = random.choice(wham_files)
        wham_file_path = "/nas/home/mbenzo/tesi/dataset/wham_noise/tt/" + wham_file_path
        noise_inj, fs_wham = librosa.load(wham_file_path, sr=16000, mono=True)  # WHAM! sample 16kHZ
        noise_inj = librosa.resample(noise_inj, orig_sr=16000, target_sr=8000)  # resample to 8kHz

        noise_inj = torch.from_numpy(noise_inj).float()
        noise_inj = noise_inj.unsqueeze(0)  # [1, T]

        # noise padding
        if X.shape[1] > noise_inj.shape[1]:
            num_repeats = int(X.shape[1] / noise_inj.shape[1]) + 1
            noise_inj = noise_inj.repeat(1, num_repeats)

        noise_inj = noise_inj[:, :X.shape[1]]

        X = F.add_noise(X, noise_inj, snr_db)

        X = X.squeeze().numpy()

    if whiteNoise_inj_20db:
        assert fs == 8000, "fs rumore diverso da fs segnale"

        # snr as tensor
        snr_db = 20
        snr_db = torch.tensor(snr_db).float()
        snr_db = snr_db.unsqueeze(0)

        # X to tensor
        X = torch.from_numpy(X).float()
        X = X.unsqueeze(0)

        fs_wn = 8000
        duration_s_wn = 4
        n_samples = duration_s_wn * fs_wn
        noise_inj = np.random.randn(n_samples).astype(np.float32)

        noise_inj = torch.from_numpy(noise_inj).float()
        noise_inj = noise_inj.unsqueeze(0)  # [1, T]

        # noise padding
        if X.shape[1] > noise_inj.shape[1]:
            num_repeats = int(X.shape[1] / noise_inj.shape[1]) + 1
            noise_inj = noise_inj.repeat(1, num_repeats)

        noise_inj = noise_inj[:, :X.shape[1]]

        X = F.add_noise(X, noise_inj, snr_db)

        X = X.squeeze().numpy()

    if whiteNoise_inj_10db:
        assert fs == 8000, "fs rumore diverso da fs segnale"

        # snr as tensor
        snr_db = 10
        snr_db = torch.tensor(snr_db).float()
        snr_db = snr_db.unsqueeze(0)

        # X to tensor
        X = torch.from_numpy(X).float()
        X = X.unsqueeze(0)

        fs_wn = 8000
        duration_s_wn = 4
        n_samples = duration_s_wn * fs_wn
        noise_inj = np.random.randn(n_samples).astype(np.float32)

        noise_inj = torch.from_numpy(noise_inj).float()
        noise_inj = noise_inj.unsqueeze(0)  # [1, T]

        # noise padding
        if X.shape[1] > noise_inj.shape[1]:
            num_repeats = int(X.shape[1] / noise_inj.shape[1]) + 1
            noise_inj = noise_inj.repeat(1, num_repeats)

        noise_inj = noise_inj[:, :X.shape[1]]

        X = F.add_noise(X, noise_inj, snr_db)

        X = X.squeeze().numpy()

    return X, fs


def apply_codec(waveform_np, sample_rate, format, encoder=None):

    wav = torch.as_tensor(waveform_np, dtype=torch.float32)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)  # [1, T]
        wav = wav.transpose(0, 1)  # [T, 1]

    encoder = torchaudio.io.AudioEffector(format=format, encoder=encoder)
    enc_wav = encoder.apply(wav, sample_rate)

    wav = enc_wav.numpy()
    wav = wav.squeeze()
    return wav