# Adapted from Music-Source-Separation-Training by Roman Solovyev (ZFTurbo),
# MIT License, Copyright (c) 2024. Original BandIt model by Karn Watcharasupat et al.
from typing import Dict, Optional
import torch
from torch import nn


class _Stft(nn.Module):
    """torchaudio.transforms.Spectrogram(power=None, normalized=True), in plain torch."""
    def __init__(self, n_fft, win_length, hop_length, window_fn, wkwargs, normalized, center, pad_mode, onesided):
        super().__init__()
        self.n_fft, self.win_length, self.hop_length = n_fft, win_length or n_fft, hop_length
        self.register_buffer("window", window_fn(self.win_length, **(wkwargs or {})))
        self.normalized, self.center, self.pad_mode, self.onesided = normalized, center, pad_mode, onesided

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        s = torch.stft(x, self.n_fft, self.hop_length, self.win_length, self.window, center=self.center,
                       pad_mode="constant" if self.pad_mode == "constant" else self.pad_mode,
                       normalized=False, onesided=self.onesided, return_complex=True)
        if self.normalized:
            s = s / self.window.pow(2.0).sum().sqrt()
        return s.reshape(shape[:-1] + s.shape[-2:])


class _Istft(_Stft):
    def forward(self, s, length=None):
        shape = s.shape
        s = s.reshape(-1, shape[-2], shape[-1])
        if self.normalized:
            s = s * self.window.pow(2.0).sum().sqrt()
        x = torch.istft(s, self.n_fft, self.hop_length, self.win_length, self.window, center=self.center,
                        normalized=False, onesided=self.onesided, length=length, return_complex=False)
        return x.reshape(shape[:-2] + x.shape[-1:])


class _SpectralComponent(nn.Module):
    def __init__(self, n_fft: int = 2048, win_length: Optional[int] = 2048, hop_length: int = 512,
                 window_fn: str = "hann_window", wkwargs: Optional[Dict] = None, power: Optional[int] = None,
                 center: bool = True, normalized: bool = True, pad_mode: str = "constant", onesided: bool = True,
                 **kwargs) -> None:
        super().__init__()
        assert power is None
        window_fn = torch.__dict__[window_fn]
        a = (n_fft, win_length, hop_length, window_fn, wkwargs, normalized, center, pad_mode, onesided)
        self.stft = _Stft(*a)
        self.istft = _Istft(*a)
