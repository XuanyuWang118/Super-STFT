import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class GOMPSNR(nn.Module):
    def __init__(self, snr_type="gompsnr", win_length=1024, n_fft=1024, hop_length=256,):
        super(GOMPSNR, self).__init__()
        kernel1 = torch.from_numpy(np.array([[-1., 0, 0], [0, 1, 0], [0, 0, 0]], dtype='float32'))
        kernel2 = torch.from_numpy(np.array([[0, -1., 0], [0, 1., 0], [0, 0, 0]], dtype='float32'))
        kernel3 = torch.from_numpy(np.array([[0, 0, -1.], [0, 1., 0], [0, 0, 0]], dtype='float32'))
        kernel4 = torch.from_numpy(np.array([[0, 0, 0], [-1., 1., 0], [0, 0, 0]], dtype='float32'))
        kernel5 = torch.from_numpy(np.array([[0, 0, 0], [0, 1., 0], [0, 0, 0]], dtype='float32'))
        kernel6 = torch.from_numpy(np.array([[0, 0, 0], [0, 1., -1.], [0, 0, 0]], dtype='float32'))
        kernel7 = torch.from_numpy(np.array([[0, 0, 0], [0, 1., 0], [-1., 0, 0]], dtype='float32'))
        kernel8 = torch.from_numpy(np.array([[0, 0, 0], [0, 1., 0], [0, -1., 0]], dtype='float32'))
        kernel9 = torch.from_numpy(np.array([[0, 0, 0], [0, 1., 0], [0, 0, -1.]], dtype='float32'))
        kernels = torch.stack([kernel1, kernel2, kernel3, kernel4, kernel5, kernel6, kernel7, kernel8, kernel9],
                              dim=0)  # (out_nch, 3, 3)
        kernels = kernels.unsqueeze(1)
        self.filters = kernels

        self.snr_type = snr_type
        self.win_length = win_length
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window = torch.hann_window(self.win_length)
        self.window = self.window / torch.sqrt(torch.sum(self.window**2))

    def g0(self, x):
        return -2 * torch.cos(x)

    def anti_wrapping_function(self, x):
        return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)

    def g1(self, x):
        return 2 * self.anti_wrapping_function(x) / np.pi - 2



    def forward(self, y, y_g):
        """
        y: target signal, (B, T)
        y_g: synthetic signal, (B, T)
        """

        spec = torch.stft(y, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
                          window=self.window.to(y.device), return_complex=True)
        spec_g = torch.stft(y_g, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
                          window=self.window.to(y.device), return_complex=True)
        rea, imag = spec.real, spec.imag
        rea_g, imag_g = spec_g.real, spec_g.imag
        if self.snr_type.lower() == "snr":  # vanilla snr
            snr = 10 * torch.log10(torch.sum(rea ** 2 + imag ** 2, dim=[1, 2]) / (
                    torch.sum((rea - rea_g) ** 2 + (imag - imag_g) ** 2, dim=[1, 2]) + 1e-8))
            return snr
        elif self.snr_type.lower() == "ompsnr":
            nomin = torch.sum(rea ** 2 + imag ** 2, dim=[1, 2])
            mag, mag_g = torch.sqrt(rea ** 2 + imag ** 2 + 1e-8).unsqueeze(1), \
                torch.sqrt(rea_g ** 2 + imag_g ** 2 + 1e-8).unsqueeze(1)
            pha, pha_g = torch.atan2(imag, rea).unsqueeze(1), torch.atan2(imag_g, rea_g).unsqueeze(1)
            omni_phase, omni_phase_g = F.conv2d(pha, self.filters.to(pha.device), bias=None, stride=1, padding=1,
                                                groups=1), \
                F.conv2d(pha_g, self.filters.to(pha.device), bias=None, stride=1, padding=1, groups=1)
            mag, mag_g = mag.repeat(1, self.filters.shape[0], 1, 1), mag_g.repeat(1, self.filters.shape[0], 1,
                                                                                  1)  # (B, 9, F, T)
            cross_part = mag * mag_g * self.g0(omni_phase - omni_phase_g)
            denomin = (mag ** 2 + mag_g ** 2 + cross_part).sum([2, 3]).mean(1) + 1e-8  # (B, 9, F, T)->(B, 9)->(B,)
            ompsnr = 10 * torch.log10(nomin / denomin)
            return ompsnr
        elif self.snr_type.lower() == "gompsnr":
            nomin = torch.sum(rea ** 2 + imag ** 2, dim=[1, 2])
            mag, mag_g = torch.sqrt(rea ** 2 + imag ** 2 + 1e-8).unsqueeze(1), \
                torch.sqrt(rea_g ** 2 + imag_g ** 2 + 1e-8).unsqueeze(1)
            pha, pha_g = torch.atan2(imag, rea).unsqueeze(1), torch.atan2(imag_g, rea_g).unsqueeze(1)
            omni_phase, omni_phase_g = F.conv2d(pha, self.filters.to(pha.device), bias=None, stride=1, padding=1,
                                                groups=1), \
                F.conv2d(pha_g, self.filters.to(pha.device), bias=None, stride=1, padding=1, groups=1)
            mag, mag_g = mag.repeat(1, self.filters.shape[0], 1, 1), mag_g.repeat(1, self.filters.shape[0], 1,
                                                                                  1)  # (B, 9, F, T)
            cross_part = mag * mag_g * self.g1(omni_phase - omni_phase_g)
            denomin = (mag ** 2 + mag_g ** 2 + cross_part).sum([2, 3]).mean(1) + 1e-8  # (B, 9, F, T) -> (B, 9) -> (B,)
            gompsnr = 10 * torch.log10(nomin / denomin)
            return gompsnr
        else:
            raise NotImplementedError("Only support for SNR, OMPSNR and GOMPSNR")


