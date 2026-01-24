import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class WeightedOmniPhaseLoss(nn.Module):
    def __init__(self, alpha=100):
        super(WeightedOmniPhaseLoss, self).__init__()
        self.alpha = alpha
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

    def anti_wrapping_function(self, x):
        return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)

    def forward(self, phase_r, phase_g, mag_r=None):
        """
        phase_r: (B, F, T)
        phase_g: (B, F, T)
        mag_r: (B, F, T)
        """

        if mag_r.ndim == 3:
            mag_r = mag_r.unsqueeze(1)  # (B, 1, F, T)

        mag_r = (mag_r / torch.max(mag_r) * self.alpha).transpose(-2, -1).contiguous()

        phase_r = phase_r.transpose(-2, -1).unsqueeze(1)  # (B,1,T,F)
        phase_g = phase_g.transpose(-2, -1).unsqueeze(1)  # (B,1,T,F)

        phase_r = F.conv2d(phase_r, self.filters.to(phase_r.device), bias=None, stride=1, padding=1)  # (B,9,T,F)
        phase_g = F.conv2d(phase_g, self.filters.to(phase_r.device), bias=None, stride=1, padding=1)  # (B,9,T,F)
        loss = 3 * torch.mean(mag_r * self.anti_wrapping_function(phase_g - phase_r))

        return loss


class CoupledOmniRILoss(nn.Module):
    def __init__(self, mag_dist_type="l1"):
        self.mag_dist_type = mag_dist_type
        super(CoupledOmniRILoss, self).__init__()
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


    def anti_wrapping_function(self, x):
        return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)


    def forward(self, rea, imag, rea_g, imag_g, phase_clip=True):
        """
        rea: target real, (B, F, T)
        imag: target imaginary, (B, F, T)
        rea_g: estimate real, (B, F, T)
        imag_g: estimate imaginary, (B, F, T)
        """
        mag, mag_g = torch.sqrt(rea ** 2 + imag ** 2 + 1e-8).unsqueeze(1), \
            torch.sqrt(rea_g ** 2 + imag_g ** 2 + 1e-8).unsqueeze(1)

        if phase_clip:
            rea_g_clipped = torch.where(torch.abs(rea_g) < 1e-8, 1e-8, rea_g)
            imag_g_clipped = torch.where(torch.abs(imag_g) < 1e-8, 1e-8, imag_g)
            pha, pha_g = torch.atan2(imag, rea).unsqueeze(1), \
                torch.atan2(imag_g_clipped, rea_g_clipped).unsqueeze(1)
        else:
            pha, pha_g = torch.atan2(imag, rea).unsqueeze(1), \
                torch.atan2(imag_g, rea_g).unsqueeze(1)

        pha = F.conv2d(pha, self.filters.to(pha.device), bias=None, stride=1, padding=1)
        pha_g = F.conv2d(pha_g, self.filters.to(pha.device), bias=None, stride=1, padding=1)

        if self.mag_dist_type.upper() == "L1":
            loss = 2 * torch.mean(
                torch.abs(mag - mag_g).repeat(1, self.filters.shape[0], 1, 1) * self.anti_wrapping_function(pha_g - pha))
        elif self.mag_dist_type.upper() == "L2":
            loss = 2 * torch.mean(
                torch.square(mag - mag_g).repeat(1, self.filters.shape[0], 1, 1) * self.anti_wrapping_function(pha_g - pha))
        return loss


class OmniRILoss(nn.Module):
    def __init__(self, mag_dist_type="l1"):
        super(OmniRILoss, self).__init__()
        self.mag_dist_type = mag_dist_type

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

    def forward(self, rea, imag, rea_g, imag_g, phase_clip=True):
        """
        rea: target real, (B, F, T)
        imag: target imaginary, (B, F, T)
        rea_g: estimate real, (B, F, T)
        imag_g: estimate imaginary, (B, F, T)
        """
        mag, mag_g = torch.sqrt(rea ** 2 + imag ** 2 + 1e-8).unsqueeze(1), torch.sqrt(
            rea_g ** 2 + imag_g ** 2 + 1e-8).unsqueeze(1)

        if phase_clip:
            rea_g_clipped = torch.where(torch.abs(rea_g) < 1e-8, 1e-8, rea_g)
            imag_g_clipped = torch.where(torch.abs(imag_g) < 1e-8, 1e-8, imag_g)
            pha, pha_g = torch.atan2(imag, rea).unsqueeze(1), \
                torch.atan2(imag_g_clipped, rea_g_clipped).unsqueeze(1)
        else:
            pha, pha_g = torch.atan2(imag, rea).unsqueeze(1), \
                torch.atan2(imag_g, rea_g).unsqueeze(1)

        pha = F.conv2d(pha, self.filters.to(pha.device), bias=None, stride=1, padding=1)
        pha_g = F.conv2d(pha_g, self.filters.to(pha.device), bias=None, stride=1, padding=1)

        cur_rea, cur_rea_g = mag.repeat(1, self.filters.shape[0], 1, 1) * torch.cos(pha), \
                             mag_g.repeat(1, self.filters.shape[0], 1, 1) * torch.cos(pha_g)
        cur_imag, cur_imag_g = mag.repeat(1, self.filters.shape[0], 1, 1) * torch.sin(pha), \
                               mag_g.repeat(1, self.filters.shape[0], 1, 1) * torch.sin(pha_g)

        if self.mag_dist_type.upper() == "L1":
            loss_R, loss_I = torch.abs(cur_rea - cur_rea_g), torch.abs(cur_imag - cur_imag_g)
        elif self.mag_dist_type.upper() == "L2":
            loss_R, loss_I = torch.square(cur_rea - cur_rea_g), torch.square(cur_imag - cur_imag_g)

        loss = (loss_R.mean() + loss_I.mean())

        return loss

