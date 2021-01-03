import torch
import numpy as np
from torch import nn
from torch.nn import functional
from collections import namedtuple
from model.utils import conv_power_method, calc_pad_sizes
import random


class SoftThreshold(nn.Module):
    def __init__(self, size, init_threshold=1e-3):
        super(SoftThreshold, self).__init__()
        self.threshold = nn.Parameter(init_threshold * torch.ones(1,size,1,1))

    def forward(self, x):
        mask1 = (x > self.threshold).float()
        mask2 = (x < -self.threshold).float()
        out = mask1.float() * (x - self.threshold)
        out += mask2.float() * (x + self.threshold)
        return out


ListaParams = namedtuple('ListaParams', ['kernel_size', 'num_filters', 'stride', 'unfoldings', 'scale_levels',
                                         'num_supports'])


class ConvLista_T(nn.Module):
    def __init__(self, params: ListaParams, A=None, B=None, C=None, threshold=1e-2):
        super(ConvLista_T, self).__init__()
        if A is None:
            A = torch.randn(params.num_filters, 1, params.kernel_size, params.kernel_size)
            l = conv_power_method(A, [512, 512], num_iters=200, stride=params.stride)
            A /= torch.sqrt(l)
        if B is None:
            B = torch.clone(A)
        if C is None:
            C = torch.clone(A)

        self.params = params

        self.apply_A, self.apply_B, self.apply_C = [], [], []
        self.soft_threshold = []
        self.downsample, self.upsample = [], []
        for i in range(self.params.scale_levels):
            self.apply_A.append(nn.ConvTranspose2d(params.num_filters, 1, kernel_size=params.kernel_size,
                                                         stride=params.stride, bias=False))
            self.apply_B.append(nn.Conv2d(1, params.num_filters, kernel_size=params.kernel_size,
                                                stride=params.stride, bias=False))
            self.apply_C.append(nn.ConvTranspose2d(params.num_filters, 1, kernel_size=params.kernel_size,
                                                         stride=params.stride, bias=False))
            self.apply_A[i].weight.data = A
            self.apply_B[i].weight.data = B
            self.apply_C[i].weight.data = C
            self.soft_threshold.append(SoftThreshold(params.num_filters, threshold))
            # TODO
            self.downsample.append(nn.Identity())
            self.upsample.append(nn.Identity())
        self.apply_A = nn.ModuleList(self.apply_A)
        self.apply_B = nn.ModuleList(self.apply_B)
        self.apply_C = nn.ModuleList(self.apply_C)
        self.soft_threshold = nn.ModuleList(self.soft_threshold)
        self.downsample = nn.ModuleList(self.downsample)
        self.upsample = nn.ModuleList(self.upsample)

    def _sample_supports(self, seed=None):
        if seed is not None:
            random.seed(seed)
        total_options = 1
        total_options = self.params.stride ** (2 * self.params.scale_levels)
        result = random.sample(range(total_options), self.params.num_supports)
        for i in range(self.params.num_supports):
            res = []
            cur_sample = result[i]
            for level in range(self.params.scale_levels):
                row = cur_sample % self.params.stride
                cur_sample = cur_sample // self.params.stride
                col = cur_sample % self.params.stride
                cur_sample = cur_sample // self.params.stride
                res.append((row, col))
            result[i] = res
        return result

    def _split_images(self, I, seed=None):
        if self.params.stride == 1:
            return I, [torch.ones_like(image, dtype=torch.bool) for image in I]

        left_pads, right_pads, top_pads, bot_pads = [], [], [], []
        for image in I:
            left_pad, right_pad, top_pad, bot_pad = calc_pad_sizes(image, self.params.kernel_size, self.params.stride)
            left_pads.append(left_pad)
            right_pads.append(right_pad)
            top_pads.append(top_pad)
            bot_pads.append(bot_pad)

        supports = self._sample_supports(seed)

        I_batched_padded = [torch.zeros(I[i].shape[0],
                                        self.params.stride ** 2,
                                        I[i].shape[1],
                                        top_pads[i] + I[i].shape[2] + bot_pads[i],
                                        left_pads[i] + I[i].shape[3] + right_pads[i]).type_as(I[i]) for i in range(self.params.scale_levels)]
        valids_batched = [torch.zeros_like(I_batched_padded[i], dtype=torch.bool) for i in range(self.params.scale_levels)]

        for num, support in enumerate(supports):
            for i in range(self.params.scale_levels):
                row_shift, col_shift = support[i]
                left_pad, right_pad, top_pad, bot_pad = left_pads[i], right_pads[i], top_pads[i], bot_pads[i]
                I_padded = functional.pad(I[i], pad=(
                    left_pad - col_shift, right_pad + col_shift, top_pad - row_shift, bot_pad + row_shift), mode='reflect')
                valids = functional.pad(torch.ones_like(I[i], dtype=torch.bool), pad=(
                    left_pad - col_shift, right_pad + col_shift, top_pad - row_shift, bot_pad + row_shift), mode='constant')
                I_batched_padded[i][:, num, :, :, :] = I_padded
                valids_batched[i][:, num, :, :, :] = valids

        I_batched_padded = [I_batched_padded[i].reshape(-1, *I_batched_padded[i].shape[2:]) for i in range(self.params.scale_levels)]
        valids_batched = [valids_batched[i].reshape(-1, *valids_batched[i].shape[2:]) for i in range(self.params.scale_levels)]
        return I_batched_padded, valids_batched

    def forward(self, I, seed=None):
        scaled_images = [self.downsample[i](I) for i in range(self.params.scale_levels)]
        I_batched_padded, valids_batched = self._split_images(scaled_images, seed)
        conv_input = [self.apply_B[i](I_batched_padded[i]) for i in range(self.params.scale_levels)]
        gamma_k = [self.soft_threshold[i](conv_input[i]) for i in range(self.params.scale_levels)]
        for k in range(self.params.unfoldings - 1):
            # Get current residual
            r_k = I_batched_padded[0]
            for i in range(self.params.scale_levels):
                r_k = r_k - self.upsample[i](self.apply_A[i](gamma_k[i]))
            # Get residual coherences
            x_k = [self.apply_B[i](self.downsample[i](r_k)) for i in range(self.params.scale_levels)]
            gamma_k = [self.soft_threshold[i](gamma_k[i] + x_k[i]) for i in range(self.params.scale_levels)]
        output_all = [self.apply_C[i](gamma_k[i]) for i in range(self.params.scale_levels)]
        output_cropped = self.upsample[0](torch.masked_select(output_all[0], valids_batched[0]))
        for i in range(1, self.params.scale_levels):
            output_cropped = output_cropped + self.upsample[i](torch.masked_select(output_all[i], valids_batched[i]))
        output_cropped = output_cropped.reshape(I.shape[0], self.params.num_supports, *I.shape[1:])
        # if self.return_all:
        #     return output_cropped
        output = output_cropped.mean(dim=1, keepdim=False)
        return output
