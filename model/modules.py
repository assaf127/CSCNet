import torch
import numpy as np
from torch import nn
from torch.nn import functional
from collections import namedtuple
from model.utils import conv_power_method, calc_pad_sizes
import random


def chunks(l, n):
    """Yield n number of striped chunks from l."""
    for i in range(0, n):
        yield l[i::n]


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
                                         'num_supports_train', 'num_supports_eval'])


class ConvLista_T(nn.Module):
    def __init__(self, params: ListaParams, A=None, B=None, C=None, threshold=1e-2):
        super(ConvLista_T, self).__init__()
        if A is None:
            A = []
            for i in range(params.scale_levels):
                cur_A = torch.randn(params.num_filters[i], 1, params.kernel_size, params.kernel_size)
                l = conv_power_method(cur_A, [512, 512], num_iters=200, stride=params.stride)
                cur_A /= torch.sqrt(l)
                A.append(cur_A)
        if B is None:
            B = [torch.clone(A[i]) for i in range(params.scale_levels)]
        if C is None:
            C = [torch.clone(A[i]) for i in range(params.scale_levels)]

        self.params = params

        apply_A, apply_B, apply_C = [], [], []
        soft_threshold = []
        downsample, upsample = [], []
        for i in range(self.params.scale_levels):
            apply_A.append(nn.ConvTranspose2d(params.num_filters[i], 1, kernel_size=params.kernel_size,
                                                         stride=params.stride, bias=False))
            apply_B.append(nn.Conv2d(1, params.num_filters[i], kernel_size=params.kernel_size,
                                                stride=params.stride, bias=False))
            apply_C.append(nn.ConvTranspose2d(params.num_filters[i], 1, kernel_size=params.kernel_size,
                                                         stride=params.stride, bias=False))
            apply_A[i].weight.data.copy_(A[i])
            apply_B[i].weight.data.copy_(B[i])
            apply_C[i].weight.data.copy_(C[i])
            soft_threshold.append(SoftThreshold(params.num_filters[i], threshold))
            # TODO
            if i == 0:
                downsample.append(nn.Identity())
                #upsample.append(nn.Identity())
            else:
                downsample.append(nn.UpsamplingBilinear2d(scale_factor=1 / 2 ** i))
                #upsample.append(nn.UpsamplingBilinear2d(scale_factor=2 ** i))
        self.apply_A = nn.ModuleList(apply_A)
        self.apply_B = nn.ModuleList(apply_B)
        self.apply_C = nn.ModuleList(apply_C)
        self.soft_threshold = nn.ModuleList(soft_threshold)
        self.downsample = nn.ModuleList(downsample)
        #self.upsample = nn.ModuleList(upsample)

    def _num_supports(self):
        return self.params.num_supports_train if self.training else self.params.num_supports_eval

    def _sample_supports(self, seed=None, num_supports=None):
        if seed is not None:
            random.seed(seed)
        if num_supports is None:
            num_supports = self._num_supports()
        total_options = 1
        total_options = self.params.stride ** (2 * self.params.scale_levels)
        result = random.sample(range(total_options), num_supports)
        for i in range(num_supports):
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

    def _split_images(self, I, supports, get_mask=True):
        if self.params.stride == 1:
            if get_mask:
                return I, [torch.ones_like(image, dtype=torch.bool) for image in I]
            else:
                return I

        num_supports = len(supports)

        left_pads, right_pads, top_pads, bot_pads = [], [], [], []
        for image in I:
            left_pad, right_pad, top_pad, bot_pad = calc_pad_sizes(image, self.params.kernel_size, self.params.stride)
            left_pads.append(left_pad)
            right_pads.append(right_pad)
            top_pads.append(top_pad)
            bot_pads.append(bot_pad)

        I_batched_padded = [torch.zeros(I[i].shape[0] // num_supports,
                                        num_supports,
                                        I[i].shape[1],
                                        top_pads[i] + I[i].shape[2] + bot_pads[i],
                                        left_pads[i] + I[i].shape[3] + right_pads[i]).type_as(I[i]) for i in range(self.params.scale_levels)]
        if get_mask:
            valids_batched = [torch.zeros_like(I_batched_padded[i], dtype=torch.bool) for i in range(self.params.scale_levels)]

        reshaped_images = [I[i].reshape(I[i].shape[0] // num_supports, num_supports, *I[i].shape[1:]) for i in range(self.params.scale_levels)]

        for num, support in enumerate(supports):
            for i in range(self.params.scale_levels):
                row_shift, col_shift = support[i]
                left_pad, right_pad, top_pad, bot_pad = left_pads[i], right_pads[i], top_pads[i], bot_pads[i]
                cur_I = reshaped_images[i][:, num, :, :, :]
                I_padded = functional.pad(cur_I, pad=(
                    left_pad - col_shift, right_pad + col_shift, top_pad - row_shift, bot_pad + row_shift), mode='reflect')
                I_batched_padded[i][:, num, :, :, :] = I_padded
                if get_mask:
                    valids = functional.pad(torch.ones_like(cur_I, dtype=torch.bool), pad=(
                        left_pad - col_shift, right_pad + col_shift, top_pad - row_shift, bot_pad + row_shift),
                                            mode='constant')
                    valids_batched[i][:, num, :, :, :] = valids

        I_batched_padded = [I_batched_padded[i].reshape(-1, *I_batched_padded[i].shape[2:]) for i in range(self.params.scale_levels)]
        if get_mask:
            valids_batched = [valids_batched[i].reshape(-1, *valids_batched[i].shape[2:]) for i in range(self.params.scale_levels)]
            return I_batched_padded, valids_batched
        else:
            return I_batched_padded

    def denoise(self, I, seed=None, supports=None):
        if supports is None:
            supports = self._sample_supports(seed)
        num_supports = len(supports)
        duplicated_image = I[:, None, :, :, :].expand(I.shape[0], num_supports, *I.shape[1:])
        duplicated_image = duplicated_image.reshape(-1, *I.shape[1:])

        scaled_images = [self.downsample[i](duplicated_image) for i in range(self.params.scale_levels)]
        scaled_images_shapes = [scaled_images[i].shape[1:] for i in range(self.params.scale_levels)]
        I_batched_padded, valids_batched = self._split_images(scaled_images, supports, get_mask=True)
        conv_input = [self.apply_B[i](I_batched_padded[i]) for i in range(self.params.scale_levels)]
        gamma_k = [self.soft_threshold[i](conv_input[i]) for i in range(self.params.scale_levels)]

        for k in range(self.params.unfoldings - 1):
            # Get current residual
            r_k = duplicated_image
            for i in range(self.params.scale_levels):
                cur_est = torch.masked_select(self.apply_A[i](gamma_k[i]), valids_batched[i])
                cur_est = cur_est.reshape(-1, *scaled_images_shapes[i])
                #cur_est = self.upsample[i](cur_est)
                cur_est = functional.interpolate(cur_est, size=I.shape[2:], mode='bilinear', align_corners=True)
                r_k = r_k - cur_est
            # Get residual coherences
            scaled_images = [self.downsample[i](r_k) for i in range(self.params.scale_levels)]
            for i in range(self.params.scale_levels):
                I_batched_padded[i] = torch.zeros_like(I_batched_padded[i])#self._split_images(scaled_images, supports, get_mask=False)
                I_batched_padded[i][valids_batched[i]] = scaled_images[i].view(-1)
            x_k = [self.apply_B[i](I_batched_padded[i]) for i in range(self.params.scale_levels)]
            gamma_k = [self.soft_threshold[i](gamma_k[i] + x_k[i]) for i in range(self.params.scale_levels)]
        output_all = [self.apply_C[i](gamma_k[i]) for i in range(self.params.scale_levels)]

        for i in range(self.params.scale_levels):
            cur_est = torch.masked_select(output_all[i], valids_batched[i])
            cur_est = cur_est.reshape(-1, *scaled_images_shapes[i])
            if i==0:
                #output_cropped = self.upsample[i](cur_est)
                output_cropped = functional.interpolate(cur_est, size=I.shape[2:], mode='bilinear', align_corners=True)
            else:
                #output_cropped = output_cropped + self.upsample[i](cur_est)
                output_cropped = output_cropped + functional.interpolate(cur_est, size=I.shape[2:], mode='bilinear', align_corners=True)
        output_cropped = output_cropped.reshape(I.shape[0], num_supports, *I.shape[1:])
        # if self.return_all:
        #     return output_cropped
        output = output_cropped.mean(dim=1, keepdim=False)
        return output

    def forward(self, I, seed=None, all_supports=False):
        if all_supports:
            num_all_supports = self.params.stride ** (2 * self.params.scale_levels)
            all_supports = self._sample_supports(num_supports=num_all_supports)
            supports_list = list(chunks(all_supports, 1 + len(all_supports) // self.params.num_supports_eval))

            output = self.denoise(I, seed=seed, supports=supports_list[0])
            for i in range(1, len(supports_list)):
                output += self.denoise(I, seed=seed, supports=supports_list[i])
            return output / len(supports_list)
        else:
            return self.denoise(I, seed=seed)
