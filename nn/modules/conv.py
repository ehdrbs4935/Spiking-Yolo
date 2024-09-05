# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Convolution modules."""

import math
import csv

import numpy as np
import torch
import torch.nn as nn

from spikingjelly.activation_based import neuron, functional, layer, surrogate

import snntorch as snn
from snntorch import spikegen

from ultralytics.nn.modules.calculator import conv_syops_counter_hook, bn_syops_counter_hook, pool_syops_counter_hook, Leaky_syops_counter_hook, silu_flops_counter_hook,IF_syops_counter_hook
from ultralytics.nn.modules.neuron import AdaptiveIFNode, AdaptiveLIFNode

from matplotlib import pyplot as plt
import matplotlib.cm as cm

from ultralytics.nn.modules.utils import spectral_flatness

__all__ = ('Conv', 'SConv', 'Conv2', 'LightConv', 'DWConv', 'DWConvTranspose2d', 'ConvTranspose', 'Focus', 'GhostConv',
           'ChannelAttention', 'SpatialAttention', 'CBAM', 'Concat', 'RepConv', 'SConv_spike', 'SConv_AT')


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation
    #default_act = nn.ReLU()

    def __init__(self, c1, c2, k=1, s=1, calculation=False, FFT=False, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.calculation = calculation
        self.FFT = FFT

    def forward(self, x):
      # 입력 feature map plot
      if self.FFT == True:
        fft_feature_map = np.fft.fft2(x[0].detach(), axes=(1, 2))
        fft_feature_map_shifted = np.fft.fftshift(fft_feature_map)
        fft_amplitude = np.abs(fft_feature_map_shifted)
        fft_amplitude = np.log(fft_amplitude + 1)

        for i in range(x.size(1)):
          if i % 5 == 0:
            fig, axes = plt.subplots(5, 2, figsize=(10, 10), constrained_layout=True)
          axes[i % 5][0].imshow(x[0, i, :, :].detach())
          axes[i % 5][1].hist(fft_amplitude[i].flatten(), bins=30)

          if i % 5 == 4 or (x.size(1) <=5 and i == x.size(1)-1):
            plt.show()

        pass

      y = self.conv(x)
      y2 = self.bn(y)
      z = self.act(y2)

      if self.calculation == True:
        self.calculation_li = []
        # conv 계층 연산 횟수 측정
        conv_syops = conv_syops_counter_hook(self.conv, x, y, "conv_conv")
        # bn 계층 연산 횟수 측정
        bn_syops = bn_syops_counter_hook(self.bn, y, y2, "conv_bn")
        # silu 계층 연산 횟수 측정
        silu_syops = silu_flops_counter_hook(self.act, y2, z, "conv_silu")

        self.calculation_li = [conv_syops, bn_syops, silu_syops]
      else:
        self.calculation_li = [0, 0, 0]

      #feature_maps = z[0,0:8,:,:].detach() # 앞의 8개 채널에 대한 feature map들만 불러온다.
      self.feature_map = []
      #self.feature_map.append(torch.cat([feature_maps[i] for i in range(feature_maps.size(0))], dim=1)) # 3차원 배열을 2차원 배열로 변환한다.
      feature_maps = z[0].detach()
      self.feature_map.append(feature_maps.sum(dim=0) / feature_maps.size(0))
      '''
      # FFT 변환 후, plot
      if self.FFT == True:
        fft_feature_map = np.fft.fft2(z[0].detach(), axes=(1,2))
        fft_feature_map_shifted = np.fft.fftshift(fft_feature_map)
        fft_amplitude = np.abs(fft_feature_map_shifted)
        fft_amplitude = np.log(fft_amplitude + 1)

        for i in range(z.size(1)):
          if i % 5 == 0:
            fig, axes = plt.subplots(5, 2, figsize=(10,10), constrained_layout=True)
          axes[i % 5][0].imshow(z[0,i,:,:].detach())
          axes[i % 5][1].hist(fft_amplitude[i].flatten(), bins=30)

          if i % 5 == 4:
            plt.show()

        pass
      '''
      return z

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class SConv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    def __init__(self, c1, c2, k=1, s=1, calculation=False, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)

        variables = {}

        with open('variables.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                variable, value = row
                variables[variable.strip()] = value.strip()

        self.timestep = int(variables['time-step'])
        neuron_name = variables['neuron']

        if neuron_name == 'IF':
            self.node = neuron.IFNode()
        elif neuron_name == 'LIF':
            self.node = neuron.LIFNode()
        else:
            raise ValueError("Non defined neuron")

        self.calculation = calculation

    def forward(self, x):
      spk_rec = []
      y = self.conv(x)
      y2 = self.bn(y)

      if self.calculation == True:
        self.calculation_li = []
        # conv 계층 연산 횟수 측정
        conv_syops = conv_syops_counter_hook(self.conv, x, y, 'sconv_conv')
        # bn 계층 연산 횟수 측정
        bn_syops = bn_syops_counter_hook(self.bn, y, y2, 'sconv_bn')

        self.calculation_li = [conv_syops, bn_syops]
      else:
        self.calculation_li = [0, 0, 0]
      shape = y2.size()

      for t in range(self.timestep):
        spk = self.node(y2.flatten(1))
        # IF or LIF 계층 연산 횟수 측정
        if self.calculation == True:
          IF_syops = IF_syops_counter_hook(self.node, y2, spk,'sconv_IF')
          if len(self.calculation_li) == 2:
            self.calculation_li.append(IF_syops)
          else:
            self.calculation_li[2] = self.calculation_li[2] + IF_syops

        spk_rec.append(spk)

      self.node.reset()
      #self.node.neuronal_reset(spk)

      spk_output = torch.stack(spk_rec).view(-1, shape[0], shape[1], shape[2], shape[3]).sum(0)

      return spk_output

    def forward_fuse(self, x):
        spk_rec = []
        x = self.conv(x)
        shape = x.size()

        for t in range(self.timestep):
            spk = self.node(x.flatten(1))
            spk_rec.append(spk)

        self.node.reset()

        spk_output = torch.stack(spk_rec).view(-1, shape[0], shape[1], shape[2], shape[3]).sum(0)
        return spk_output #self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0]:i[0] + 1, i[1]:i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__('cv2')
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters."""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes the GhostConv object with input channels, output channels, kernel size, stride, groups and
        activation.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, 'conv'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(in_channels=self.conv1.conv.in_channels,
                              out_channels=self.conv1.conv.out_channels,
                              kernel_size=self.conv1.conv.kernel_size,
                              stride=self.conv1.conv.stride,
                              padding=self.conv1.conv.padding,
                              dilation=self.conv1.conv.dilation,
                              groups=self.conv1.conv.groups,
                              bias=True).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if hasattr(self, 'nm'):
            self.__delattr__('nm')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension
        self.calculation = False

    def forward(self, x):
      self.calculation_li = [0]
      """Forward pass for the YOLOv8 mask Proto module."""
      return torch.cat(x, self.d)


class SConv_spike(nn.Module):
  """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

  def __init__(self, c1, c2, k=1, s=1, calculation=False ,FFT=False, p=None, g=1, d=1, act=True):
    """Initialize Conv layer with given arguments including activation."""
    super().__init__()
    self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
    self.bn = nn.BatchNorm2d(c2)
    self.act = nn.SiLU()

    variables = {}

    with open('variables.csv', 'r') as csvfile:
      reader = csv.reader(csvfile)
      for row in reader:
        variable, value = row
        variables[variable.strip()] = value.strip()

    #self.timestep = int(variables['time-step'])
    self.timestep = 1

    # Spiking 뉴런 추가
    neuron_name = variables['neuron']
    if neuron_name == 'IF':
        self.node = neuron.IFNode()
    elif neuron_name == 'LIF':
        self.node = neuron.LIFNode()
    else:
        raise ValueError("Non defined neuron")

    self.calculation = calculation
    self.FFT = FFT

  def forward(self, x):
    if self.calculation == True:
      self.calculation_li = []

    # 입력 feature map(원본) plot
    '''
    if self.FFT == True:
      fft_feature_map = np.fft.fft2(x[0].detach(), axes=(1, 2))
      fft_feature_map_shifted = np.fft.fftshift(fft_feature_map)
      fft_amplitude = np.abs(fft_feature_map_shifted)
      fft_amplitude = np.log(fft_amplitude + 1)

      for i in range(x.size(1)):
        if i % 5 == 0:
          fig, axes = plt.subplots(5, 2, figsize=(10, 10), constrained_layout=True)
        axes[i % 5][0].imshow(x[0, i, :, :].detach())
        axes[i % 5][1].hist(fft_amplitude[i].flatten(), bins=30)

        if i % 5 == 4 or (x.size(1) <= 5 and i == x.size(1) - 1):
          plt.show()

      pass
    '''

    if self.FFT == True:
        fft_feature_maps = np.fft.fft2(x.detach().cpu(), axes=(2, 3))  # 배치에 포함된 모든 이미지와 모든 채널에 대해서 fft 변환을 수행
        fft_feature_maps_shifted = np.fft.fftshift(fft_feature_maps)
        fft_amplitudes = np.abs(fft_feature_maps_shifted)

        fft_amplitudes = fft_amplitudes.reshape(x.size(0), x.size(1),
                                                x.size(2) * x.size(3))  # 각 채널의 2차원 형태의 feature map 값을 1차원 형태로 합침.
        flatness_arr = spectral_flatness(fft_amplitudes)
        sort_arr = np.sort(flatness_arr)
        threshold_arr = sort_arr[:, sort_arr.shape[1] // 2]
        threshold_arr = threshold_arr.reshape(len(threshold_arr), 1)
        top_idx_arr = np.where(flatness_arr >= threshold_arr)[1]
        top_idx_arr = top_idx_arr.reshape(sort_arr.shape[0], -1)  # 상위 50% 지수
        bottom_idx_arr = np.where(flatness_arr < threshold_arr)[1]
        bottom_idx_arr = bottom_idx_arr.reshape(sort_arr.shape[0], -1)  # 상위 50% 지수

        print("-----------FLATNESS-----------")
        print(flatness_arr)
        print("-----------threholds-----------")
        print(threshold_arr)
        print("-----------Top 50% index -----------")
        print(top_idx_arr)
        print("-----------Bottom 50% index -----------")
        print(bottom_idx_arr)

    spk_rec = []  # record output spikes

    # generate spikes from input data (x)
    spikes = spikegen.rate(x, num_steps=self.timestep)

    # 입력 feature map(스파이크) plot
    '''
    if self.FFT == True:
      fft_feature_map = np.fft.fft2(spikes[0,0,:,:].detach(), axes=(1, 2))
      fft_feature_map_shifted = np.fft.fftshift(fft_feature_map)
      fft_amplitude = np.abs(fft_feature_map_shifted)
      fft_amplitude = np.log(fft_amplitude + 1)

      for i in range(spikes.size(2)):
        if i % 5 == 0:
          fig, axes = plt.subplots(5, 2, figsize=(10, 10), constrained_layout=True)
        axes[i % 5][0].imshow(spikes[0, 0, i, :, :].detach())
        axes[i % 5][1].hist(fft_amplitude[i].flatten(), bins=30)

        if i % 5 == 4 or (x.size(1) <= 5 and i == x.size(1) - 1):
          plt.show()

      pass
    '''
    # 스파이크로 변경하고자 하는 채널 리스트
    '''
    #change_list = [0, 3, 4, 6, 8, 11, 12, 13] # 대비가 명확하지 않은 채널 (정보의 손실이 클 것으로 예상됨.)
    #change_list = [1, 2, 5, 7, 9, 10, 14, 15] # 대비가 명확한 채널 (정보 손실이 크지 않을 것으로 예상됨.)
    #change_list = [0, 2, 3, 4, 6, 8, 9, 11 ]  # 주파수 분포가 고른 채널
    change_list = [1, 5, 7, 10, 12, 13, 14, 15]  # 주파수 분포가 고르지 않은 채널

    for idx in change_list:
      x[0,idx,:,:] = spikes[0,0,idx,:,:]
    '''
    for row in range(bottom_idx_arr.shape[0]):
        for col in range(bottom_idx_arr.shape[1]):
            x[row, col, :, :] = spikes[0, row, col, :, :]
    # 변환한 입력 feature map plot
    '''
    if self.FFT == True:
      fft_feature_map = np.fft.fft2(x[0].detach(), axes=(1, 2))
      fft_feature_map_shifted = np.fft.fftshift(fft_feature_map)
      fft_amplitude = np.abs(fft_feature_map_shifted)
      fft_amplitude = np.log(fft_amplitude + 1)

      for i in range(x.size(1)):
        if i % 5 == 0:
          fig, axes = plt.subplots(5, 2, figsize=(10, 10), constrained_layout=True)
        axes[i % 5][0].imshow(x[0, i, :, :].detach())
        axes[i % 5][1].hist(fft_amplitude[i].flatten(), bins=30)

        if i % 5 == 4 or (x.size(1) <= 5 and i == x.size(1) - 1):
          plt.show()

      pass
    '''

    # z : 원본 입력 feature map에 대한 forward 결과
    y = self.conv(x)
    y2 = self.bn(y)
    z = self.act(y2)

    # spk_output : feature map(스파이크)에 대한 forward 결과
    for t in range(self.timestep):
      #cur_conv = self.conv(spikes[t])
      cur_conv = self.conv(x)
      # spk_conv, mem_conv = self.lif_conv(cur_conv, mem_conv)
      cur_bn = self.bn(cur_conv)
      # spk_bn, mem_bn = self.lif_bn(cur_bn, mem_bn)
      spk_bn = self.node(cur_bn)

      spk_rec.append(spk_bn)  # record spikes

    shape = cur_bn.size()
    self.node.reset()

    spk_output = torch.stack(spk_rec).view(-1, shape[0], shape[1], shape[2], shape[3]).sum(0)


    # 출력 feater map plot
    '''
    if self.FFT == True:
      fft_feature_map = np.fft.fft2(z[0].detach(), axes=(1, 2))
      fft_feature_map_shifted = np.fft.fftshift(fft_feature_map)
      fft_amplitude = np.abs(fft_feature_map_shifted)
      fft_amplitude = np.log(fft_amplitude + 1)

      for i in range(z.size(1)):
        if i % 5 == 0:
          fig, axes = plt.subplots(5, 2, figsize=(10, 10), constrained_layout=True)
        axes[i % 5][0].imshow(z[0, i, :, :].detach())
        axes[i % 5][1].hist(fft_amplitude[i].flatten(), bins=30)

        if i % 5 == 4 or (x.size(1) <= 5 and i == x.size(1) - 1):
          plt.show()

      pass
    '''
    if self.calculation == True:
      # conv 계층 연산 횟수 측정
      conv_syops = conv_syops_counter_hook(self.conv, spikes[t], cur_conv, "sconv_conv")
      # lif_conv(Leaky) 계층 연산 횟수 측정
      # lif_conv_syops = Leaky_syops_counter_hook(self.lif_conv, cur_conv, "sconv_lif_conv")
      # bn 계층 연산 횟수 측정
      bn_syops = bn_syops_counter_hook(self.bn, cur_conv, cur_bn, "sconv_bn")
      # lif_bn(Leaky) 계층 연산 횟수 측정
      node_syops = IF_syops_counter_hook(self.node, cur_bn, "sconv_node")

      if len(self.calculation_li) == 0:
        self.calculation_li = [conv_syops, bn_syops, node_syops]
      else:
        self.calculation_li[0] = self.calculation_li[0] + conv_syops
        self.calculation_li[1] = self.calculation_li[0] + bn_syops
        self.calculation_li[2] = self.calculation_li[0] + node_syops
    else:
      self.calculation_li = [0, 0, 0]

    # feature_maps = z[0,0:8,:,:].detach() # 앞의 8개 채널에 대한 feature map들만 불러온다.
    self.feature_map = []
    # self.feature_map.append(torch.cat([feature_maps[i] for i in range(feature_maps.size(0))], dim=1)) # 3차원 배열을 2차원 배열로 변환한다.
    feature_maps = spk_output[0].detach()
    self.feature_map.append(feature_maps.sum(dim=0) / feature_maps.size(0))
    '''
    # FFT 변환 후, plot
    if self.FFT == True:
      fft_feature_map = np.fft.fft2(z[0].detach(), axes=(1,2))
      fft_feature_map_shifted = np.fft.fftshift(fft_feature_map)
      fft_amplitude = np.abs(fft_feature_map_shifted)
      fft_amplitude = np.log(fft_amplitude + 1)

      for i in range(z.size(1)):
        if i % 5 == 0:
          fig, axes = plt.subplots(5, 2, figsize=(10,10), constrained_layout=True)
        axes[i % 5][0].imshow(z[0,i,:,:].detach())
        axes[i % 5][1].hist(fft_amplitude[i].flatten(), bins=30)

        if i % 5 == 4:
          plt.show()

      pass
    '''
    return spk_output
    '''
    # input spikes during self.timestep
    for t in range(self.timestep):
      cur_conv = self.conv(spikes[t])
      # spk_conv, mem_conv = self.lif_conv(cur_conv, mem_conv)
      cur_bn = self.bn(cur_conv)
      # spk_bn, mem_bn = self.lif_bn(cur_bn, mem_bn)
      spk_bn = self.node(cur_bn)

      if self.calculation == True:
        # conv 계층 연산 횟수 측정
        conv_syops = conv_syops_counter_hook(self.conv, spikes[t], cur_conv, "sconv_conv")
        # lif_conv(Leaky) 계층 연산 횟수 측정
        # lif_conv_syops = Leaky_syops_counter_hook(self.lif_conv, cur_conv, "sconv_lif_conv")
        # bn 계층 연산 횟수 측정
        bn_syops = bn_syops_counter_hook(self.bn, cur_conv, cur_bn, "sconv_bn")
        # lif_bn(Leaky) 계층 연산 횟수 측정
        node_syops = IF_syops_counter_hook(self.node, cur_bn, "sconv_node")

        if len(self.calculation_li) == 0:
          self.calculation_li = [conv_syops, bn_syops, node_syops]
        else:
          self.calculation_li[0] = self.calculation_li[0] + conv_syops
          self.calculation_li[1] = self.calculation_li[0] + bn_syops
          self.calculation_li[2] = self.calculation_li[0] + node_syops
      else:
        self.calculation_li = [0, 0, 0]

      spk_rec.append(spk_bn)  # record spikes

    shape = cur_bn.size()
    self.node.reset()

    spk_output = torch.stack(spk_rec).view(-1, shape[0], shape[1], shape[2], shape[3]).sum(0)

    self.feature_map = []
    # self.feature_map.append(torch.cat([feature_maps[i] for i in range(feature_maps.size(0))], dim=1)) # 3차원 배열을 2차원 배열로 변환한다.
    feature_maps = spk_output[0].detach()
    self.feature_map.append(feature_maps.sum(dim=0) / feature_maps.size(0))

    
    # FFT 변환 후, plot
    if self.FFT == True:
      fft_feature_map = np.fft.fft2(spk_output[0].detach(), axes=(1, 2))
      fft_feature_map_shifted = np.fft.fftshift(fft_feature_map)
      fft_amplitude = np.abs(fft_feature_map_shifted)
      fft_amplitude = np.log(fft_amplitude + 1)

      for i in range(spk_output.size(1)):
        if i % 5 == 0:
          fig, axes = plt.subplots(5, 2, figsize=(10, 10), constrained_layout=True)
        axes[i % 5][0].imshow(spk_output[0, i, :, :].detach())
        axes[i % 5][1].hist(fft_amplitude[i].flatten(), bins=30)

        if i % 5 == 4:
          plt.show()

      pass
    
    return spk_output
    '''
class SConv_AT(nn.Module):
  """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

  def __init__(self, c1, c2, k=1, s=1, calculation=False, p=None, g=1, d=1, act=True):
    """Initialize Conv layer with given arguments including activation."""
    super().__init__()
    self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
    self.bn = nn.BatchNorm2d(c2)

    variables = {}

    with open('variables.csv', 'r') as csvfile:
      reader = csv.reader(csvfile)
      for row in reader:
        variable, value = row
        variables[variable.strip()] = value.strip()

    self.timestep = int(variables['time-step'])

    # Spiking 뉴런 추가
    neuron_name = variables['neuron']
    if neuron_name == 'IF':
        self.node = AdaptiveIFNode()
    elif neuron_name == 'LIF':
        self.node = AdaptiveLIFNode()
    else:
        raise ValueError("Non defined neuron")

    self.calculation = calculation

  def forward(self, x):
    if self.calculation == True:
      self.calculation_li = []
    # mem_conv = self.lif_conv.init_leaky()  # reset/init hidden states at t=0
    spk_rec = []  # record output spikes

    # generate spikes from input data (x)
    spikes = spikegen.rate(x, num_steps=self.timestep)

    # input spikes during self.timestep
    for t in range(self.timestep):
      cur_conv = self.conv(spikes[t])
      # spk_conv, mem_conv = self.lif_conv(cur_conv, mem_conv)
      cur_bn = self.bn(cur_conv)
      spk_bn = self.node(cur_bn.flatten(1))

      if self.calculation == True:
        # conv 계층 연산 횟수 측정
        conv_syops = conv_syops_counter_hook(self.conv, spikes[t], cur_conv, "sconv_conv")
        # lif_conv(Leaky) 계층 연산 횟수 측정
        # lif_conv_syops = Leaky_syops_counter_hook(self.lif_conv, cur_conv, "sconv_lif_conv")
        # bn 계층 연산 횟수 측정
        bn_syops = bn_syops_counter_hook(self.bn, cur_conv, cur_bn, "sconv_bn")
        # lif_bn(Leaky) 계층 연산 횟수 측정
        node_syops = IF_syops_counter_hook(self.node, cur_bn, "sconv_node")

        if len(self.calculation_li) == 0:
          self.calculation_li = [conv_syops, bn_syops, node_syops]
        else:
          self.calculation_li[0] = self.calculation_li[0] + conv_syops
          self.calculation_li[1] = self.calculation_li[0] + bn_syops
          self.calculation_li[2] = self.calculation_li[0] + node_syops

      if self.calculation == False:
        self.calculation_li = [0, 0, 0]

      spk_rec.append(spk_bn)  # record spikes

    self.node.reset()

    shape = cur_bn.size()

    spk_output = torch.stack(spk_rec).view(-1, shape[0], shape[1], shape[2], shape[3]).sum(0)

    return spk_output

