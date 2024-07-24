import numpy as np
import torch
import torch.nn as nn


def spike_rate(inp):
    # T = inp.shape[1]
    num = inp.unique()
    if len(num) <= 2 and inp.max() <= 1 and inp.min() >= 0:
        spike = True
        spike_rate = (inp.sum() / inp.numel()).item()
    else:
        spike = False
        spike_rate = 1

    return spike, spike_rate


def conv_syops_counter_hook(conv_module, input, output, name="layer"):
    # Can have multiple inputs, getting the first one
    # 입력으로 여러 개의 데이터 샘플이 하나로 묶여진 배치가 들어올 경우, 배치 내의 첫 번째 데이터 샘플을 가져온다.
    #input = input[0]
    spike, rate = spike_rate(input)

    batch_size = input.shape[0]
    output_dims = list(output.shape[2:])

    kernel_dims = list(conv_module.kernel_size)
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_syops = int(np.prod(kernel_dims)) * in_channels * filters_per_channel

    active_elements_count = batch_size * int(np.prod(output_dims))

    overall_conv_syops = conv_per_position_syops * active_elements_count

    bias_syops = 0

    # conv 계층에 편향이 존재할 경우
    if conv_module.bias is not None:
        bias_syops = out_channels * active_elements_count

    overall_syops = overall_conv_syops + bias_syops
    # overall_syops : ANN에서의 연산 횟수

    if spike:
        #print("{}, spike: {}, rate: {}, calculation: {}".format(name, spike, rate, int(overall_syops) * rate))
        return int(overall_syops) * rate
    else:
        #print("{}, spike: {}, rate: {}, calculation: {}".format(name, spike, rate, int(overall_syops)))
        return int(overall_syops)
    '''
    conv_module.__syops__[0] += int(overall_syops)

    
    if spike:
        conv_module.__syops__[1] += int(overall_syops) * rate
    # conv_module.__syops__[1] : SNN에서의 연산 횟수 (ANN에서의 연산 횟수 * 스파이크 발화율)
    else:
        conv_module.__syops__[2] += int(overall_syops)

    conv_module.__syops__[3] += rate * 100
    '''


def bn_syops_counter_hook(module, input, output, name="layer"):
    #input = input[0]
    spike, rate = spike_rate(input)
    batch_syops = np.prod(input.shape)
    if module.affine:
        batch_syops *= 2
    #module.__syops__[0] += int(batch_syops)

    if spike:
        #module.__syops__[1] += int(batch_syops) * rate
        #print("{}, spike: {}, rate: {}, calculation: {}".format(name, spike, rate, int(batch_syops) * rate))
        return int(batch_syops) * rate
    else:
        #module.__syops__[2] += int(batch_syops)
        #print("{}, spike: {}, rate: {}, calculation: {}".format(name, spike, rate, int(batch_syops)))
        return int(batch_syops)

    module.__syops__[3] += rate * 100



def upsample_syops_counter_hook(module, input, output, name="layer"):
    output_size = output[0]
    batch_size = output_size.shape[0]
    output_elements_count = batch_size
    for val in output_size.shape[1:]:
        output_elements_count *= val
    #module.__syops__[0] += int(output_elements_count)

    spike, rate = spike_rate(output[0])

    if spike:
        #module.__syops__[1] += int(output_elements_count) * rate
        #print("{}, spike: {}, rate: {}, calculation: {}".format(name, spike, rate, int(output_elements_count) * rate))
        return int(output_elements_count) * rate
    else:
        #module.__syops__[2] += int(output_elements_count)
        #print("{}, spike: {}, rate: {}, calculation: {}".format(name, spike, rate, int(output_elements_count)))
        return int(output_elements_count)

    module.__syops__[3] += rate * 100


def pool_syops_counter_hook(module, input, output, name="layer"):
    #input = input[0]
    spike, rate = spike_rate(input)
    #module.__syops__[0] += int(np.prod(input.shape))

    if spike:
        #module.__syops__[1] += int(np.prod(input.shape)) * rate
        #print("{}, spike: {}, rate: {}, calculation: {}".format(name, spike, rate, int(np.prod(input.shape)) * rate))
        return int(np.prod(input.shape)) * rate
    else:
        #module.__syops__[2] += int(np.prod(input.shape))
        #print("{}, spike: {}, rate: {}, calculation: {}".format(name, spike, rate, int(np.prod(input.shape))))
        return int(np.prod(input.shape))

    module.__syops__[3] += rate * 100


def IF_syops_counter_hook(module, input, output, name="layer"):
  # active_elements_count = input[0].numel()
  # IF 뉴런에서 발생하는 누적 연산(accumulation)을 구하고 싶은 것이라면, 입력으로 들어오는 데이터의 개수를 세야 할 것이다.
  active_elements_count = int(torch.count_nonzero(input))

  # 입력되는 스파이크의 개수를 구한다.
  spike, rate = spike_rate(input)
  '''
  module.__syops__[0] += int(active_elements_count)
  
  #spike, rate = spike_rate(output[0])
  
  module.__syops__[1] += int(active_elements_count)
  module.__syops__[3] += rate * 100
  '''
  #print("{}, spike: {}, rate: {}, calculation: {}".format(name, spike, rate, int(active_elements_count) * rate))
  return int(active_elements_count)


def LIF_syops_counter_hook(module, input, output):
    active_elements_count = input[0].numel()
    #module.__syops__[0] += int(active_elements_count)

    spike, rate = spike_rate(output[0])
    #module.__syops__[1] += int(active_elements_count)
    #module.__syops__[3] += rate * 100
    return int(active_elements_count)

def Leaky_syops_counter_hook(module, input, name="layer"):
  #print("{}, spike: {}, rate: {}, calculation: {}".format(name, False, None, int(torch.count_nonzero(input))))
  return int(torch.count_nonzero(input))


def silu_flops_counter_hook(module, input, output, name="layer"):
  # active_elements_count = output.numel()
  active_elements_count = input.numel()
  # module.__flops__ += int(active_elements_count)
  #print("{}, spike: {}, rate: {}, calculation: {}".format(name, False, 1, int(active_elements_count) * 5))
  return int(active_elements_count) * 5
