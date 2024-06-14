# Spiking-Yolo
ultralytics library (Spiking Yolo version)

**Yolo v8's spiking version**  

[Reference] : Yolo v8 (https://github.com/ultralytics/ultralytics)

* some layers in Yolo v8 are changed to spiking layers   
  * Conv -> SConv (conv.py)
  * C2f -> SC2f (block.py)
  * Bottleneck -> SBottleneck (block.py)
  * SPPF -> SSPPF (block.py)

* The functions which calculate the number of operations is added.
  * [Reference] : syops-counter (https://github.com/iCGY96/syops-counter)
 
  * **Calculation of spike rate**
    ```python
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
    ```
  * **Convolutional layer's operation**
    ```python
    def conv_syops_counter_hook(conv_module, input, output):
        # Can have multiple inputs, getting the first one
        # 입력으로 여러 개의 데이터 샘플이 하나로 묶여진 배치가 들어올 경우, 배치 내의 첫 번째 데이터 샘플을 가져온다.
        input = input[0]
        spike, rate = spike_rate(input)
    
        print("spike: {}, rate: {}".format(spike, rate))
    
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
            return int(overall_syops) * rate
        else:
            return int(overall_syops)
    ```
  * **Batch Normalization layer's operation**
    ```python
    def bn_syops_counter_hook(module, input, output):
        input = input[0]
        spike, rate = spike_rate(input)
        batch_syops = np.prod(input.shape)
        if module.affine:
            batch_syops *= 2
        if spike:
            return int(batch_syops) * rate
        else:
            return int(batch_syops)
    ```
  * **Upsample layer's operation**
    ```python
    def upsample_syops_counter_hook(module, input, output):
        output_size = output[0]
        batch_size = output_size.shape[0]
        output_elements_count = batch_size
        for val in output_size.shape[1:]:
            output_elements_count *= val
        
        spike, rate = spike_rate(output[0])
    
        if spike:
            return int(output_elements_count) * rate
        else:
            return int(output_elements_count)
    ```
  * **MaxPool layer's operation**
    ```python
    def pool_syops_counter_hook(module, input, output):
        input = input[0]
        spike, rate = spike_rate(input)
    
        if spike:
            return int(np.prod(input.shape)) * rate
        else:
            return int(np.prod(input.shape))
    ```
  * **IF layer's operation**
    ```python
    def IF_syops_counter_hook(module, input, output):
        active_elements_count = input[0].numel()
    
        spike, rate = spike_rate(output[0])
    
        return int(active_elements_count)
    ```
  * **LIF layer's operation**
    ```python
    def LIF_syops_counter_hook(module, input, output):
        active_elements_count = input[0].numel()
    
        spike, rate = spike_rate(output[0])
    
        return int(active_elements_count)
    ```
