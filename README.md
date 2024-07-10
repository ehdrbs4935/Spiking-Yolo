# Spiking-Yolo
## Reference
* Yolov8 (https://github.com/ultralytics/ultralytics)
* syops-counter (https://github.com/iCGY96/syops-counter)
* snntorch (https://github.com/jeshraghian/snntorch)
* spiking-jelly (https://github.com/fangwei123456/spikingjelly)
  
## How to use
1. Create a directory named **'ultralytics'** in 'site-packages' directory at the python path.
2. Download the all files to the **'ultralytics'** folder.
   
## Updates
### Updates for Spiking layers
  * **Existing layers(class) in the Yolov8 model**
    * **Conv⭐️** : (Conv2d) -> (BatchNorm2d) -> (SiLU)
    * **Bottleneck**
    * **C2f**
    * **SPPF**
    * **Concat**
    * **Upsample**
      
    ※ `/nn/modules/conv.py`: Conv, Concat  
    ※ `/nn/modules/block.py` : Bottleneck, C2f, SPPF
  
  * **Updated layers(class) in the Spiking-Yolo model**
    * **SConv⭐️** : (Conv2d) -> (BatchNorm2d) -> (LIF/IF)
    * **SBottleneck**
    * **SC2f**
    * **SSPPF**
    * **SConv_spike⭐️** : (Spike encoding) -> (Conv2d) -> (BatchNorm2d) -> (LIF/IF)
    * **SBottleneck_spike**
    * **SC2f_spike**
      
    ※ `/nn/modules/conv.py`: SConv, SConv_spike  
    ※ `/nn/modules/block.py` : SBottleneck, SC2f, SSPPF, SBottleneck_spike, SC2f_spike

### Updates for FLOPS calculation
We add the `/nn/modules/calculator.py` file to calculate the number of FLOPS for each layer. 

Each time data is feded into the model, it prints the following items for each layer.
* **Wheter the input data is spikes**
* **Spike firing rate of input data**
* **the number of FLOPS**  
  ※ Reference : Efficient Federated Learning with Spike Neural Networks for Traffic Sign Recognition
(https://arxiv.org/abs/2205.14315)

  * CNN
    ```math
    FLOPs(l) = \begin{cases}
        ks^2 \times M^2_{out} \times C_{in} \times C_{out} & \text{Conv} \\
        C_{in} \times M^2{in} & \text{BN or AP} \\
        N_{in} \times N_{out}
      \end{cases}
    ```
    
  * Spike rate
    ```math
    R_s(l) = \frac {n_l^s} {N_l}
    ```
    
  * SNN
    ```math
    FLOPs(l)_{SNN} = FLOPs(l) \times R_s(l) \times T
    ```
 


