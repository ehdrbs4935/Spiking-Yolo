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
        C_{in} \times M^2_{in} & \text{BN or AP} \\
        N_{in} \times N_{out} & \text{FC}
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
 
## Setting layers of model
### YAML file
You can set up the model by writing a YAML file in diriectory `/cfg/models/v8`.  
The YAML file contains the layers that configure the model, as shown below.  

* **yolov8.yaml (YOLOv8n - ANN)**
  ```yaml
  # Ultralytics YOLO 🚀, AGPL-3.0 license
  # YOLOv8 object detection model with P3-P5 outputs. For Usage examples see https://docs.ultralytics.com/tasks/detect
  
  # Parameters
  nc: 80  # number of classes
  scales: # model compound scaling constants, i.e. 'model=yolov8n.yaml' will call yolov8.yaml with scale 'n'
    # [depth, width, max_channels]
    n: [0.33, 0.25, 1024]  # YOLOv8n summary: 225 layers,  3157200 parameters,  3157184 gradients,   8.9 GFLOPs
    s: [0.33, 0.50, 1024]  # YOLOv8s summary: 225 layers, 11166560 parameters, 11166544 gradients,  28.8 GFLOPs
    m: [0.67, 0.75, 768]   # YOLOv8m summary: 295 layers, 25902640 parameters, 25902624 gradients,  79.3 GFLOPs
    l: [1.00, 1.00, 512]   # YOLOv8l summary: 365 layers, 43691520 parameters, 43691504 gradients, 165.7 GFLOPs
    x: [1.00, 1.25, 512]   # YOLOv8x summary: 365 layers, 68229648 parameters, 68229632 gradients, 258.5 GFLOPs
  
  # YOLOv8.0n backbone
  backbone:
    # [from, repeats, module, args]
    - [-1, 1, Conv, [64, 3, 2]]  # 0-P1/2
    - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
    - [-1, 3, C2f, [128, True]]
    - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8
    - [-1, 6, C2f, [256, True]]
    - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16
    - [-1, 6, C2f, [512, True]]
    - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32
    - [-1, 3, C2f, [1024, True]]
    - [-1, 1, SPPF, [1024, 5]]  # 9
  
  # YOLOv8.0n head
  head:
    - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
    - [[-1, 6], 1, Concat, [1]]  # cat backbone P4
    - [-1, 3, C2f, [512]]  # 12
  
    - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
    - [[-1, 4], 1, Concat, [1]]  # cat backbone P3
    - [-1, 3, C2f, [256]]  # 15 (P3/8-small)
  
    - [-1, 1, Conv, [256, 3, 2]] # 16
    - [[-1, 12], 1, Concat, [1]]  # cat head P4
    - [-1, 3, C2f, [512]]  # 18 (P4/16-medium)
  
    - [-1, 1, Conv, [512, 3, 2]] # 19
    - [[-1, 9], 1, Concat, [1]]  # cat head P5
    - [-1, 3, C2f, [1024]]  # 21 (P5/32-large)
  
    - [[15, 18, 21], 1, Detect, [nc]]  # Detect(P3, P4, P5)
  ```
  
* **yolov8_12.yaml (SConv_spike, SC2f_spike)**
  ```yaml
  # Ultralytics YOLO 🚀, AGPL-3.0 license
  # YOLOv8 object detection model with P3-P5 outputs. For Usage examples see https://docs.ultralytics.com/tasks/detect
  
  # Parameters
  nc: 80  # number of classes
  scales: # model compound scaling constants, i.e. 'model=yolov8n.yaml' will call yolov8.yaml with scale 'n'
    # [depth, width, max_channels]
    n: [0.33, 0.25, 1024]  # YOLOv8n summary: 225 layers,  3157200 parameters,  3157184 gradients,   8.9 GFLOPs
    s: [0.33, 0.50, 1024]  # YOLOv8s summary: 225 layers, 11166560 parameters, 11166544 gradients,  28.8 GFLOPs
    m: [0.67, 0.75, 768]   # YOLOv8m summary: 295 layers, 25902640 parameters, 25902624 gradients,  79.3 GFLOPs
    l: [1.00, 1.00, 512]   # YOLOv8l summary: 365 layers, 43691520 parameters, 43691504 gradients, 165.7 GFLOPs
    x: [1.00, 1.25, 512]   # YOLOv8x summary: 365 layers, 68229648 parameters, 68229632 gradients, 258.5 GFLOPs
  
  # YOLOv8.0n backbone
  backbone:
    # [from, repeats, module, args]
    - [-1, 1, Conv, [64, 3, 2, True]]  # 0-P1/2
    - [-1, 1, Conv, [128, 3, 2]]  # 1-P2/4
    - [-1, 3, C2f, [128, True, True]]
    - [-1, 1, Conv, [256, 3, 2]]  # 3-P3/8
    - [-1, 6, C2f, [256, True]]
    - [-1, 1, Conv, [512, 3, 2]]  # 5-P4/16
    - [-1, 6, C2f, [512, True]]
    - [-1, 1, Conv, [1024, 3, 2]]  # 7-P5/32
    - [-1, 3, C2f, [1024, True]]
    - [-1, 1, SPPF, [1024, 5, True]]  # 9
  
  # YOLOv8.0n head
  head:
    - [-1, 1, Upsample, [None, 2, 'nearest', True]]
    - [[-1, 6], 1, Concat, [1]]  # cat backbone P4
    - [-1, 3, C2f, [512]]  # 12
  
    - [-1, 1, Upsample, [None, 2, 'nearest']]
    - [[-1, 4], 1, Concat, [1]]  # cat backbone P3
    - [-1, 3, C2f, [256]]  # 15 (P3/8-small)
  
    - [-1, 1, SConv_spike, [256, 3, 2, True]]
    - [[-1, 12], 1, Concat, [1]]  # cat head P4
    - [-1, 3, SC2f_spike, [512, [None, [0, None], None]]]  # 18 (P4/16-medium)
  
    - [-1, 1, SConv_spike, [512, 3, 2]]
    - [[-1, 9], 1, Concat, [1]]  # cat head P5
    - [-1, 3, SC2f_spike, [1024,[None, [0, 1], 1], False, True]]  # 21 (P5/32-large)
  
    - [[15, 18, 21], 1, Detect, [nc]]  # Detect(P3, P4, P5)
  ```

  **<"args" of the modules>**
    * **Conv**
      * **args[0]** : the number of output channels
      * **args[1]** : kernel size (Default=1)
      * **args[2]** : stride size (Default=1)
      * **args[3]** : calculation (Default=False) : 'Whether or not to count FLOPS at that layer'
        
    * **C2f**
      * **args[0]** : the number of output channels
      * **args[1]** : shortcut (Default=False)
      * **args[2]** : calculation (Default=False)
        
    * **SPPF**
      * **args[0]** : the number of output channels
      * **args[1]** : kernel size of 'Maxpool2d' layer (Default=5)
      * **args[2]** : calculation (Default=False)
        
    * **Upsample**
      * **args[0]** : size (Default=None)
      * **args[1]** : scale factor (Default=None)
      * **args[2]** : mode (Default='nearest')
      * **args[3]** : calculation (Default=False)
        
    * **SConv_spike**
      * **args[0]** : the number of output channels
      * **args[1]** : kernel size (Default=1)
      * **args[2]** : stride size (Default=1)
      * **args[3]** : calculation (Default=False)
        
    * **SC2f_spike**
      * **args[0]** : the number of output channels
      * **args[1]⭐️** : the list storing the numbers of **'Conv'** modules to be converted into spike layers
        * The list inside the args[1] stores the numbers of **'Conv'** layers in the **'SBottleneck_spike'** module.
        * If you want to convert **'Conv'** module to **'SConv_spike'** module, write the number of module.
        * Otherwise, write **'None'**.
      * **args[2]** : shortcut (Default=False)
      * **args[3]** : calculation (Default=False)
