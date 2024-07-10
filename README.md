# Spiking-Yolo
## Reference
* Yolov8 (https://github.com/ultralytics/ultralytics)
* syops-counter (https://github.com/iCGY96/syops-counter)
* snntorch ()
* spiking-jelly ()
  
## How to use
1. Create a directory named **'ultralytics'** in 'site-packages' directory at the python path.
2. Download the all files to the **'ultralytics'** folder.
   
## Updates
### Updates for Spiking layers
  * Existing layers(class) in the Yolov8 model
    * **Conv⭐️** : (Conv2d) -> (BatchNorm2d) -> (SiLU)
    * **Bottleneck**
    * **C2f**
    * **SPPF**
    * **Concat**
    * **Upsample**
      
  ※ `/nn/modules/conv.py`: Conv, Concat  
  ※ `/nn/modules/block.py` : Bottleneck, C2f, SPPF
  
  * Updated layers(class) in the Spiking-Yolo model
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
* Wheter the input data is spikes
* Spike firing rate of input data
* the number of FLOPS

