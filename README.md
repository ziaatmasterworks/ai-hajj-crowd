# ai-hajj-crowd

## Inference
- Install docker [official docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Check GPU in docker 
  ``` sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi```
- Download [Nvidia-TensorRT image](https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt)
  ``` docker run --gpus all -it --rm -v local_dir:/home/work_station.io/nvidia/tensorrt:21.09-py3 ``` 
- Install dependencies
  - Install [torch+cuda](https://pytorch.org/get-started/previous-versions/)
    ``` pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html```
  - Install torch2trt
  - Gitclone [source](https://github.com/NVIDIA-AI-IOT/torch2trt)
  - cd torch2trt
  - python setup.py install
  - Install opencv-ptyhon
    ``` pip install opencv-python```
  - Solve opencv-python error in docker [source](https://github.com/conda-forge/pygridgen-feedstock/issues/10)

- Git Clone [source](https://github.com/cvlab-stonybrook/DM-Count) CHANGE THIS TO YOUR URL
- cd DM-Count
- Download pre-train weights from [DM-Count](https://github.com/cvlab-stonybrook/DM-Count) to ./PreTrainWeights
- Edit video_path,model_path and output_video_Path in dm-count-trt.py
``` python dm-count-trt.py ```

## Train
- Collect wild crowd images
- Annotate with a single point(x,y) per head in image
    - Annotations format [[x1,y1],[x2,y2]...]
- Store in .npy format
- Divide data in 70/20/10 Train/Test/val each folder should have image and .npy file
     - test.jpg test.npy ...
     - Size of all images should be same (To avoid cuda out of memory error)
 - python train.py --dataset qnrf --data-dir path_to_data_dir --device 0
