## This script has officil method of converting pth file to trt/perofrm inferernce 

import cv2
import time
import torch
import numpy as np
import torchvision
from models import vgg19
from torchvision import transforms
from torch2trt import TRTModule
from torch2trt import torch2trt
xx = torch.randn(1, 3, 512, 512).cuda()
model_trt = TRTModule()
## load pre-train model
model_path = "/home/workStation/pth_weights/model_qnrf.pth"
device = torch.device('cuda')  # device can be "cpu" or "gpu"
model = vgg19()
model.to(device)
model.load_state_dict(torch.load(model_path, device))
model.eval()
model_trt = torch2trt(model, [xx], fp16_mode=True, max_batch_size=1)

# device = torch.device('cuda')
mean = 255.0 * np.array([0.485, 0.456, 0.406])
stdev = 255.0 * np.array([0.229, 0.224, 0.225])

normalize = torchvision.transforms.Normalize(mean, stdev)

def preprocess(camera_value):
    global device, normalize
    x = camera_value
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = transforms.ToTensor()(x).unsqueeze(0)
    x = x.to(device)
    x = x[None, ...]
    return x

def execute(image):

    output = model_trt(preprocess(image))
    ff = output[0].sum()
    return ff

if __name__ == '__main__':

    # Read video
    video_path = '/home/workStation/sample_video.mp4'
    out_put_Video_Path = '/home/workStation/ModelOP.mp4'
    out = cv2.VideoWriter(out_put_Video_Path,cv2.VideoWriter_fourcc(*'MP4V'), 30,(1920,1080))

    cap = cv2.VideoCapture(video_path)

    if (cap.isOpened()== False): 
        print("Error opening video  file")

    count_val = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            # resize frame to 512x512
            start_time = time.time()
            frame_512 = cv2.resize(frame,(512,512)) 
            count,fps_count = execute(frame_512)
            fps_count = 1.0 / (time.time() - start_time)
            frame_op = cv2.putText(frame, 'Threshold Count :'+str(int(count)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)
            frame_op = cv2.putText(frame, 'FPS :'+str(int(fps_count)), (50,150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)
            out.write(frame_op)
            count_val = count_val + 1
            print('shape of imgae',frame.shape)
            print('frame count :',count)
            print('FPS:',fps_count)
            print('No of persons:',count)
        if ret == False:
            cap.release()
            out.release()