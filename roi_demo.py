import torch
from models import vgg19
import gdown
from PIL import Image
from torchvision import transforms
import gradio as gr
import cv2
import numpy as np
import scipy
import numpy
import time


pathOut = '/home/alpha/master_works/crowd/final_demo_data/angle_5/model_op_test_pre_train.mp4'
video_path = '/home/alpha/master_works/crowd/DM-Count/source_video/req.mp4'
out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'MP4V'), 30,(2160,720))

## load pre-train model
model_path = "pretrained_models/model_qnrf.pth"
device = torch.device('cuda')  # device can be "cpu" or "gpu"

model = vgg19()
model.to(device)
model.load_state_dict(torch.load(model_path, device))
model.eval()
## load pre-train model



# ## load custom train model

# model_path = '/home/alpha/master_works/crowd/DM-Count/ckpts_old/input-512_wot-0.1_wtv-0.01_reg-10.0_nIter-100_normCood-0/50_ckpt.tar'
# device = torch.device('cuda')  # device can be "cpu" or "gpu"
# deta = torch.load(model_path)
# model = vgg19()
# model.to(device)
# model.load_state_dict(deta['model_state_dict'], device)
# model.eval()

# ## load cusotm train model


def predict(inp):
    inp = transforms.ToTensor()(inp).unsqueeze(0)
    inp = inp.to(device)
    start_time = time.time()
    with torch.set_grad_enabled(False):
        outputs, _ = model(inp)
    count = torch.sum(outputs).item()
    vis_img = outputs[0, 0].cpu().numpy()
    print("FPS: ", 1.0 / (time.time() - start_time))
    print(vis_img.sum())
    heat_map = cv2.resize(vis_img,(1080,720))
    print(heat_map.sum())
    zeta = heat_map.copy()
    # normalize density map values from 0 to 1, then map it to 0-255.
    vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-5)
    vis_img = (vis_img * 255).astype(np.uint8)
    vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
    print(vis_img.sum())
    heat_map = cv2.resize(vis_img,(1080,720))
    return heat_map, int(count) ,zeta

def count_from_polygon(raw_heatMap):
    polygonCoordinate = np.array([[202, 338], [290, 259], [445, 230], [569, 214], [672, 199], [753, 195], [821, 232], [870, 277], [925, 330], [975, 394], [988, 461], [971, 542], [891, 601], [795, 631], [650, 644], [508, 638], [383, 616], [280, 570], [206, 522], [179, 466]])
    mask = cv2.fillPoly(heatMap, [polygonCoordinate], (1,1,1))
    indices = np.where(np.all(mask == (1,1,1), axis=-1))
    coords = zip(indices[0], indices[1])
    count = []
    for xx,yy in coords:
        if raw_heatMap[xx,yy] > 0.013:
            count.append(raw_heatMap[xx,yy])
            ## since heatmap is resized to input frame size we are divising by 64.47. this fraction is input_size/original_model_output suze
    return int(sum(count)/64.47)


def heat_map_roi(img):
    ## takes heatmap resized to original img size and ploygon coordinates
    ## ploygon coordinated
    pts = np.array([[202, 338], [290, 259], [445, 230], [569, 214], [672, 199], [753, 195], [821, 232], [870, 277], [925, 330], [975, 394], [988, 461], [971, 542], [891, 601], [795, 631], [650, 644], [508, 638], [383, 616], [280, 570], [206, 522], [179, 466]])
    x,y = 0,0
    h,w,_ = img.shape
    ## make an black image of input image size
    mask = np.zeros(img.shape[:2], np.uint8)
    ## create a mask image which has black color on roi outside of polygon and has white color in polygon [black and any = black, white and anything = anyhting]
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dst = cv2.bitwise_and(img, img, mask=mask)
    return dst

cap = cv2.VideoCapture(video_path)
if (cap.isOpened()== False): 
    print("Error opening video  file")

# Read until video is completed
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        start_time_ = time.time()
        frame = cv2.resize(frame,(1080,720))
        heatMap,count,raw_heatMap  = predict(frame)
        delta = heatMap.copy() 
        no_ppl_roi = count_from_polygon(raw_heatMap)
        # frame = cv2.putText(frame, 'Person_count_ROI:'+str(int(no_ppl_roi)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)
        heatMap_delta = heat_map_roi(delta)
        v_img = cv2.hconcat([frame,heatMap_delta])
        cv2.rectangle(v_img, (50, 50), (50 + 370, 50 + 58), (0,0,0), -1)
        cv2.putText(v_img, 'Count:'+str(int(no_ppl_roi)), (50,100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2, cv2.LINE_AA)
        cv2.putText(v_img, 'ROI:Hateem/Hijr Ismail', (50,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, cv2.LINE_AA)

        print("FPS_full fps: ", 1.0 / (time.time() - start_time_))
        print('sjape of combined framess',len(v_img))
        out.write(v_img)
        cv2.imshow('Frame',v_img)
        cv2.waitKey(15) 

    if ret == False:
        cv2.destroyAllWindows()
        cap.release()
        out.release()

