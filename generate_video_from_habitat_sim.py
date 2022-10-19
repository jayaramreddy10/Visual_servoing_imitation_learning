import os
import sys
from PIL import Image
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)
import shutil


from utils.frame_utils import read_gen, flow_to_image
from utils.photo_error import mse_
from calculate_flow import FlowNet2Utils
from imageio import imread
import cv2

import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)

from model import VisualServoingLSTM
from interactionmatrix import InteractionMatrix
import habitatenv as hs
import imageio
from PIL import Image
from os import listdir
from os.path import isfile, join

def read_img(file_name):
    im = imread(file_name)
    if im.shape[2] > 3:
        im =  im[:,:,:3]
        #reshape image into (384,512,3) as flownet takes image with specified dimension
    print(im.size)
    im = Image.fromarray(im).resize((384,512))
    print(im.size)
    return im

def main():
    #specify test data (initial , depth, target image) and initial location,orientation of robot in env.
    folder = sys.argv[1]
    x = float(sys.argv[2])
    y = float(sys.argv[3])
    z = float(sys.argv[4])
    w = float(sys.argv[5])
    p = float(sys.argv[6])
    q = float(sys.argv[7])
    r = float(sys.argv[8])
    vel_init = int(sys.argv[9])
    depth_type = int(sys.argv[10])

    ITERS = 100

    if vel_init == 1:
        vel_init_type = 'RANDOM'
    else:
        vel_init_type = 'IBVS'

    if depth_type == 1:
        depth_type = 'TRUE'
    else:
        depth_type = 'FLOW'
    
    if os.path.exists(folder+'/results'):
        shutil.rmtree(folder+'/results')
    # Create folder for results
    if not os.path.exists(folder+'/results'):
        os.makedirs(folder+'/results')
    

    flow_utils = FlowNet2Utils()
    init_state = [x, y, z, w, p, q, r]
    env = hs.HabitatEnv(folder, init_state, depth_type)   #to create habitat Env instance we pass initial state, depth type.
    img_source_path = folder + "/initial_image.png"
    img_src = read_gen(img_source_path)
    #reshape image into (384,512,3) as mse fn requires img_src, img_goal to be of same dim
    img_src = cv2.resize(img_src, (512, 384))
                # vel = np.array(size=[1,1,6])
    vel = np.array([0,0.5,0,0,0,0]).reshape(1,6)
    # vel = torch.tensor(vel, dtype = torch.float32).cuda()

    step = 0
    for i in range(600):
        # Update new Source Image and Depth using predicted velocities
        img_src, d1 = env.example_generate_video(vel, step+1, folder)
        step = step + 1
        

if __name__ == '__main__':
    main()

#Run script:
#/home/jayaram/mount_ada_node/deep_mpcvs/DeepMPCVS/test_image 0 0 0 0 0 0 1 1 1 0
#python run.py /home2/jayaram.reddy/deep_mpcvs/DeepMPCVS/test_image 0.4 0.4 0.2 1 0 0 0 1 1 

#Line