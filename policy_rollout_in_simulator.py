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

from torchvision import models, transforms
from utils.frame_utils import read_gen, flow_to_image
from utils.photo_error import mse_
from calculate_flow import FlowNet2Utils
from imageio import imread
import cv2

import torch
from torch import nn
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)

from interactionmatrix import InteractionMatrix
import habitatenv as hs
import imageio
from PIL import Image
from os import listdir
from os.path import isfile, join

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
    
    #folder: /home/jayaram/mount_ada_node/deep_mpcvs/DeepMPCVS/test_behaviour_cloning
    if os.path.exists(folder+'/results'):
        shutil.rmtree(folder+'/results')
    # Create folder for results
    if not os.path.exists(folder+'/results'):
        os.makedirs(folder+'/results')

    flow_utils = FlowNet2Utils()
    init_state = [x, y, z, w, p, q, r]
    env = hs.HabitatEnv(folder, init_state, depth_type)   #to create habitat Env instance we pass initial state, depth type.

    #load checkpoint (of behaviour cloning)
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    # define model
    video_name = 'fps_changed_van_gogh_scene_sim_video'
    check_point_path = os.path.join('test_behaviour_cloning', 'models', video_name, 'model-20-behaviour_cloning_resnet50.pth')
    checkpoint = torch.load(check_point_path)
    model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features
    model.fc = torch.nn.Sequential(torch.nn.Linear(num_features, 75), torch.nn.ReLU(), torch.nn.Linear(75, 6)) 
    # if use_gpu:
    #     model = model.cuda()
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    # Update new Source Image and Depth
    init_vel = np.array([0,0.5,0,0,0,0]).reshape(1,6)  #give random vel initially
    step = 0
    img_src, d1 = env.example_generate_video(init_vel, step+1, folder)
    input_size = 224

    # apply transform to match input dimensions for resnet50 model
    img_trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(input_size),
        # transforms.RandomSizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_src = img_trans(img_src).unsqueeze(0)
    print('img shape: {}'.format(img_src.shape))

    #rollout policy
    for i in range(300):
        with torch.no_grad():
            #get velocity from loaded model
            vel = model(img_src)
            vel = np.array(vel).reshape(1,6)

        # Update new Source Image and Depth using predicted velocities
        img_src, d1 = env.example_generate_video(vel, step+1, folder)
        img_src = img_trans(img_src).unsqueeze(0)
        step = step + 1
        

if __name__ == '__main__':
    main()

#Run script:
#/home/jayaram/mount_ada_node/deep_mpcvs/DeepMPCVS/test_image 0 0 0 0 0 0 1 1 1 0
#python policy_rollout_in_simulator.py /home2/jayaram.reddy/deep_mpcvs/DeepMPCVS/test_behaviour_cloning 0.4 0.4 0.2 1 0 0 0 1 1 

#Line