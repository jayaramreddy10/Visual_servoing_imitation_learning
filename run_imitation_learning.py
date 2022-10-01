import os
import sys
from PIL import Image
import argparse
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)

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
from hyper_params_imitation_learning import h_params

parser = argparse.ArgumentParser(description='Train Visual servoing with imitation learning')
parser.add_argument('-d', "--dataset_path", help = "path to video dataset", required = True, type = str)
parser.add_argument('-f', "--flow_velocity_pairs_path", help = "path to save (flow,vel) pairs", required = True, type = str)
args = parser.parse_args()

def get_depth_map(img, resize):
    # Load a MiDas model for depth estimation
    #model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    #model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

    midas = torch.hub.load("intel-isl/MiDaS", model_type)

    # Move model to GPU if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)
    midas.eval()

    # Load transforms to resize and normalize the image
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            transform = midas_transforms.dpt_transform
    else:
            transform = midas_transforms.small_transform

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Apply input transforms
    input_batch = transform(img).to(device)

    # Prediction and resize to original resolution
    with torch.no_grad():
            prediction = midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
            ).squeeze()

    depth_map = prediction.cpu().numpy()
    depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    depth_map = (depth_map*255).astype(np.uint8)
    depth_map = cv2.resize(depth_map, resize)
    return depth_map
    #depth_map = cv2.applyColorMap(depth_map , cv2.COLORMAP_MAGMA)

class Dataset:
    def __init__(self):
        self.video_paths = [v for v in os.listdir(args.dataset_path) if '.mp4' in v]

    def extract_frames_from_video(self, path, max_frames=0, resize=(h_params.img_size_y, h_params.img_size_x)):
        print('path : {}'.format(path))
        cap = cv2.VideoCapture(path)
        frames = []
        depth_frames = []
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, resize)
                frame = frame[:, :, [2, 1, 0]]
                frames.append(frame)

                #get depth of frame
                depth_frame = get_depth_map(frame, resize)
                depth_frames.append(depth_frame)

                if len(frames) == max_frames:
                    break
        finally:
            cap.release()
        return np.array(frames), np.array(depth_frames)

    def __len__(self):
        return len(self.video_paths)

    def get_image_velocity_pairs_from_videos(self):
        video_paths = self.video_paths
        #print('video paths : {}'.format(video_paths))
        flow_vel_pairs = []
        #loop thru each video
        for idx, path in enumerate(video_paths):
            #extract frames from video
            #print('video path: {}'.format(type(path)))
            frames, depth_frames = self.extract_frames_from_video(path, h_params.max_frames)
            video_len = frames.shape[0]
            flow_utils = FlowNet2Utils()
            intermat = InteractionMatrix()
            video_name = path.split('.')[0]
            f = open(folder + '/results_imitation_learning' + "/log_" + video_name + ".txt", "w+")
            for i in range(video_len - 1):
                current_frame = frames[i]
                current_depth = depth_frames[i]
                next_frame = frames[i + 1]
                #calculate flow b/w current and next frame
                f12 = None   
                f12 = flow_utils.flow_calculate(current_frame, next_frame)   #has shape (384, 512, 2)
                #print('flow map shape: {}'.format(f12.shape))
                #Get interaction matrices
                #If f12, d1 --> (384,512), then Lsx, Lsy will have shape (384, 512, 6)
                vel ,Lsx, Lsy = intermat.getData(f12, current_depth) 

                #Now L(i) * v(i) = flow(i) = f12  
                #flatten Lsx, Lsy to 2 dimensional matrices
                Lsx = Lsx.reshape((-1, 6))
                Lsy = Lsy.reshape((-1, 6))
                #flatten f12 to one dimensional vector
                f12 = f12.reshape((-1, 2))
                f12 = f12.flatten()

                #L matrix will have rows from Lsx, Lsy alternatively.
                row_Lsx, col_Lsx = np.shape(Lsx)
                row_Lsy, col_Lsy = np.shape(Lsy)
                assert col_Lsx == col_Lsy, 'number of cols should be same'
                L = np.ravel([Lsx, Lsy], order = "F").reshape(col_Lsx, row_Lsx + row_Lsy).T
                
                #L(i) * v(i) = flow(i) = f12 
                vel = np.linalg.inv(L.T @ L) @ L.T @ f12 

                #L = torch.cat((Lsx, Lsy), -1)      #concatenated along depth -- (384, 512, 12)
                #vel = np.random.normal(size=[1,1,6])
                #vel = torch.tensor(vel, dtype = torch.float32).cuda()
                #vels = vels.repeat(1, 1, 1, 2)

                flow_vel_pairs.append((f12, vel))
                #save velocity
                f.write("Velocity at frame "  + str(i) + "= " + str(vel) + "\n")
                #save flow maps 

                Lsx = torch.tensor(Lsx, dtype = torch.float32).cuda()
                Lsy = torch.tensor(Lsy, dtype = torch.float32).cuda()
                f12 = torch.tensor(f12, dtype = torch.float32).cuda()

if __name__ == '__main__':
    use_gpu = torch.cuda.is_available()
    print('use_gpu: {}'.format(use_gpu))

    #load train and test dataset
    dataset = Dataset()
    # Create folder for results
    folder = args.flow_velocity_pairs_path
    if not os.path.exists(folder + '/results_imitation_learning'):
        os.makedirs(folder + '/results_imitation_learning')
    dataset.get_image_velocity_pairs_from_videos()

    """
    train_data_loader = data_utils.DataLoader(
        train_dataset, batch_size=hparams.batch_size, shuffle=True,
        num_workers=hparams.num_workers)

    test_data_loader = data_utils.DataLoader(
        test_dataset, batch_size=hparams.batch_size,
        num_workers=4)

    device = torch.device("cuda" if use_gpu else "cpu")

     # Model
    model = Wav2Lip().to(device)

    print('total trainable params {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=hparams.initial_learning_rate, betas=(0.5, 0.999))

    # Train!
    train(device, model, train_data_loader, test_data_loader, optimizer, n_epochs=hparams.nepochs)
    """
#Run script:
#/home/jayaram/mount_ada_node/deep_mpcvs/DeepMPCVS/test_image 0 0 0 0 0 0 1 1 1 0
#/home2/jayaram.reddy/deep_mpcvs/DeepMPCVS/test_image 0 0 0 0 0 0 1 1 1 0

#Line