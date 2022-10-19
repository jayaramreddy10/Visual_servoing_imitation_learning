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
from tqdm import tqdm
import pandas as pd

import torch
from torch import nn
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
from torch.utils import data as data_utils
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms, datasets, models

from model import VisualServoingLSTM
from interactionmatrix import InteractionMatrix
import habitatenv as hs
import imageio
from PIL import Image
from os import listdir
from os.path import isfile, join
from hyper_params_imitation_learning import h_params
from behaviour_cloning import Behaviour_cloning_model
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Train Visual servoing with imitation learning')
parser.add_argument('-d', "--dataset_path", help = "path to video dataset", required = True, type = str)
#parser.add_argument('-f', "--flow_velocity_pairs_path", help = "path to save (flow,vel) pairs", required = True, type = str)
args = parser.parse_args()

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

class Video_Dataset(Dataset):   #this class handles a single video
    def __init__(self, img_vel_df, root_dir, is_train, transform = None):
        self.img_vel_df = img_vel_df
        self.root_dir = root_dir  #folder where images are present
        self.is_train = is_train
        self.transform = transform

    def __len__(self):
        return len(self.img_vel_df)

    def __getitem__(self, idx):
        if(torch.is_tensor(idx)):
            idx = idx.to_list()
        img_name = os.path.join(self.root_dir, self.img_vel_df.iloc[idx, 0])
        tmp_img = cv2.imread(img_name)
        img = Image.fromarray(tmp_img)
        if self.transform:
            img = self.transform(img)

        vel = self.img_vel_df.iloc[idx, 1:]
        vel = torch.from_numpy(vel.to_numpy(dtype ='float32'))
        #print('__getitem returned vel: {}'.format(vel.shape))
        return img, vel

def get_depth_map(img, resize):
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

def extract_frames_from_video(video_name, folder, max_frames=0, resize=(h_params.img_size_y, h_params.img_size_x)):
    print('extracting frames from: {}'.format(video_name))
    cap = cv2.VideoCapture(os.path.join(folder, video_name))
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

def get_image_velocity_pairs_from_video(video_name, folder):
    flows = []
    images = []
    velocities = []

    #extract frames from video
    #print('video path: {}'.format(type(path)))
    frames, depth_frames = extract_frames_from_video(video_name, folder, h_params.max_frames)
    video_len = frames.shape[0]
    flow_utils = FlowNet2Utils()
    intermat = InteractionMatrix()
    video_name = video_name.split('.')[0]
    img_vel_pairs = []
    img_vel_df = pd.DataFrame(columns=['image', 'V_x', 'V_y', 'V_z', 'w_x', 'w_y', 'w_z'])
    #create folder to store images extracted from video
    if not os.path.exists(os.path.join(folder, video_name)):
        os.mkdir(os.path.join(folder, video_name))
    f = open(folder + "/log_" + video_name + ".txt", "w+")
    for i in range(video_len - 1):
        current_frame = frames[i]
        cv2.imwrite(folder + '/' + video_name + '/' + str(i+1) + '.png', cv2.cvtColor(current_frame, cv2.COLOR_RGB2BGR))    
        images.append(current_frame)
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
        velocities.append(vel)

        #L = torch.cat((Lsx, Lsy), -1)      #concatenated along depth -- (384, 512, 12)
        #vel = np.random.normal(size=[1,1,6])
        #vel = torch.tensor(vel, dtype = torch.float32).cuda()
        #vels = vels.repeat(1, 1, 1, 2)

        img_vel_pairs.append((current_frame, vel))
        flows.append(f12)
        #save velocity
        f.write("Velocity at frame "  + str(i) + "= " + str(vel) + "\n")
        #save flow maps 

        #append row to data frame
        img_vel_df.loc[i] = [str(i + 1) + '.png'] + list(vel)

        Lsx = torch.tensor(Lsx, dtype = torch.float32).cuda()
        Lsy = torch.tensor(Lsy, dtype = torch.float32).cuda()
        f12 = torch.tensor(f12, dtype = torch.float32).cuda()

    return img_vel_pairs, img_vel_df


def augment_brightness(image):    
    choice = np.random.choice(2)
    if choice == 1:
        return image + np.random.randint(1,20)
    return image - np.random.randint(0,30)

def train_model(model, loaders, criterion, optimizer, video_name, device = 'cuda', n_epochs = h_params.n_epochs, save_model_name = 'behaviour_cloning_resnet50'):
    # Directory for saving model
    if not os.path.exists(os.path.join(folder, 'models', video_name)):
        os.mkdir(os.path.join(folder, 'models', video_name))
    
    for current_epoch in range(1, n_epochs+1):
        
        # TRAIN
        train_steps = 30
        pbar = tqdm(desc = "training Epoch number", total = train_steps)
        model.train()
        
        for i in range(train_steps):
            images, velocities = next(iter(loaders["train"]))
            # print(velocities)
            # print(velocities.shape)
            
            # wrap them in Variable
            # if use_gpu:
            #     inputs = Variable(inputs.cuda())
            #     labels = Variable(labels.cuda())
            # else:
            #     inputs, labels = Variable(inputs), Variable(labels)
            #images, velocities = next(train_loader)
            
            images = images.to(device)
            velocities = velocities.to(device)
            
            # set the optimizer to zero gradients
            optimizer.zero_grad()
            
            # pass the inputs through the model
            outputs = model(images)
            # print(outputs)
            # print(outputs.shape)

            # calculate loss
            loss = criterion(outputs.squeeze(1), velocities)
            # backpropagate
            loss.backward()
            # optimize
            optimizer.step()
            
            pbar.set_postfix({'Train Loss': loss.item()})
            pbar.update(1)
        
        pbar.close()
        
        # VALIDATION
        model.eval()
        validation_steps=10
        with torch.no_grad():
            
            val_pbar = tqdm(desc="Validation Epoch number", total = validation_steps)
            
            val_losses = []
            
            for i in range(validation_steps):
                images, velocities = next(iter(loaders["val"]))
                # print(velocities)
                # print(velocities.shape)
                #images, velocities = next(val_loader)
                # move the data to selected device
                images = images.to(device)
                velocities = velocities.to(device)

                # pass the inputs through the model
                outputs = model(images)
                # print(outputs)
                # print(outputs.shape)

                # calculate loss
                loss = criterion(outputs.squeeze(1), velocities)
                val_losses.append(loss.item())
                
                val_pbar.set_postfix({'Val Loss': loss.item()})
                val_pbar.update(1)
            
            mean_val_loss = sum(val_losses)/len(val_losses)
        
        val_pbar.set_postfix({'Avg Val Loss': mean_val_loss})
        val_pbar.close()
        
        # Save model every 2 epochs
        if(current_epoch % 2 == 0):
            torch.save(model.state_dict(), f'test_behaviour_cloning/models/{video_name}/model-{current_epoch}-{save_model_name}.pth')
            scheduler.step()

if __name__ == '__main__':
    use_gpu = torch.cuda.is_available()
    print('use_gpu: {}'.format(use_gpu))

    # folder for storing results (it contains input videos as well)
    folder = os.path.join(args.dataset_path, 'test_behaviour_cloning')
    # if not os.path.exists(folder):
    #     os.makedirs(folder)

    #loop thru each video
    video_paths = [v for v in os.listdir(os.path.join(args.dataset_path, 'test_behaviour_cloning')) if '.mp4' in v]
    #print('video paths : {}'.format(video_paths))

    for idx, path in enumerate(video_paths):
        #load train and test dataset (img, vel) pairs for single video
        #path: video_name.mp4
        img_vel_pairs, img_vel_df = get_image_velocity_pairs_from_video(path, folder)
        video_name = path.split('.')[0]
        img_vel_df.to_csv(folder + '/' + video_name + '.csv', sep='\t', encoding='utf-8')
        # X_train, X_val, y_train, y_val = train_test_split(images, velocities, test_size = 0.25)
        # print('Train data size:',len(X_train))
        # print('Validation data size:',len(X_val))

        train, val = train_test_split(img_vel_df, test_size = 0.2)

        input_size = 224
        # Create Data Generators
        train_trans = transforms.Compose([
            transforms.Resize(input_size),
            # transforms.RandomSizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        valid_trans = transforms.Compose([
            transforms.Resize(input_size),
            # transforms.Scale(256),
            # transforms.CenterCrop(224),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        train_set = Video_Dataset(train, os.path.join(folder, video_name), is_train = True, transform = train_trans)
        val_set = Video_Dataset(val, os.path.join(folder, video_name), is_train = True, transform = valid_trans)

        train_loader = torch.utils.data.DataLoader(train_set, batch_size = h_params.batch_size, shuffle=True, num_workers=4)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size = h_params.batch_size, shuffle=True, num_workers=4)
        #test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

        dataset_sizes = {
            'train': len(train_loader.dataset), 
            'val': len(val_loader.dataset)
        }

        # train_data_loader = data_utils.DataLoader(train_dataset, batch_size=h_params.batch_size, shuffle=True, num_workers=h_params.num_workers)
        # test_data_loader = data_utils.DataLoader(test_dataset, batch_size=h_params.batch_size, num_workers=4)

        device = torch.device("cuda" if use_gpu else "cpu")

        # define model
        use_gpu = torch.cuda.is_available()
        #model = Behaviour_cloning_model().to(device)
        model = models.resnet50(pretrained=True)

        #I recommend training with these layers unfrozen for a couple of epochs after the initial frozen training
        for param in model.parameters():
            param.requires_grad = False

        num_features = model.fc.in_features
        model.fc = torch.nn.Sequential(torch.nn.Linear(num_features, 75), torch.nn.ReLU(), torch.nn.Linear(75, 6)) 
        if use_gpu:
            model = model.cuda()

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = h_params.l_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.5, step_size=2)
        loaders = {'train':train_loader, 'val':val_loader}

        # Train the model
        train_model(model, loaders, criterion, optimizer, video_name, n_epochs = h_params.n_epochs, device = device)
    
#Run script:
#python run_imitation_learning.py -d /home2/jayaram.reddy/deep_mpcvs/DeepMPCVS 
#python run_imitation_learning.py -d /home/jayaram

