import torch 
import numpy as np
from PIL import Image
import sys
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, "/home/jayaram/sandbox-image/flownet2-pytorch")

from flownet2_pytorch_official.flownet2_pytorch.models import FlowNet2
from utils.frame_utils import read_gen, flow_to_image

class Args():
  fp16 = False
  rgb_max = 255.

class FlowNet2Utils():
    def __init__(self):
        args = Args()
        self.net = FlowNet2(args).cuda()
        #dict = torch.load("/home/jayaram/mount_ada_node/deep_mpcvs/DeepMPCVS/FlowNet2_checkpoint.pth.tar")
        dict = torch.load("/home2/jayaram.reddy/deep_mpcvs/DeepMPCVS/FlowNet2_checkpoint.pth.tar")
        self.net.load_state_dict(dict["state_dict"])

    def flow_calculate(self, img1, img2):
        images = [img1, img2]
        images = np.array(images).transpose(3, 0, 1, 2)
        im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda()
        result = self.net(im).squeeze()
        data = result.data.cpu().numpy().transpose(1, 2, 0)
        return data
    
    def writeFlow(self, name, flow):
        f = open(name, 'wb')
        f.write('PIEH'.encode('utf-8'))
        np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
        flow = flow.astype(np.float32)
        flow.tofile(f)
        f.flush()
        f.close()
    
    def save_flow_with_image(self, folder):
        img_source_path = folder + "/initial_image.png"
        img_goal_path = folder + "/desired_image.png"
        img_src = read_gen(img_source_path)
        img_goal = read_gen(img_goal_path)
        f12 = self.flow_calculate(img_src, img_goal)
        self.writeFlow(folder + "/flow.flo", f12)
        flow_image = flow_to_image(f12)
        im = Image.fromarray(flow_image)
        im.save(folder + "/flow.png")
