import habitat_sim
import habitat_sim.agent
from habitat_sim.utils.common import (
    quat_from_angle_axis,
    quat_from_magnum,
    quat_to_magnum,
)
from habitat_sim.utils.data import ImageExtractor
import matplotlib.pyplot as plt
import cv2
import torch
import numpy as np
import PIL

#to move block of code to left ,shift+tab
def save_sample(sample, depth_map, cnt):
    img = sample["rgba"]
    depth = sample["depth"]
    semantic = sample["semantic"]
    arr = [img, depth, semantic]
    titles = ["rgba", "depth", "semantic"]
    for i, data in enumerate(arr):
        if titles[i] == 'rgba':
                cv2.imwrite(folder + "/results/" + titles[i] + "_" + str(cnt) + ".png", cv2.cvtColor(data, cv2.COLOR_RGBA2BGRA))
                cv2.imwrite(folder + "/results/" + 'depth_map' + "_" + str(cnt) + ".png", depth_map)

def display_sample(sample):
    img = sample["rgba"]
    depth = sample["depth"]
    semantic = sample["semantic"]

    arr = [img, depth, semantic]
    titles = ["rgba", "depth", "semantic"]
    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)

    plt.show()

def get_depth_map(img):
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
        return depth_map
        #depth_map = cv2.applyColorMap(depth_map , cv2.COLORMAP_MAGMA)

#scene_glb = folder + "/habitat-test-scenes/van-gogh-room.glb"
#folder = '/home2/jayaram.reddy/deep_mpcvs/DeepMPCVS/test_image' 
folder = '/home/jayaram/mount_ada_node/deep_mpcvs/DeepMPCVS/test_image' 
scene_glb = folder + "/habitat_test_scenes/van-gogh-room.glb" 

extractor = ImageExtractor(
            scene_glb,
            img_size=(512, 512),
            output=["rgba", "depth", "semantic"],
)

        # Use the list of train outputs instead of the default, which is the full list
        # of outputs (test + train)
extractor.set_mode('train')

        # Index in to the extractor like a normal python list
sample = extractor[0]

        # Or use slicing
samples = extractor[1:39]
for (i, sample) in enumerate(samples):
    img = sample["rgba"]
    print(img.shape)
    depth_map = get_depth_map(img)
    save_sample(sample, depth_map, i)
    #display_sample(sample)
        # Close the extractor so we can instantiate another one later
        # (see close method for detailed explanation)
extractor.close()
