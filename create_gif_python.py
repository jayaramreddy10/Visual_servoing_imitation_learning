import imageio
import os

from numpy import sort 

filenames = [f for f in os.listdir('/home/jayaram/mount_ada_node/deep_mpcvs/DeepMPCVS/test_image/results') if 'rgba' in f]
filenames.sort()

with imageio.get_writer('flow.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread("test_image/results/" + filename)
        writer.append_data(image)
