import os
import cv2
import glob
import ffmpeg

image_folder = '/home/jayaram/mount_ada_node/deep_mpcvs/DeepMPCVS/test_image/results'
video_name = 'van_gogh_scene_sim_video.avi'

# images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
# frameSize = (500, 500)

# out = cv2.VideoWriter(video_name,cv2.VideoWriter_fourcc(*'DIVX'), 60, frameSize)

# for filename in glob.glob(image_folder):
#     img = cv2.imread(filename)
#     out.write(img)

# out.release()
#####################################################################################################3
# images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
# images = sorted(images)
# frame = cv2.imread(os.path.join(image_folder, images[0]))
# height, width, layers = frame.shape

# video = cv2.VideoWriter(video_name, 0, 1, (width,height))

# for image in images:
#     video.write(cv2.imread(os.path.join(image_folder, image)))

# cv2.destroyAllWindows()
# video.release()
#####################################################################################################3
# stream = ffmpeg.input(video_name) # video location

# # stream = stream.trim(start = 0, duration=599).filter('setpts', 'PTS-STARTPTS')
# stream = stream.filter('fps', fps=30, round='up')

# stream = ffmpeg.output(stream, 'fps_changed_' + video_name+ '.mp4')

# ffmpeg.run(stream)

#conda install x264=='1!152.20180717' ffmpeg=4.0.2 -c conda-forge
#simply run below command
#ffmpeg -framerate 30 -pattern_type glob -i '*.png' ../../policy_rollout_video_sim.mp4