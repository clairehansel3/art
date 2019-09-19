import PIL.Image
import numpy as np
import os
import tempfile

def save_image(array, path):
    array2 = np.flip((255 * (array / np.max(array))).astype(np.uint8), axis=0)
    image = PIL.Image.fromarray(array2, mode='L')
    with open(path, 'wb') as f:
        image.save(f)

def save_video(array, path):
    if os.path.isfile(path):
        os.remove(path)
    with tempfile.TemporaryDirectory() as tmpdirname:
        for i in range(array.shape[0]):
            save_image(array[i], tmpdirname + '/image_{}.png'.format(i))
        os.system('ffmpeg -framerate 10 -i \'' + tmpdirname + '/image_%d.png\' -c:v libx264 -profile:v high -crf 30 -pix_fmt yuv420p ' + path)



'''


import cv2
import numpy as np

def save_image(array, path):
    array2 = np.flip((255 * (array / np.max(array))).astype(np.uint8), axis=0)
    cv2.imwrite(path, array2)

def save_video(array, path):
    fps = 0.5
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_writer = cv2.VideoWriter(path, fourcc, 1, array.shape[1:])
    for i in range(array.shape[0]):
        #print(i)
        video_writer.write(np.flip((255 * (array[i] / np.max(array[i]))).astype(np.uint8), axis=0))
    video_writer.release()



import PIL.Image
import numpy as np
import cv2

class Image(object):

    def __init__(self, array):
        self.array = np.flip((255 * (array / np.max(array))).astype(np.uint8), axis=0)

    def render(self, filename):
        image = PIL.Image.fromarray(self.array, mode='L')
        with open(filename, 'wb') as f:
            image.save(f)

class Animation(object):

    def __init__(self, array):
        self.array = array

    def render(self, filename):


        img = numpy.zeros([5,5,3])

img[:,:,0] = numpy.ones([5,5])*64/255.0
img[:,:,1] = numpy.ones([5,5])*128/255.0
img[:,:,2] = numpy.ones([5,5])*192/255.0

cv2.imwrite('color_img.jpg', img)
cv2.imshow("image", img);
cv2.waitKey();

for i in range(len(files)):
    filename=pathIn + files[i]
    #reading each files
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)

    #inserting the frames into an image array
    frame_array.append(img)out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])
out.release()
'''
