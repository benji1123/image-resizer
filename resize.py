import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import sys, time

# Mask (convolution matrix)
def convMask(s,a):
    if (abs(s) >=0) & (abs(s) <=1):
        return (a+2)*(abs(s)**3)-(a+3)*(abs(s)**2)+1
    elif (abs(s) > 1) & (abs(s) <= 2):
        return a*(abs(s)**3)-(5*a)*(abs(s)**2)+(8*a)*abs(s)-4*a
    # $s=0$ if not in [0,2]
    return 0

#Padding
def padding(img,H,W,C):
    zimg = np.zeros((H+4,W+4,C))            # image-buffer
    zimg[2:H+2,2:W+2,:C] = img              # original image with 2-pixel-deep border
   
    #Pad the first/last two col and row
    zimg[2:H+2,0:2,:C]=img[:,0:1,:C]            # left
    zimg[H+2:H+4,2:W+2,:]=img[H-1:H,:,:]        # bottom
    zimg[2:H+2,W+2:W+4,:]=img[:,W-1:W,:]        # right
    zimg[0:2,2:W+2,:C]=img[0:1,:,:C]            # top

    #Pad the missing eight points (above 4 ranges miss the corners)
    zimg[0:2,0:2,:C]=img[0,0,:C]                # top-left
    zimg[H+2:H+4,0:2,:C]=img[H-1,0,:C]          # bot-left
    zimg[H+2:H+4,W+2:W+4,:C]=img[H-1,W-1,:C]    # bot-right
    zimg[0:2,W+2:W+4,:C]=img[0,W-1,:C]          # top-right
    return zimg

# https://github.com/yunabe/codelab/blob/master/misc/terminal_progressbar/progress.py
def get_progressbar_str(progress):
    END = 170
    MAX_LEN = 30
    BAR_LEN = int(MAX_LEN * progress)
    return ('Progress:[' + '=' * BAR_LEN +
            ('>' if BAR_LEN < MAX_LEN else '') +
            ' ' * (MAX_LEN - BAR_LEN) +
            '] %.1f%%' % (progress * 100.))

# Bicubic operation
def bicubic(img, ratio, a):
    #Get image size
    H,W,C = img.shape
    print(img.shape)

    img = padding(img,H,W,C)
    print(img.shape)

    #Create new image
    dH = math.floor(H*ratio)
    dW = math.floor(W*ratio)
    dst = np.zeros((dH, dW, 3))

    h = 1/ratio

    print('Start bicubic interpolation')
    print('It will take a little while...')
    inc = 0

    # Image has 3 channels (R,G,B)
    for c in range(C):
        print(' starting: channel ' + str(c) + '...')
        for j in range(dH):
            for i in range(dW):
                # map our points
                x, y = i * h + 2 , j * h + 2

                x1 = 1 + x - math.floor(x)
                x2 = x - math.floor(x)
                x3 = math.floor(x) + 1 - x
                x4 = math.floor(x) + 2 - x

                y1 = 1 + y - math.floor(y)
                y2 = y - math.floor(y)
                y3 = math.floor(y) + 1 - y
                y4 = math.floor(y) + 2 - y

                mat_l = np.matrix([[convMask(x1,a),convMask(x2,a),convMask(x3,a),convMask(x4,a)]])
                mat_m = np.matrix([[img[int(y-y1),int(x-x1),c],img[int(y-y2),int(x-x1),c],img[int(y+y3),int(x-x1),c],img[int(y+y4),int(x-x1),c]],
                                   [img[int(y-y1),int(x-x2),c],img[int(y-y2),int(x-x2),c],img[int(y+y3),int(x-x2),c],img[int(y+y4),int(x-x2),c]],
                                   [img[int(y-y1),int(x+x3),c],img[int(y-y2),int(x+x3),c],img[int(y+y3),int(x+x3),c],img[int(y+y4),int(x+x3),c]],
                                   [img[int(y-y1),int(x+x4),c],img[int(y-y2),int(x+x4),c],img[int(y+y3),int(x+x4),c],img[int(y+y4),int(x+x4),c]]])
                mat_r = np.matrix([[convMask(y1,a)],[convMask(y2,a)],[convMask(y3,a)],[convMask(y4,a)]])
                dst[j, i, c] = np.dot(np.dot(mat_l, mat_m), mat_r)

                # Print progress
                inc = inc + 1
                sys.stderr.write('\r\033[K' + get_progressbar_str(inc/(C*dH*dW)))
                sys.stderr.flush()
    sys.stderr.write('\n')
    sys.stderr.flush()
    return dst

# Read image
img = plt.imread('img_test_lr.png')
x = plt.imshow(img)
plt.show()

# Scale factor
ratio = 2
# Coefficient
a = -1/2

dst = bicubic(img, ratio, a)
print('Completed!')
y = plt.imshow(dst)
plt.show()
# displayImg = Image.fromarray(img, mode='RGB')
# displayImg.show()
