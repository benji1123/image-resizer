import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import sys, time

'''
    from the paper:
    * h = sampling increment
    * u = mask/convolution-matrix
    * xk = interpolation-node 
'''

# Mask (convolution matrix) which is piece-wise
'''
"interpolation kernels can be effectively used to create new interpolation algorithms. 
The cubic convolution algorithm is derived from a set of conditions..."
'''
def convMask(s,a=-1/2):
    # piece 1
    if (abs(s) >=0) & (abs(s) <=1):
        return (a+2)*(abs(s)**3)-(a+3)*(abs(s)**2)+1
    # piece 2
    elif (abs(s) > 1) & (abs(s) <= 2):
        return a*(abs(s)**3)-(5*a)*(abs(s)**2)+(8*a)*abs(s)-4*a
    # piece 3
    return 0

#Padding
def padding(img,H,W,C):
    zimg = np.zeros((H+4,W+4,C))                # buffer
    zimg[2:H+2,2:W+2,:C] = img                  # original image with 2-px-wide border
    #Pad the first/last two col and row
    zimg[0:2,2:W+2,:C]=img[0:1,:,:C]            # top
    zimg[H+2:H+4,2:W+2,:]=img[H-1:H,:,:]        # bottom
    zimg[2:H+2,W+2:W+4,:]=img[:,W-1:W,:]        # right
    zimg[2:H+2,0:2,:C]=img[:,0:1,:C]            # left
    #Pad missing corner-points
    zimg[0:2,0:2,:C]=img[0,0,:C]                # top-left
    zimg[0:2,W+2:W+4,:C]=img[0,W-1,:C]          # top-right
    zimg[H+2:H+4,0:2,:C]=img[H-1,0,:C]          # bot-left
    zimg[H+2:H+4,W+2:W+4,:C]=img[H-1,W-1,:C]    # bot-right
    
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
def bicubic(img, ratio=2, a=-1/2):
    #Get image size
    H,W,C = img.shape
    print("\n\ninput-image dimensions:", img.shape)

    ''' each original-pixel is used as a centroid of a 4x4 interpolation-box;
        thus, we pad all sides of the image with 2-layers of pixels. This fulills 
        the otherwise incomplete interpolation-box when we use pixels on the
        outer rows/cols of the image (boundary cases)
    '''
    _img = np.zeros((H+4,W+4,C))                # buffer
    _img[2:H+2,2:W+2,:C] = img                  # original image with 2-px-wide border
    #Pad the first/last two col and row
    _img[0:2,2:W+2,:C] = img[0:1,:,:C]            # top
    _img[H+2:H+4,2:W+2,:] = img[H-1:H,:,:]        # bottom
    _img[2:H+2,W+2:W+4,:] = img[:,W-1:W,:]        # right
    _img[2:H+2,0:2,:C]=img[:,0:1,:C]            # left
    #Pad missing corner-points
    _img[0:2,0:2,:C] = img[0,0,:C]                # top-left
    _img[0:2,W+2:W+4,:C] = img[0,W-1,:C]          # top-right
    _img[H+2:H+4,0:2,:C] = img[H-1,0,:C]          # bot-left
    _img[H+2:H+4,W+2:W+4,:C] = img[H-1,W-1,:C]    # bot-right
    # ...............................................................

    dH, dW = math.floor(H*ratio), math.floor(W*ratio)
    dst = np.zeros((dH, dW, 3)) # output buffer
    print("output-image dimensions will be: ", dst.shape)
    h = 1/ratio # 'h' = sampling-increment
    inc = 0     # progress bar

            
    # draw output-image px-by-px (one color-channel at a time)
    for j in range(dH):
        for i in range(dW):
            # Image has 3 color-channels (R,G,B)
            for channel in range(C): 
                x = i * h + 2
                y = j * h + 2
                x1 = 1 + x - math.floor(x)
                x2 = x - math.floor(x)
                x3 = math.floor(x) + 1 - x
                x4 = math.floor(x) + 2 - x
                y1 = 1 + y - math.floor(y)
                y2 = y - math.floor(y)
                y3 = math.floor(y) + 1 - y
                y4 = math.floor(y) + 2 - y

                # [ u(x1) u(x2) u(x3) u(x4) ]
                mat_l = np.matrix([[convMask(x1,a),convMask(x2,a),convMask(x3,a),convMask(x4,a)]])

                # [ u(y1) u(y2) u(y3) u(y4) ]
                mat_r = np.matrix([[convMask(y1,a)],[convMask(y2,a)],[convMask(y3,a)],[convMask(y4,a)]])

                # matrix
                ''' [ f11 ... f14]
                    [ .   .     .]
                    [ .      .  .]
                    [ f41 ... f44]
                '''
                mat_m = np.matrix([
                                    [img[int(y-y1),int(x-x1),channel],img[int(y-y2),int(x-x1),channel],img[int(y+y3),int(x-x1),channel],img[int(y+y4),int(x-x1),channel]],
                                    [img[int(y-y1),int(x-x2),channel],img[int(y-y2),int(x-x2),channel],img[int(y+y3),int(x-x2),channel],img[int(y+y4),int(x-x2),channel]],
                                    [img[int(y-y1),int(x+x3),channel],img[int(y-y2),int(x+x3),channel],img[int(y+y3),int(x+x3),channel],img[int(y+y4),int(x+x3),channel]],
                                    [img[int(y-y1),int(x+x4),channel],img[int(y-y2),int(x+x4),channel],img[int(y+y3),int(x+x4),channel],img[int(y+y4),int(x+x4),channel]]
                                ])
                
                # compute cth color-channel value for pixel-(j,i) of output image
                dst[j, i, channel] = np.dot(np.dot(mat_l, mat_m), mat_r)

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

# 'a' is selected because we have only 7 EQNs and 8 unknowns (including 'a')
a = -1/2 # -0.5 is chosen from a PAPER; but, -0.75 might also be okay

dst = bicubic(img, ratio, a)
print('Completed!')
y = plt.imshow(dst)
plt.show()
