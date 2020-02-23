import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math

def main():
    # Read image
    img = plt.imread('img_example_lr.png')
    x = plt.imshow(list(img))
    plt.show()

    # Scale factor
    ratio = 1.5
    # 'a' is selected because we have only 7 EQNs and 8 unknowns (including 'a')
    a = -1/2 # -0.5 is chosen from a PAPER; but, -0.75 might also be okay
    dst = resizeImage(img, ratio, a)
    print('Completed!')

    # plot Input vs Output
    imageDict = {}
    imageDict["input"] = img
    imageDict["output"] = dst
    plotHelper(imageDict, 2, 1)

# ...................................................

# Linear Algebra functions
def zeroMatrix(w, h):
    return [[0 for i in range(w)] for j in range(h)]

'''
    from Keys' paper:
    * h = sampling increment
    * u = mask/convolution-matrix
    * xk = interpolation-node 
'''

#Plot a dictionary of figures: 
# https://stackoverflow.com/questions/11159436/multiple-figures-in-a-single-window
def plotHelper(figures, nrows = 2, ncols=1):
    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind,title in enumerate(figures):
        axeslist.ravel()[ind].imshow(figures[title], cmap=plt.gray())
        axeslist.ravel()[ind].set_title(title)
    plt.tight_layout() # optional
    plt.show()

# Mask (convolution matrix) which is piece-wise
'''
"interpolation kernels can be effectively used to create new interpolation algorithms. 
The cubic convolution algorithm is derived from a set of conditions..."
'''
def convMask(s,a=-1/2):
    # not in [-2,2]
    if not 0 <= s <= 2:
        return 0
    # [-1,1]
    elif (abs(s) >=0) and (abs(s) <=1):
        return (a+2)*(abs(s)**3)-(a+3)*(abs(s)**2)+1
    # (1,2], [-2,-1)
    else: #(abs(s) > 1) & (abs(s) <= 2)
        return a*(abs(s)**3)-(5*a)*(abs(s)**2)+(8*a)*abs(s)-4*a


# Bicubic operation
def resizeImage(img, ratio=2, a=-1/2):
    #Get image size
    H,W,C = img.shape
    print("\n\ninput-image dimensions:", img.shape)

    ''' each original-pixel is used as a centroid of a 4x4 interpolation-box;
        thus, we pad all sides of the image with 2-layers of pixels. This fulills 
        the otherwise incomplete interpolation-box when we use pixels on the
        outer rows/cols of the image (boundary cases)
    '''
    _img = np.pad(img, 2, mode='edge')
    print(_img[0][9])
    # _img = np.ones((H+4,W+4,C))                # buffer
    # # np.array([[[0 for c in range(C)] for i in range(W+4)] for j in range(H+4)]) <= this didn't work
    # _img[2:H+2,2:W+2,:C] = img                  # original image with 2-px-wide border
    # #Pad the first/last two col and row
    # _img[0:2,2:W+2,:C] = img[0:1,:,:C]            # top
    # _img[H+2:H+4,2:W+2,:] = img[H-1:H,:,:]        # bottom
    # _img[2:H+2,W+2:W+4,:] = img[:,W-1:W,:]        # right
    # _img[2:H+2,0:2,:C]=img[:,0:1,:C]            # left
    # #Pad missing corner-points
    # _img[0:2,0:2,:C] = img[0,0,:C]                # top-left
    # _img[0:2,W+2:W+4,:C] = img[0,W-1,:C]          # top-right
    # _img[H+2:H+4,0:2,:C] = img[H-1,0,:C]          # bot-left
    # _img[H+2:H+4,W+2:W+4,:C] = img[H-1,W-1,:C]    # bot-right
    img, _img = _img, img                         # swap so we can plot both at the end for comparison
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
                x, y = i*h + 2, j*h + 2
                x1, y1 = 1 + x - math.floor(x), 1 + y - math.floor(y)
                x2, y2 = x - math.floor(x), y - math.floor(y)
                x3, y3 = math.floor(x) + 1 - x, math.floor(y) + 1 - y
                x4, y4 = math.floor(x) + 2 - x, math.floor(y) + 2 - y

                '''@ 1,1
                x = 1*(1/2)+2 = 2.5
                x1 = 1 + 2.5 - 2 = 1.5
                x2 = 0.5
                x3 = 0.5
                x4 = 4-2.5 = 1.5
                '''
                # [ u(x1) u(x2) u(x3) u(x4) ]
                mat_l = np.array([[convMask(x1,a),convMask(x2,a),convMask(x3,a),convMask(x4,a)]])
                # [ u(y1) u(y2) u(y3) u(y4) ]
                mat_r = np.array([[convMask(y1,a)],[convMask(y2,a)],[convMask(y3,a)],[convMask(y4,a)]])
                # matrix
                ''' [ f11 ... f14]
                    [ .   .     .]
                    [ .      .  .]
                    [ f41 ... f44]
                '''
                mat_m = np.ones((4,4))
                mat_m[0] = [img[int(y-y1),int(x-x1),channel],img[int(y-y2),int(x-x1),channel],img[int(y+y3),int(x-x1),channel],img[int(y+y4),int(x-x1),channel]]
                mat_m[1] = [img[int(y-y1),int(x-x2),channel],img[int(y-y2),int(x-x2),channel],img[int(y+y3),int(x-x2),channel],img[int(y+y4),int(x-x2),channel]]
                mat_m[2] = [img[int(y-y1),int(x+x3),channel],img[int(y-y2),int(x+x3),channel],img[int(y+y3),int(x+x3),channel],img[int(y+y4),int(x+x3),channel]]
                mat_m[3] = [img[int(y-y1),int(x+x4),channel],img[int(y-y2),int(x+x4),channel],img[int(y+y3),int(x+x4),channel],img[int(y+y4),int(x+x4),channel]]
                
                # compute cth color-channel value for pixel-(j,i) of output image
                dst[j, i, channel] = np.dot(np.dot(mat_l, mat_m), mat_r)
    return dst

if __name__ == '__main__':
    main()
