import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def ALPHA():
    return -0.5

def main():
   

   '''......... EDIT THESE PARAMETERS ...........'''
    inputImg = plt.imread('img_example_lr.png')
    scaleFactor = 2
    # ..........................................


    _temp = plt.imshow(list(inputImg))
    plt.show()

    # call resizing function
    print('starting...')
    dst = resizeImage(inputImg, scaleFactor)

    # plotting everything
    imageDict = {
        "input" : inputImg,
        "resized output (bnw)" : dst
    }
    inIm = plt.imshow(inputImg); plt.show()
    outIm = plt.imshow(dst);     plt.show()
    plotHelper(imageDict, 2, 1); plt.show()

# ...................................................
def zeroMatrix(w, h):
    return [[0 for i in range(w)] for j in range(h)]
'''
    from Keys' paper:
    * h = sampling increment
    * u = mask/convolution-matrix
    * xk = interpolation-node 
'''
#Plot a dictionary of figures [2]: 
def plotHelper(figures, numRows = 2, numCols=1):
    fig, axes = plt.subplots(ncols=numCols, nrows=numRows)
    for i,title in enumerate(figures):
        axes.ravel()[i].imshow(figures[title])
        axes.ravel()[i].set_title(title)
    plt.tight_layout()

# Mask (convolution matrix) piece-wise 
'''
"interpolation kernels can be effectively used to create new interpolation algorithms. 
The cubic convolution algorithm is derived from a set of conditions..."
'''
def convMatrix(s):
    # not in [-2,2]
    if not 0 <= s <= 2: 
        return 0
    # [-1,1]
    elif 0 <= abs(s) <=1: 
        return (ALPHA()+2) * abs(s)**3 - (ALPHA()+3) * abs(s)**2 + 1
    # (1,2], [-2,-1)
    else: 
        return ALPHA() * abs(s)**3 - 5*ALPHA() * abs(s)**2 + 8*ALPHA() * abs(s)- 4*ALPHA()

# return the part after the decimal-point of a num (i.e. 1.5 -> 0.5)
def decimalPart(num): 
    return int(str(num).split('.')[-1])

def pad(img, h, w):
    #  in case when you have odd number
    top_pad = np.floor((h - img.shape[0]) / 2).astype(np.uint16)
    bottom_pad = np.ceil((h - img.shape[0]) / 2).astype(np.uint16)
    right_pad = np.ceil((w - img.shape[1]) / 2).astype(np.uint16)
    left_pad = np.floor((w - img.shape[1]) / 2).astype(np.uint16)
    return np.copy(np.pad(img, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='edge'))

# Bicubic Operation
def resizeImage(img, scaleFactor=2):
    #store dim before image is padded
    heightNoPad = img.shape[0]
    widthNoPad = img.shape[1]
    print("\n\ninput-image dimensions:", img.shape)
    img = np.pad(img,2,mode='edge')

    print("padded dims: ", img.shape)

    ''' each original-pixel is used as a centroid of an interpolation-box that stretches outwards in 4 dirs;
        thus, we pad all sides of the image with 2-layers of pixels. This fulills 
        the otherwise incomplete interpolation-box when we use pixels on the
        outer rows/cols of the image (boundary cases)
    '''
    # ...............................................................
    # compute image parameters
    outputImage = np.zeros((int(heightNoPad*scaleFactor), int(widthNoPad*scaleFactor), 3)) # output buffer
    samplingIncrement = 1/scaleFactor # sampling-increment
    print("scale factor: ", scaleFactor, "\noutput-image dimensions will be: ", outputImage.shape)        
    
    # g(x,y) => interpolate & draw new image pixel-by-pixel
    for j in range(outputImage.shape[0]):
        for i in range(outputImage.shape[1]):
            # Image has 3 color-channels (R,G,B)
            for channel in range(3): 
                x, y = (i*samplingIncrement+2), (j*samplingIncrement+2)
                fMatrix = np.ones((4,4))

                # fMatrix -> 4x4 matrix of "sample values" or the nearest 16 pixels used to compute g(x,y)
                fMatrix[0] = [ # row 1
                            img[ int((j*samplingIncrement+2)-(y-y//1+1)), int((i*samplingIncrement+2)-(x-x//1+1)), channel ],  
                            img[ int((j*samplingIncrement+2)-(y-y//1)),   int((i*samplingIncrement+2)-(x-x//1+1)), channel ],
                            img[ int((j*samplingIncrement+2)+(y//1-y+1)), int((i*samplingIncrement+2)-(x-x//1+1)), channel ], 
                            img [int((j*samplingIncrement+2)+(y//1-y+2)), int((i*samplingIncrement+2)-(x-x//1+1)), channel ]
                            ]
                            # debugging stuff:
                            # if j==58 and i==100:
                            #     import math
                            #     x, y = i * samplingIncrement + 2 , j * samplingIncrement + 2
                            #     x1 = 1 + x - math.floor(x)
                            #     x2 = x - math.floor(x)
                            #     x3 = math.floor(x) + 1 - x
                            #     x4 = math.floor(x) + 2 - x
                            #     y1 = 1 + y - math.floor(y)
                            #     y2 = y - math.floor(y)
                            #     y3 = math.floor(y) + 1 - y
                            #     y4 = math.floor(y) + 2 - y
                            #     print("x = ", x)
                            #     print("x1 = ", x1)
                            #     print("x2 = ", x2)
                            #     print("x3 = ", x3)
                            #     print("x4 = ", x4)
                            #     print("x-x1 = ", x-x1)
                            #     print("x-x2 = ", x-x2)
                            #     print("x+x3 = ", x+x3)
                            #     print("x+x4 = ", x+x4)
                            #     return
                fMatrix[1] = [ # row 2
                            img[ int((j*samplingIncrement+2)-(y-y//1+1)), int((i*samplingIncrement+2)-(x-x//1)), channel],
                            img[ int((j*samplingIncrement+2)-(y-y//1+1)), int((i*samplingIncrement+2)-(x-x//1)), channel],
                            img[ int((j*samplingIncrement+2)+(y//1-y+1)), int((i*samplingIncrement+2)-(x-x//1)), channel],
                            img[ int((j*samplingIncrement+2)+(y//1-y+2)), int((i*samplingIncrement+2)-(x-x//1)), channel]]
                fMatrix[2] = [ # row 3
                            img[ int((j*samplingIncrement+2)-(y-y//1+1)), int((i*samplingIncrement+2)+(x//1-x+1)), channel],
                            img[ int((j*samplingIncrement+2)-(y-y//1+1)), int((i*samplingIncrement+2)+(x//1-x+1)), channel],
                            img[ int((j*samplingIncrement+2)+(y//1-y+1)), int((i*samplingIncrement+2)+(x//1-x+1)), channel],
                            img[ int((j*samplingIncrement+2)+(y//1-y+2)), int((i*samplingIncrement+2)+(x//1-x+1)), channel]]
                fMatrix[3] = [ # row 4
                            img[ int((j*samplingIncrement+2)-(y-y//1+1)), int((i*samplingIncrement+2)+(x//1-x+2)), channel],
                            img[ int((j*samplingIncrement+2)-(y-y//1+1)), int((i*samplingIncrement+2)+(x//1-x+2)), channel],
                            img[ int((j*samplingIncrement+2)+(y//1-y+1)), int((i*samplingIncrement+2)+(x//1-x+2)), channel],
                            img[ int((j*samplingIncrement+2)+(y//1-y+2)), int((i*samplingIncrement+2)+(x//1-x+2)), channel]]  
                # u_x dot fMatrix dot u_y --> where u_x and u_y are interpolation-kernel vectors in the below arguments       
                _s1 = np.dot(np.array([convMatrix((x-x//1+1)),convMatrix((x-x//1)),convMatrix((x//1-x+1)),convMatrix((x//1-x+2))]), fMatrix)
                _s2 = np.dot(_s1,np.array([[convMatrix((y-y//1+1))], [convMatrix((y-y//1))], [convMatrix((y//1-y+1))], [convMatrix((y//1-y+2))]]))
                # the color-channels are computed one at a time --> monochromatic images would therefore require 3x less operations 
                outputImage[j, i, channel] = _s2
    return outputImage

if __name__ == '__main__':
    main()

'''
References
[1] https://chartio.com/resources/tutorials/how-to-save-a-plot-to-a-file-using-matplotlib/
[2] https://stackoverflow.com/questions/11159436/multiple-figures-in-a-single-window
[3] http://www.ncorr.com/download/publications/keysbicubic.pdf
'''
