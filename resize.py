import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def ALPHA():
    return -0.5

def main():
    # Read image
    inputImg = plt.imread('img_example_lr.png')
    _temp = plt.imshow(list(inputImg))
    plt.show()

    # call resizing function
    scaleFactor = 1.5
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
def convMask(s):
    # not in [-2,2]
    if not 0 <= s <= 2: 
        return 0
    # [-1,1]
    elif 0 <= abs(s) <=1: 
        return (ALPHA()+2) * abs(s)**3 - (ALPHA()+3) * abs(s)**2 + 1
    # (1,2], [-2,-1)
    else: 
        return ALPHA()* abs(s)**3 - 5*ALPHA() * abs(s)**2 + 8*ALPHA() * abs(s)- 4*ALPHA()

# return the part after the decimal-point of a num (i.e. 1.5 -> 0.5)
def decimalPart(num): 
    return int(str(num).split('.')[-1])

# Bicubic Operation
def resizeImage(img, scaleFactor=2):
    #store dim before image is padded
    heightNoPad = img.shape[0]
    widthNoPad = img.shape[1]
    numChannels = img.shape[2]
    print("\n\ninput-image dimensions:", img.shape)

    ''' each original-pixel is used as a centroid of an interpolation-box that stretches outwards in 4 dirs;
        thus, we pad all sides of the image with 2-layers of pixels. This fulills 
        the otherwise incomplete interpolation-box when we use pixels on the
        outer rows/cols of the image (boundary cases)
    '''
    img = np.pad(img, 2, mode='edge') # add 2-layers around img (nearest nighbour)

    # ...............................................................

    # compute image parameters
    newHight, newWidth = int(heightNoPad*scaleFactor), int(widthNoPad*scaleFactor)
    outputImage = np.zeros((newHight, newWidth, 3)) # output buffer
    samplingIncrement = 1/scaleFactor # sampling-increment

    # console output
    print("scale factor: ", scaleFactor)
    print("output-image dimensions will be: ", outputImage.shape)
            
    # draw new image pixel-by-pixel
    for j in range(newHight):
        for i in range(newWidth):
            # Image has 3 color-channels (R,G,B)
            for channel in range(numChannels): 
                x, y = (i*samplingIncrement+2), (j*samplingIncrement+2)
                fMatrix = np.ones((4,4))
                fMatrix[0] = [ # row 1
                            img[int((j*samplingIncrement+2)-(y-y//1+1)),int((i*samplingIncrement+2)-(x-x//1+1)),channel],  
                            img[int((j*samplingIncrement+2)-(y-y//1)),int((i*samplingIncrement+2)-(x-x//1+1)),channel],
                            img[int((j*samplingIncrement+2)+(y//1-y+1)),int((i*samplingIncrement+2)-(x-x//1+1)),channel], 
                            img[int((j*samplingIncrement+2)+(y//1-y+2)),int((i*samplingIncrement+2)-(x-x//1+1)),channel]
                            ]
                fMatrix[1] = [ # row 2
                            img[int((j*samplingIncrement+2)-(y-y//1+1)),int((i*samplingIncrement+2)-(x-x//1)),channel],
                            img[int((j*samplingIncrement+2)-(y-y//1+1)),int((i*samplingIncrement+2)-(x-x//1)),channel],
                            img[int((j*samplingIncrement+2)+(y//1-y+1)),int((i*samplingIncrement+2)-(x-x//1)),channel],
                            img[int((j*samplingIncrement+2)+(y//1-y+2)),int((i*samplingIncrement+2)-(x-x//1)),channel]]
                fMatrix[2] = [ # row 3
                            img[int((j*samplingIncrement+2)-(y-y//1+1)),int((i*samplingIncrement+2)+(x//1-x+1)),channel],
                            img[int((j*samplingIncrement+2)-(y-y//1+1)),int((i*samplingIncrement+2)+(x//1-x+1)),channel],
                            img[int((j*samplingIncrement+2)+(y//1-y+1)),int((i*samplingIncrement+2)+(x//1-x+1)),channel],
                            img[int((j*samplingIncrement+2)+(y//1-y+2)),int((i*samplingIncrement+2)+(x//1-x+1)),channel]]
                fMatrix[3] = [ # row 4
                            img[int((j*samplingIncrement+2)-(y-y//1+1)),int((i*samplingIncrement+2)+(x//1-x+2)),channel],
                            img[int((j*samplingIncrement+2)-(y-y//1+1)),int((i*samplingIncrement+2)+(x//1-x+2)),channel],
                            img[int((j*samplingIncrement+2)+(y//1-y+1)),int((i*samplingIncrement+2)+(x//1-x+2)),channel],
                            img[int((j*samplingIncrement+2)+(y//1-y+2)),int((i*samplingIncrement+2)+(x//1-x+2)),channel]
                            ]  
                # g(x,y) some matrix mult         
                _s1 = np.dot(np.array([[convMask((x-x//1+1)),convMask((x-x//1)),convMask((x//1-x+1)),convMask((x//1-x+2))]]), fMatrix)
                _s2 = np.dot(_s1,np.array([[convMask((y-y//1+1))],[convMask((y-y//1))],[convMask((y//1-y+1))],[convMask((y//1-y+2))]]))
                outputImage[j, i, channel] = _s2
    return outputImage

if __name__ == '__main__':
    main()

'''
Resources
[1] https://chartio.com/resources/tutorials/how-to-save-a-plot-to-a-file-using-matplotlib/
[2] https://stackoverflow.com/questions/11159436/multiple-figures-in-a-single-window
[3] http://www.ncorr.com/download/publications/keysbicubic.pdf
'''
