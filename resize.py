import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def ALPHA():
    return -0.5

def main():
    # Read image
    inputImg = plt.imread('img_example_lr.png')
    x = plt.imshow(list(inputImg))
    plt.show()
    scaleFactor = 1.5

    # call resizing function
    dst = resizeImage(inputImg, scaleFactor)
    print('Completed!')

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
def plotHelper(figures, nrows = 2, ncols=1):
    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind,title in enumerate(figures):
        axeslist.ravel()[ind].imshow(figures[title])
        axeslist.ravel()[ind].set_title(title)
    plt.tight_layout()

# Mask (convolution matrix) which is piece-wise
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
        return (ALPHA()+2)*(abs(s)**3)-(ALPHA()+3)*(abs(s)**2)+1
    # (1,2], [-2,-1)
    else: 
        return ALPHA()*(abs(s)**3)-(5*ALPHA())*(abs(s)**2)+(8*ALPHA())*abs(s)-4*ALPHA()

def decimalPart(num): 
    return int(str(num).split('.')[-1])

# Bicubic operation
def resizeImage(img, scaleFactor=2):
    #Get image size
    H,W,C = img.shape
    print("\n\ninput-image dimensions:", img.shape)

    ''' each original-pixel is used as a centroid of an interpolation-box that stretches outwards in 4 dirs;
        thus, we pad all sides of the image with 2-layers of pixels. This fulills 
        the otherwise incomplete interpolation-box when we use pixels on the
        outer rows/cols of the image (boundary cases)
    '''
    img = np.pad(img, 2, mode='edge') # add 2-layers around img (nearest nighbour)

    # ...............................................................

    dH, dW = int((H*scaleFactor)//1), int((W*scaleFactor)//1)
    dst = np.zeros((dH, dW, 3)) # output buffer
    print("output-image dimensions will be: ", dst.shape)
    h = 1/scaleFactor # sampling-increment
            
    # draw output-image px-by-px (one color-channel at a time)
    for j in range(dH):
        for i in range(dW):
            # Image has 3 color-channels (R,G,B)
            for channel in range(C): 
                x, y = (i*h+2), (j*h+2)
                mat_m = np.ones((4,4))
                mat_m[0] = [
                            img[int((j*h+2)-(y-y//1+1)),int((i*h+2)-(x-x//1+1)),channel],  
                            img[int((j*h+2)-(y-y//1)),int((i*h+2)-(x-x//1+1)),channel],
                            img[int((j*h+2)+(y//1-y+1)),int((i*h+2)-(x-x//1+1)),channel], 
                            img[int((j*h+2)+(y//1-y+2)),int((i*h+2)-(x-x//1+1)),channel]
                            ]
                mat_m[1] = [
                        img[int((j*h+2)-(y-y//1+1)),int((i*h+2)-(x-x//1)),channel],
                        img[int((j*h+2)-(y-y//1+1)),int((i*h+2)-(x-x//1)),channel],
                        img[int((j*h+2)+(y//1-y+1)),int((i*h+2)-(x-x//1)),channel],
                        img[int((j*h+2)+(y//1-y+2)),int((i*h+2)-(x-x//1)),channel]]
                mat_m[2] = [
                            img[int((j*h+2)-(y-y//1+1)),int((i*h+2)+(x//1-x+1)),channel],
                            img[int((j*h+2)-(y-y//1+1)),int((i*h+2)+(x//1-x+1)),channel],
                            img[int((j*h+2)+(y//1-y+1)),int((i*h+2)+(x//1-x+1)),channel],
                            img[int((j*h+2)+(y//1-y+2)),int((i*h+2)+(x//1-x+1)),channel]]
                mat_m[3] = [
                            img[int((j*h+2)-(y-y//1+1)),int((i*h+2)+(x//1-x+2)),channel],
                            img[int((j*h+2)-(y-y//1+1)),int((i*h+2)+(x//1-x+2)),channel],
                            img[int((j*h+2)+(y//1-y+1)),int((i*h+2)+(x//1-x+2)),channel],
                            img[int((j*h+2)+(y//1-y+2)),int((i*h+2)+(x//1-x+2)),channel]
                            ]               
                _s1 = np.dot(np.array([[convMask((x-x//1+1)),convMask((x-x//1)),convMask((x//1-x+1)),convMask((x//1-x+2))]]), mat_m)
                _s2 = np.dot(_s1,np.array([[convMask((y-y//1+1))],[convMask((y-y//1))],[convMask((y//1-y+1))],[convMask((y//1-y+2))]]))
                dst[j, i, channel] = _s2
    return dst

if __name__ == '__main__':
    main()

'''
Resources
[1] https://chartio.com/resources/tutorials/how-to-save-a-plot-to-a-file-using-matplotlib/
[2] https://stackoverflow.com/questions/11159436/multiple-figures-in-a-single-window
[3] http://www.ncorr.com/download/publications/keysbicubic.pdf
'''
