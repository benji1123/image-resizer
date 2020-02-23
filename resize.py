import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def main():
    # Read image
    inputImg = plt.imread('img_example_lr.png')
    x = plt.imshow(list(inputImg))
    plt.show()
    
    alpha = -0.5 # -0.5 is chosen from a PAPER; but, -0.75 might be okay
    scaleFactor = 1.5

    # call resizing function
    dst = resizeImage(inputImg, scaleFactor, alpha)
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
def convMask(s,a=-1/2):
    # not in [-2,2]
    if not 0 <= s <= 2: 
        return 0
    # [-1,1]
    elif 0 <= abs(s) <=1: 
        return (a+2)*(abs(s)**3)-(a+3)*(abs(s)**2)+1
    # (1,2], [-2,-1)
    else: 
        return a*(abs(s)**3)-(5*a)*(abs(s)**2)+(8*a)*abs(s)-4*a

def decimalPart(num): return int(str(num).split('.')[-1])


# Bicubic operation
def resizeImage(img, ratio=2, a=-0.5):
    #Get image size
    H,W,C = img.shape
    print("\n\ninput-image dimensions:", img.shape)

    ''' each original-pixel is used as a centroid of an interpolation-box that goes outwards in 4 dirs;
        thus, we pad all sides of the image with 2-layers of pixels. This fulills 
        the otherwise incomplete interpolation-box when we use pixels on the
        outer rows/cols of the image (boundary cases)
    '''
    img = np.pad(img, 2, mode='edge')

    # ...............................................................

    dH, dW = int((H*ratio)//1), int((W*ratio)//1)
    dst = np.zeros((dH, dW, 3)) # output buffer
    print("output-image dimensions will be: ", dst.shape)
    h = 1/ratio # sampling-increment
            
    # draw output-image px-by-px (one color-channel at a time)
    for j in range(dH):
        for i in range(dW):
            # Image has 3 color-channels (R,G,B)
            for channel in range(C): 
                x, y = (i*h+2), (j*h+2)
                x1, y1, x2, y2 = (x-x//1+1), (y-y//1+1), (x-x//1), (y-y//1)   
                x3, y3, x4, y4 = (x//1-x+1), (y//1-y+1), (x//1-x+2), (y//1-y+2)
                mat_m = np.ones((4,4))
                mat_m[0] = [img[int(y-y1),int(x-x1),channel],img[int(y-y2),int(x-x1),channel],img[int(y+y3),int(x-x1),channel],img[int(y+y4),int(x-x1),channel]]
                mat_m[1] = [img[int(y-y1),int(x-x2),channel],img[int(y-y2),int(x-x2),channel],img[int(y+y3),int(x-x2),channel],img[int(y+y4),int(x-x2),channel]]
                mat_m[2] = [img[int(y-y1),int(x+x3),channel],img[int(y-y2),int(x+x3),channel],img[int(y+y3),int(x+x3),channel],img[int(y+y4),int(x+x3),channel]]
                mat_m[3] = [img[int(y-y1),int(x+x4),channel],img[int(y-y2),int(x+x4),channel],img[int(y+y3),int(x+x4),channel],img[int(y+y4),int(x+x4),channel]]
                _s1 = np.dot(np.array([[convMask(x1,a),convMask(x2,a),convMask(x3,a),convMask(x4,a)]]), mat_m)
                dst[j, i, channel] = np.dot(_s1,np.array([[convMask(y1,a)],[convMask(y2,a)],[convMask(y3,a)],[convMask(y4,a)]]))
    return dst

if __name__ == '__main__':
    main()

'''
Resources
[1] https://chartio.com/resources/tutorials/how-to-save-a-plot-to-a-file-using-matplotlib/
[2] https://stackoverflow.com/questions/11159436/multiple-figures-in-a-single-window
[3] http://www.ncorr.com/download/publications/keysbicubic.pdf
'''
