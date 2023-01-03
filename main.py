from matplotlib import pyplot as plt
import cv2
import numpy as np
import math as mt

def fmap1(img, dims):
    nrows, ncols = dims
    out = np.zeros([nrows,ncols])
    for n1 in range(nrows):
        for n2 in range(ncols):
            out[n1][n2] = mt.sqrt(img[n1][n2])         
    return out

def fmap2(img, dims):
    nrows, ncols = dims
    out = np.zeros([nrows,ncols])
    for n1 in range(nrows):
        for n2 in range(ncols):
            out[n1][n2] = img[n1][n2] ** 2
    return out

def fmap3(img, dims):
    nrows, ncols = dims
    out = np.zeros([nrows,ncols])
    for n1 in range(nrows):
        for n2 in range(ncols):
            out[n1][n2] = img[n1][n2]
    return out

def fmap4(img, dims):
    nrows, ncols = dims
    out = np.zeros([nrows,ncols])
    for n1 in range(nrows):
        for n2 in range(ncols):
            if img[n1][n2] <= 0.5:
                out[n1][n2] = mt.sqrt(img[n1][n2] / 2)
            else:
                out[n1][n2] = 1 - mt.sqrt((1-img[n1][n2]) / 2)
    return out
    

def fmap5(img, dims):
    nrows, ncols = dims
    out = np.zeros([nrows,ncols])
    for n1 in range(nrows):
        for n2 in range(ncols):
            if img[n1][n2] <= 0.5:
                out[n1][n2] = 2*img[n1][n2]**2
            else:
                out[n1][n2] = 1-2*(1-img[n1][n2])**2
    return out

def teager(img, dims, m):
    nrows, ncols = dims
    out = np.zeros([nrows,ncols])
    for n1 in range(1,nrows-1):
        for n2 in range(1,ncols-1):
            out[n1][n2] = (img[n1][n2]**(2/m)) * 3
            out[n1][n2] -= ((img[n1+1][n2+1] * img[n1-1][n2-1])**(1/m)) / 2 
            out[n1][n2] -= ((img[n1+1][n2-1] * img[n1-1][n2+1])**(1/m)) / 2 
            out[n1][n2] -= ((img[n1+1][n2] * img[n1-1][n2])**(1/m))
            out[n1][n2] -= ((img[n1][n2+1] * img[n1][n2-1])**(1/m))
            if out[n1][n2] < 0:
                out[n1][n2] = 0
            if out[n1][n2] > 1:
                out[n1][n2] = 1
    return out

def teager_filter(img, k, m, fmap_int = None, fmap_out = None):
    if(fmap_int):
        img = fmap_int(img, img.shape)
    blur = teager(img, img.shape, m)
    img = img + blur * k
    if(fmap_out):
        img = fmap_out(img, img.shape)
    return img

img = cv2.imread('origin.png')
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
gray = gray.astype(float) / 255

out = teager_filter(gray, 0.5, 2)
plt.imsave('no_mapping.png', out, cmap='gray')

out = teager_filter(gray, 0.5, 2, fmap2)
plt.imsave('fmap2.png', out, cmap='gray')

out = teager_filter(gray, 0.5, 2, fmap5)
plt.imsave('fmap5.png', out, cmap='gray')