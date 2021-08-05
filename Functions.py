import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans

import pywt
from matplotlib import pyplot as plt

import pysal as ps
from pysal import lib
from esda.moran import Moran
import pandas as pd


from skimage import io, color
from scipy.optimize import linprog

from PIL import Image
import cv2

import random

from sklearn.cluster import KMeans
from python_tsp.exact import solve_tsp_dynamic_programming

import warnings



### ISI Paper implementation

## Moran's I

def Moran_coeff(wp):
    size = wp.shape[0]
    w = ps.lib.weights.lat2W(size,size)

    moran = Moran(wp, w, transformation='r')
    return moran.I


## wavestrap

def wp_boot(wp):
    wp1 = wp.flatten()
    size1 = wp.flatten().shape[-1]
    Boot_wave = np.random.choice(wp1,size1)
    wp_boot = Boot_wave.reshape(-1,wp.shape[-1])
    return wp_boot

## Wavestrap algo


def wavestrap_algo(x,thresh,level):
    wp = pywt.WaveletPacket2D(data=x, wavelet='db4', mode='sym',maxlevel=level)


    img_num = len([node.path for node in wp.get_level(level, 'natural')])
    img_dim =  int(np.sqrt(img_num))

    matrix_alphabet = [node.path for node in wp.get_level(level, 'natural')]


    Moran_matrix = np.zeros(img_num)

    for i in range(img_num):
        Moran_matrix[i] = Moran_coeff(wp[matrix_alphabet[i]].data)

    Moran_matrix = Moran_matrix.reshape(-1,img_dim) 

    wp_boot_matrix = []

    Index = Moran_matrix.flatten()

    for i in range(img_num):
        if i == 0:
            wp_boot_matrix.append(wp[matrix_alphabet[0]].data)
        elif abs(Index[i]) <= thresh:
            wp_boot_matrix.append(wp_boot(wp[matrix_alphabet[i]].data))
        else:
            wp_boot_matrix.append(wp[matrix_alphabet[i]].data)

    n = len(wp_boot_matrix)

    for i in range(n):
        wp[matrix_alphabet[i]].data = wp_boot_matrix[i]
    
    New_image = wp.reconstruct(update=True)
    
    return New_image




### Rubner paper implementation


def euclidean_dist(row_colors, candidate_prototypes):
    # Based on expansion of the l2 norm formula, python broadcasting and numpy's Einstein summation.
    e1 = np.einsum('ij,ji->i', row_colors, row_colors.T)
    e2 = np.einsum('ij,ji->i', candidate_prototypes,
                   candidate_prototypes.T)[:, None]
    e3 = 2 * candidate_prototypes @ row_colors.T
    dists = np.sqrt(e1 + e2 - e3)

    return dists


def assignment(dists):

    assig = np.argmin(dists, 0)

    return assig


def KMEANS(image, init_prototypes, verbose=True, module=2, k=2):

    # A prototype must be of the same dimension as a data point.
    assert image.shape[1] == init_prototypes.shape[1]

    range_ = np.arange(image.shape[0])
    distance = euclidean_dist(image, init_prototypes)
    new_assig = assignment(distance)
    indices = [new_assig == i for i in range(k)]
    new_prototypes = np.array([np.mean(image[idx, :], 0) for idx in indices])
    old_prototypes = init_prototypes.copy()
    counter = 0

    while (new_prototypes != old_prototypes).any():

        counter += 1
        old_prototypes = new_prototypes.copy()
        distance = euclidean_dist(image, old_prototypes)
        new_assig = assignment(distance)
        indices = [new_assig == i for i in range(k)]
        new_prototypes = np.array(
            [np.mean(image[idx, :], 0) for idx in indices])

        if verbose:
            if counter % module == 0:
                print(f"Loss = {np.sum([distance[idx] for idx in zip(new_assig, range_)]).round(4)}")

    if verbose:
        print(f"Final loss = {np.sum([distance[idx] for idx in zip(new_assig, range_)]).round(4)}")

    return new_prototypes, new_assig


def rgb2lab(x):
    return color.rgb2lab(x)


def lab2rgb(x):
    return color.lab2rgb(x)

## Compute the EMD between two images, not my code:

def Comp_EMD(P,Q,l):
    D = np.ndarray(shape=(l, l))

    for i in range(l):
        for j in range(l):
            D[i,j] = abs(range(l)[i] - range(l)[j])


    A_r = np.zeros((l, l, l))
    A_t = np.zeros((l, l, l))

    for i in range(l):
        for j in range(l):
            A_r[i, i, j] = 1
            A_t[i, j, i] = 1


    A = np.concatenate((A_r.reshape((l, l**2)), A_t.reshape((l, l**2))), axis=0)

    b = np.concatenate((P, Q), axis=0)
    c = D.reshape((l**2))
    
    opt_res = linprog(c, A_eq=A, b_eq=b, bounds=[0, None]) # bounds to set the flow matrix greater than 0
    emd = opt_res.fun
    
    return emd


## Add noise, not my code

def pep_salt(image,amount):
    #row,col,ch = image.shape
    row,col = image.shape
    s_vs_p = 0.5
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[coords] = 1

    # Pepper mode
    num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    out[coords] = 0
    return out


## Last step for ISIS paper

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()


def wav_EMD(X,Y):
    
    level = 1

    X_tf = pywt.WaveletPacket2D(data=X, wavelet='db4', mode='sym',maxlevel=level)
    Y_tf = pywt.WaveletPacket2D(data=Y, wavelet='db4', mode='sym',maxlevel=level)
    
    ## for X_tf

    matrix_alphabet = [node.path for node in X_tf.get_level(level, 'natural')]

    n = len(matrix_alphabet)

    Img_name1 = ["" for x in range(n)]

    for i in range(n):
        Img_name1[i] = "wav_decomp_image" + "_" + str(i) + ".png"


    for i in range(n):
        plt.imshow(X_tf[matrix_alphabet[i]].data)
        plt.axis('off')     
        plt.savefig(Img_name1[i], bbox_inches = 'tight', pad_inches = 0)
        
    ## for Y_tf
    
    matrix_alphabet1 = [node.path for node in Y_tf.get_level(level, 'natural')]

    n = len(matrix_alphabet)

    Img_name2 = ["" for x in range(n)]

    for i in range(n):
        Img_name2[i] = "wav_decomp_image1" + "_" + str(i) + ".png"


    for i in range(n):
        plt.imshow(Y_tf[matrix_alphabet1[i]].data)
        plt.axis('off')     
        plt.savefig(Img_name2[i], bbox_inches = 'tight', pad_inches = 0)
    
    ####
    
    n = len(matrix_alphabet)
    store_wav = np.zeros(n^2)
    
    X = Img_name1
    Y = Img_name2
    
    for i in range(n):
        for j in range(n):

            ## flatten image
            
            start = np.array(Image.open(X[i]))
            start = start[:,:,:3]
            X_new = start.reshape(len(start),-1)
    
            target = np.array(Image.open(Y[j]))
            target = target[:,:,:3]
            Y_new = target.reshape(len(target),-1)

            total_clusters = 2
    
            np.random.seed(1998)

            # Initialize the K-Means model
            kmeans = MiniBatchKMeans(n_clusters = total_clusters)
            # Fitting the model to training set
            kmeans.fit(X_new)

            clusters, number = np.unique(kmeans.labels_, return_counts=True)

            weights = number/np.sum(number)

            # Initialize the K-Means model
            kmeans = MiniBatchKMeans(n_clusters = total_clusters)
            # Fitting the model to training set
            kmeans.fit(Y_new)

            clusters1, number1 = np.unique(kmeans.labels_, return_counts=True)

            weights1 = number1/np.sum(number1)

            P = weights
            Q = weights1
            l = P.shape[0]

            store_wav[i] = Comp_EMD(P,Q,l)
    
    wav_mat = np.sum(store_wav)
    return wav_mat

## Construct the confidence band

def EMD_image(Img,Img1):

    X = np.array(Image.open(Img))
    X = X[:,:,:3]
    X_new = X.reshape(len(X),-1)
    
    X1 = np.array(Image.open(Img1))
    X1 = X1[:,:,:3]
    X1_new = X1.reshape(len(X1),-1)

    total_clusters = 2
    
    np.random.seed(1998)
    
    # Initialize the K-Means model
    kmeans = MiniBatchKMeans(n_clusters = total_clusters)
    # Fitting the model to training set
    kmeans.fit(X_new)

    clusters, number = np.unique(kmeans.labels_, return_counts=True)

    weights = number/np.sum(number)

    # Initialize the K-Means model
    kmeans = MiniBatchKMeans(n_clusters = total_clusters)
    # Fitting the model to training set
    kmeans.fit(X1_new)

    clusters1, number1 = np.unique(kmeans.labels_, return_counts=True)

    weights1 = number1/np.sum(number1)

    P = weights
    Q = weights1
    l = P.shape[0]

    return Comp_EMD(P,Q,l)

## Conf Band 

# dim is the white noise size and B is thenumber of repetition of the process

def Conf_band(dim,B):

    ## Theoritical white noise cov matrice and its image

    tcorr_mat = np.zeros((dim,dim))
    np.fill_diagonal(tcorr_mat, 1)

    t_wn = ""
    t_wn = "theoritcal_wn" + ".png"

    plt.imshow(tcorr_mat)
    plt.axis('off')
    plt.savefig(t_wn, bbox_inches = 'tight', pad_inches = 0)

    L_band = np.zeros(B)
    U_band = np.zeros(B)

    for h in range(B):

        ## White noise

        series = np.random.normal(0,1,dim**2)

        ## Cov matrix

        ind = np.int(np.sqrt(series.shape[0]))
        corr_mat = abs(np.corrcoef(series.reshape(ind,ind)))

        ## Plot all the bootstrap:

        Bootstraped_image = []
        thresh = 0.15
        x = corr_mat

        level = 1
        sim_range = 3

        for i in range(sim_range + 1):
                Bootstraped_image.append(wavestrap_algo(x,thresh,level))
                x = wavestrap_algo(x,thresh,level)

        ## last step of wavestrap        

        rng = len(Bootstraped_image)
        EMD_mat = np.empty((rng,rng))
        EMD_mat[:] = np.nan # to avoid the bug of the double for loop

        for i in range(rng):
            for j in range(rng):
                warnings.filterwarnings("ignore")
                EMD_mat[i,j] = wav_EMD(Bootstraped_image[i],Bootstraped_image[j])

        ## TSP

        permutation, distance = solve_tsp_dynamic_programming(EMD_mat)

        ## Save the figures to compute to construct the confidence band.

        Img_confidence = ["" for x in range(len(permutation))]

        for i in range(len(permutation)):
            Img_confidence[i] = "Conf_int" + "_" + str(i) + ".png"

        for i in range(len(permutation)):
            plt.imshow(Bootstraped_image[permutation[i]])
            plt.axis('off')
            plt.savefig(Img_confidence[i], bbox_inches = 'tight', pad_inches = 0)

        ## Make the confidence interval of our test.

        L_band[h] = EMD_image(t_wn,Img_confidence[0])
        U_band[h] = EMD_image(t_wn,Img_confidence[len(permutation) - 1])
    
    return L_band, U_band

## test for white noise

def test_for_WN(Lower_band,Upper_band,X,dim):
    
    ## Theoritical white noise 
    
    mat = np.zeros((dim,dim))
    np.fill_diagonal(mat, 1)
    
    t_wn1 = ""
    t_wn1 = "theoritcal_wn1" + ".png"

    plt.imshow(mat)
    plt.axis('off')
    plt.savefig(t_wn1, bbox_inches = 'tight', pad_inches = 0)
    
    EMD = EMD_image(t_wn1,X)
    
    if EMD >= Lower_band and EMD <= Upper_band:
        print('We accept that this is a white noise')
        
    else:
        print('we can not conclude if this is a white noise')
    
