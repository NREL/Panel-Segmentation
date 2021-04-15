# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 15:43:02 2021

@author: edun2
"""
import numpy as np
from scipy.spatial.distance import pdist, squareform
def getAffinityMatrix(coordinates, k = 7):
    """
    Calculate affinity matrix based on input coordinates matrix and the numeber
    of nearest neighbours.
    
    Apply local scaling based on the k nearest neighbour
        References:
    https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf
    """
    # calculate euclidian distance matrix
    dists = squareform(pdist(coordinates)) 
    
    # for each row, sort the distances ascendingly and take the index of the 
    #k-th position (nearest neighbour)
    knn_distances = np.sort(dists, axis=0)[k]
    knn_distances = knn_distances[np.newaxis].T
    
    # calculate sigma_i * sigma_j
    local_scale = knn_distances.dot(knn_distances.T)

    affinity_matrix = dists * dists
    affinity_matrix = -affinity_matrix / local_scale
    # divide square distance matrix by local scale
    affinity_matrix[np.where(np.isnan(affinity_matrix))] = 0.0
    # apply exponential
    affinity_matrix = np.exp(affinity_matrix)
    np.fill_diagonal(affinity_matrix, 0)
    return affinity_matrix

import matplotlib.pyplot as plt
from scipy.sparse import csgraph
# from scipy.sparse.linalg import eigsh
from numpy import linalg as LA
def eigenDecomposition(A, plot = True, topK = 5):
    """
    :param A: Affinity matrix
    :param plot: plots the sorted eigen values for visual inspection
    :return A tuple containing:
    - the optimal number of clusters by eigengap heuristic
    - all eigen values
    - all eigen vectors
    
    This method performs the eigen decomposition on a given affinity matrix,
    following the steps recommended in the paper:
    1. Construct the normalized affinity matrix: L = D−1/2ADˆ −1/2.
    2. Find the eigenvalues and their associated eigen vectors
    3. Identify the maximum gap which corresponds to the number of clusters
    by eigengap heuristic
    
    References:
    https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf
    http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Luxburg07_tutorial_4488%5b0%5d.pdf
    """
    L = csgraph.laplacian(A, normed=True)
    n_components = A.shape[0]
    
    # LM parameter : Eigenvalues with largest magnitude (eigs, eigsh), that is, largest eigenvalues in 
    # the euclidean norm of complex numbers.
#     eigenvalues, eigenvectors = eigsh(L, k=n_components, which="LM", sigma=1.0, maxiter=5000)
    eigenvalues, eigenvectors = LA.eig(L)
    
    if plot:
        plt.title('Largest eigen values of input matrix')
        plt.scatter(np.arange(len(eigenvalues)), eigenvalues)
        plt.grid()
        
    # Identify the optimal number of clusters as the index corresponding
    # to the larger gap between eigen values
    index_largest_gap = np.argsort(np.diff(eigenvalues))[::-1][:topK]
    nb_clusters = index_largest_gap + 1
        
    return nb_clusters, eigenvalues, eigenvectors

from PIL import Image
# Open the image form working directory
X = np.array(Image.open(r'C:\Users\edun2\OneDrive - University of Florida\Research\NREL\202.PNG'))

affinity_matrix = getAffinityMatrix(X[:,:,0], k = 7)
affinity_matrix = getAffinityMatrix(X[:,:,0], k = 10)
k, _,  _ = eigenDecomposition(affinity_matrix)
print(f'Optimal number of clusters {k}')


from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction import image as imagex
graph = graph = imagex.img_to_graph(X)
n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
n_components


import cv2
import numpy as np
import matplotlib.pyplot as plt

def connected_component_label(img,ii):
    FOLDER_PATH = r"C:\Users\edun2\OneDrive - University of Florida\Desktop\NREL\clustering"
    
    # Getting the input image
    #img = cv2.imread(path, 0)
    # Converting those pixels with values 1-127 to 0 and others to 1
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
    # Applying cv2.connectedComponents() 
    num_labels, labels = cv2.connectedComponents(img[:,:,2].reshape(640,640))
    
    # Map component labels to hue val, 0-179 is the hue range in OpenCV
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # Converting cvt to BGR
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0
    
    
    # Showing Original Image
    #plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    #plt.axis("off")
    #plt.title("Orginal Image")
    #plt.show()
    
    #Showing Image after Component Labeling
    plt.figure()
    plt.imshow(cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Image after Component Labeling")
    kurs = "image %ii.png" % ii
    plt.savefig(FOLDER_PATH+"\\"+kurs, format='png', dpi=300)
    plt.show()

    return num_labels, labels

import glob
image_file_path = r"C:\Users\edun2\OneDrive - University of Florida\Desktop\NREL\image_segmentation_satellite\solar_arrays\data\test_masks\test"
image_file_list = []
files = glob.glob (image_file_path + "/*")

jj=[]
labels=[]
ii=1
for img_file in files:
    image = cv2.imread(img_file)
    image_file_list.append(image)
    #Convert the image_file_list to a 4d numpy array and return it
    img_np_array = np.array(image_file_list)
    ja,label = connected_component_label(image,ii)
    jj.append(ja)
    labels.append(label)
    ii=ii+1
    
               
jj,labels = connected_component_label(r'C:\Users\edun2\OneDrive - University of Florida\Research\NREL\355.PNG')
jj

labels

from panel_segmentation import panel_detection as pseg
import numpy as np
from tensorflow.keras.preprocessing import image as imagex
import matplotlib.pyplot as plt
import os
from PIL import Image

#CREATE AN INSTANCE OF THE PANELDETECTION CLASS TO RUN THE ANALYSIS
panelseg = pseg.PanelDetection(classifier_file_path =r'C:\Users\edun2\OneDrive - University of Florida\Research\NREL\Panel-Segmentation\panel_segmentation\VGG16_classification_model.h5',
                               model_file_path =r'C:\Users\edun2\OneDrive - University of Florida\Research\NREL\Panel-Segmentation\panel_segmentation\examples\VGG16Net_ConvTranpose_complete.h5', 
                               )

img = Image.open(file_name_save)
#Show the generated satellite image
plt.imshow(img)

try:
    f = open("./panel_segmentation/VGG16_classification_model.h5")
    # Do something with the file
except IOError:
    print("File not accessible")
finally:
    f.close()