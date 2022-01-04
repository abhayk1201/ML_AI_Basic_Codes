# Abhay Kumar (kumar95)
# CS540 HW3: PCA

from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

'''
load_and_center_dataset(filename) — load the dataset from a provided .npy file,
re-center it around the origin and return it as a NumPy array of floats
'''
def load_and_center_dataset(filename):
    dataset = np.load(filename)
    u_mean = np.mean (dataset, axis = 0)
    return dataset-u_mean


''' 
get_covariance(dataset) — calculate and return the covariance
matrix of the dataset as a NumPy matrix (d x d array)
'''
def get_covariance(dataset):
    #cov = np.zeros(dataset.shape)
    cov = np.dot( np.transpose(dataset), dataset)
    #result for our sample dataset should be a d x d (1024 x 1024) matrix.
    return cov/(len(dataset)-1)


''' 
get_eig(S, m) — perform eigen decomposition on the covariance matrix S 
and return a diagonal matrix (NumPy array) with the largest m eigenvalues on the 
diagonal, and a matrix (NumPy array) with the corresponding eigenvectors as columns
'''
def get_eig(S, m):
    eig_value, eig_vec = eigh(S, eigvals=(len(S)-m,len(S)-1))
    return np.diag(eig_value[::-1]), eig_vec[:,::-1]


'''
get_eig_perc(S, perc) — similar to get_eig, but instead of returning the first m,
return all eigenvalues and corresponding eigenvectors in similar format 
as get_eig that explain more than perc % of variance
'''
def get_eig_perc(S, perc) :
    
    eig_value, eig_vec = eigh(S)
    ind = eig_value/sum(eig_value) > perc
    eig_value = eig_value[ind]
    eig_vec = eig_vec[:,ind]
    return np.diag(eig_value[::-1]), eig_vec[:,::-1]

    
'''
project_image(image, U) — project each image into your m-dimensional space and 
return the new representation as a d x 1 NumPy array
'''
 
def project_image(image, U):
    return np.dot(U, np.dot(np.transpose(U),image))

'''
    display_image(orig, proj) — use matplotlib to display a visual representation of the 
    original image and the projected image side-by-side
'''
def display_image(orig, proj) :
    orig_img = np.transpose(np.reshape(orig,(32,32)))
    proj_img = np.transpose(np.reshape(proj,(32,32)))
    fig, axs = plt.subplots(figsize=(15, 5), ncols=2, constrained_layout=True)
    axs[0].set_title("Original")
    axs[1].set_title("Projection")
    img1 = axs[0].imshow(orig_img, aspect='equal')
    plt.colorbar(img1, ax= axs[0], aspect=75)
    img2 = axs[1].imshow(proj_img, aspect='equal')
    plt.colorbar(img2,  ax= axs[1], aspect=75)
    plt.show()