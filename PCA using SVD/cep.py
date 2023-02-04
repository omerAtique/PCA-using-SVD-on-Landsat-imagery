import numpy as np #operations on matrixes
from sklearn.utils.extmath import svd_flip #just for the svd_flip function
import matplotlib.pyplot as plt #used for plotting our images or graphs if required
import cv2 #helps in manipulating the image and with preprocessing
import numpy.linalg as la #the Linear Algebra library in numpy. Used for SVD function.
from sklearn.decomposition import PCA #extracted the pca application steps from this function
plt.style.use('ggplot')

def pcaSVD(X, k): #PCA function start

    #split the image into RGB bands for individual processing/PCA application
    blue, green, red = cv2.split(X)

    #standardization
    blue = blue/255
    green = green/255
    red = red/255
    
    #mean required for further operations
    b_m = np.mean(blue)
    g_m = np.mean(green)
    r_m = np.mean(red)

    #subtract corresponding mean with the rgb matrices?
    b_mean = blue - b_m
    g_mean = green - g_m
    r_mean = red - r_m

    #SVD decomposition into U, sigma and V
    U_b, s_b, Vt_b = la.svd(b_mean, full_matrices = False)
    #get the required no of components as specified by the value of 'k'
    components_b = Vt_b[0:k]
    #svd_flip U and V. flip eigenvectors' sign to enforce deterministic output
    U_b, Vt_b = svd_flip(U_b, Vt_b)
    #sigma is a row of eigenvalues and we need a nxn matrix with eigenvalues on them main diagnal
    Sigma_b = np.diag(s_b)
    #Final operation to get all the principal components 500xk
    pcab = np.dot(U_b[:, 0:k], Sigma_b[0:k, 0:k])    #Inverse_transform function to stretch the reduced image back to original size 500x500 
    pca_b = np.dot(pcab, components_b) + b_m

    U_g, s_g, Vt_g = la.svd(g_mean, full_matrices = False)
    components_g = Vt_g[0:k]
    U_g, Vt_g = svd_flip(U_g, Vt_g)
    Sigma_g = np.diag(s_g)
    pcag = np.dot(U_g[:, 0:k], Sigma_g[0:k, 0:k])
    pca_g = np.dot(pcag, components_g) + g_m

    U_r, s_r, Vt_r = la.svd(r_mean, full_matrices = False)
    components_r = Vt_r[0:k]
    U_r, Vt_r = svd_flip(U_r, Vt_r)
    Sigma_r = np.diag(s_r)
    pcar = np.dot(U_r[:, 0:k], Sigma_r[0:k, 0:k])
    pca_r = np.dot(pcar, components_r) + r_m

    
    #Display the blue green and red bands of each image after PCA applications
    
    fig = plt.figure(figsize = (15, 7.2)) 
    fig.add_subplot(131)
    plt.title("PCAb Image")
    plt.imshow(pca_b)
    
    fig.add_subplot(132)
    plt.title("PCAg Image")
    plt.imshow(pca_g)
    
    fig.add_subplot(133)
    plt.title("PCAr Image")
    plt.imshow(pca_r)
    plt.show()
    
    #combine the red green and blue bands back to the merged image
    img_reduced = (cv2.merge((pca_r, pca_g, pca_b)))

    return img_reduced

im = ['im1.TIF','im2.TIF', 'im3.TIF','im4.TIF','im5.TIF','im6.TIF','im7.TIF','im8.TIF','im9.TIF','im10.TIF','im11.TIF','im12.TIF',]

no_img = 12
merImg = 0
img_reduced = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  
for i in range(12):
    #reads image. Iterates thru bands one after the other 
    image = cv2.imread(im[i])

    #resizes the image from (7841,7701,3)
    im1 = cv2.resize(image, (1000, 1000),interpolation = cv2.INTER_LINEAR)
    
    #rearranges the RGB according to requirement. Used here to elimated any ambiguity.
    #we arrange as Blue, Green, Red.
    img = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)

    #here we call the the PCA function with value of k as 10
    img_reduced[i] = pcaSVD(img, 10)

    #merges the images to  make one combined image
    merImg = merImg + (img_reduced[i])/no_img

    print(i+1, "..")
    
    #mean squared error implementation
    err = np.sum((img.astype("float")-img_reduced[i].astype("float"))**2)
    err /= float(img.shape[0] * img.shape[1])

    print("The mean squared Error is : ", err)
    
    #shows the original image against the image after PCA implementation 
    
    fig = plt.figure(figsize = (10, 7.2)) 
    fig.add_subplot(121)
    plt.title("Original Image")
    plt.imshow(img)
    fig.add_subplot(122)
    plt.title("Reduced Image")
    plt.imshow(img_reduced[i])
    plt.show()
    
    
    
#shows the final merged image+
fig = plt.figure(figsize = (10, 7.2)) 
fig.add_subplot(121)
plt.title("Final Merged Image")
plt.imshow(merImg)
plt.show()