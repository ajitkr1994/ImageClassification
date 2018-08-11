import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import LinearSVC, SVC
from sklearn import metrics



# initialise OpenCV's SIFT
sift=cv2.SIFT()

# Initialize the y vectors for training SVM
y_train = []
y_test = []



def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if (f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png'))]

#Use the following function when reading an image through OpenCV and displaying through plt.
def showfig(image, ucmap):
    #There is a difference in pixel ordering in OpenCV and Matplotlib.
    #OpenCV follows BGR order, while matplotlib follows RGB order.
    if len(image.shape)==3 :
        b,g,r = cv2.split(image)       # get b,g,r
        image = cv2.merge([r,g,b])     # switch it to rgb
    imgplot=plt.imshow(image, ucmap)
    imgplot.axes.get_xaxis().set_visible(False)
    imgplot.axes.get_yaxis().set_visible(False)
    
def get_cluster_centers_of_descriptors(k, descriptor_mat):

    descriptor_mat=np.double(np.vstack(descriptor_mat))

    print 'Inside clustering module, desciptor_mat.shape', descriptor_mat.shape

    kmeans = KMeans(n_clusters=k)
    kmeans = kmeans.fit(descriptor_mat)
    labels = kmeans.predict(descriptor_mat)
    cluster_centers = kmeans.cluster_centers_
    return cluster_centers

def get_descriptors(dataset):
    
    # name of all the folders together
    folders=['Cars','Airplanes','Motorbikes']

    print 'Getting SIFT features for %s dataset'%dataset
    
    folder_number=-1
    count = 0
    descriptors=[]
      
    for folder in folders:
        folder_number+=1

        #get all the training images from a particular class 
        filenames=get_imlist('C:/Baka/Projects/ImageCategorization/%s/%s'%(dataset,folder))
##        print filenames
        
        for image_name in filenames:
            img=cv2.imread(image_name)

            # carry out normal preprocessing routines
            gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            gray=cv2.resize(gray, (400, 200), interpolation=cv2.INTER_AREA)
            gray=cv2.equalizeHist(gray)

            #get all the SIFT descriptors for an image
            keypoints, des=sift.detectAndCompute(gray, None)
##            print 'descriptor.shape', des.shape
            
            descriptors.append(des)

            count = count+1
##            print 'Done for', str(count), 'images.'

            if dataset == 'Train':
                y_train.append(folder_number)
            elif dataset == 'Test':
                y_test.append(folder_number)

    return descriptors


def cluster_features_train(img_descs, cluster_model):
    """
    Cluster the training features using the cluster_model
    and convert each set of descriptors in img_descs
    to a Visual Bag of Words histogram.
    Parameters:
    -----------
    X : list of lists of SIFT descriptors (img_descs)
    training_idxs : array/list of integers
        Indicies for the training rows in img_descs
    cluster_model : clustering model (eg KMeans from scikit-learn)
        The model used to cluster the SIFT features
    Returns:
    --------
    X, cluster_model :
        X has K feature columns, each column corresponding to a visual word
        cluster_model has been fit to the training set
    """
    n_clusters = cluster_model.n_clusters
    # Concatenate all descriptors in the training set together
    training_descs = img_descs
    all_train_descriptors = [desc for desc_list in training_descs for desc in desc_list]
    all_train_descriptors = np.array(all_train_descriptors)

    if all_train_descriptors.shape[1] != 128:
        raise ValueError('Expected SIFT descriptors to have 128 features, got', all_train_descriptors.shape[1])

    # train kmeans or other cluster model on those descriptors selected above
    print 'Training KNN model...'
    cluster_model.fit(all_train_descriptors)
    print('Done clustering. Using clustering model to generate BoW histograms for each image.')

    # compute set of cluster-reduced words for each image
    img_clustered_words = [cluster_model.predict(raw_words) for raw_words in img_descs]

    # finally make a histogram of clustered word counts for each image. These are the final features.
    img_bow_hist = np.array(
        [np.bincount(clustered_words, minlength=n_clusters) for clustered_words in img_clustered_words])

    X = img_bow_hist
    print('Done generating BoW histograms.')

    return X, cluster_model

def cluster_features_test(img_descs, cluster_model):
    n_clusters = cluster_model.n_clusters
    # Concatenate all descriptors in the training set together
    test_descs = img_descs
    all_test_descriptors = [desc for desc_list in test_descs for desc in desc_list]
    all_test_descriptors = np.array(all_test_descriptors)

    if all_test_descriptors.shape[1] != 128:
        raise ValueError('Expected SIFT descriptors to have 128 features, got', all_test_descriptors.shape[1])

    # compute set of cluster-reduced words for each image
    img_clustered_words = [cluster_model.predict(raw_words) for raw_words in img_descs]

    # finally make a histogram of clustered word counts for each image. These are the final features.
    img_bow_hist = np.array(
        [np.bincount(clustered_words, minlength=n_clusters) for clustered_words in img_clustered_words])

    X = img_bow_hist
    print('Done generating BoW histograms.')

    return X

def train_and_test_model(classifier, Xtrain, ytrain, Xtest, ytest, is_neural_net=False):
    classifier.fit(Xtrain, ytrain)

    pred = classifier.predict(Xtrain)
    score = metrics.accuracy_score(ytrain, pred)

##    print 'Accuracy on training set = ', score*100

    pred = classifier.predict(Xtest)
    score = metrics.accuracy_score(ytest, pred)

##    print 'Accuracy on test set = ', score*100
    return score*100

########################################################################################################################
# Starting the actual code to train and test SVM on the features
########################################################################################################################

# for keeping all the descriptors from the data.
descriptor_template = get_descriptors('Template')
descriptor_train = get_descriptors('Train')
descriptor_test = get_descriptors('Test')
##print 'descriptor_template.shape', descriptor_template[0].shape


NUMBER_OF_CLUSTERS = 250


while NUMBER_OF_CLUSTERS <= 250:
    print "="*80
    print 'Number of clusters = ', NUMBER_OF_CLUSTERS
    #Training the model on template images to get the bag of words vocabulary, and cluster centers.
    template_features , knn = cluster_features_train(descriptor_template, KMeans(n_clusters=NUMBER_OF_CLUSTERS))

    #Use the trained knn model to generate training and test features.
    X_train = cluster_features_test(descriptor_train, knn)
    X_test = cluster_features_test(descriptor_test, knn)


    ###################################################################
    # Implement SVM
    ###################################################################


    
    print 'Training SVM model...'
    c_parameter = 0.001
    best_score = 0
    while c_parameter <= 1000:
        svmModel = LinearSVC(C=c_parameter,penalty="l2", dual= False, tol=1e-3) #Penalty is L2 and not twelve.
    ##    print 'c_parameter = ', c_parameter
        score = train_and_test_model(svmModel, X_train, y_train, X_test, y_test)
        c_parameter *=2

        if score > best_score:
            best_score = score

    print 'Number of clusters = ',str(NUMBER_OF_CLUSTERS),'; Accuracy of SVM = ',str(best_score)

    NUMBER_OF_CLUSTERS = NUMBER_OF_CLUSTERS + 100


