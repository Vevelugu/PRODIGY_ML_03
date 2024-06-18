# I have trained the SVM model on images of cats and dogs and saved the model
# to a pickle file on github.com. 
# To test the model, download the raw pickle file from "Vevelugu/PRODIGY_ML_03" and 
# path location in the "model_path" variable
# The test function can be used with the model directly and is hence put here first 
# The train function which was used to train the model is under the test function.
# To use the test function against a model  

import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler as mms
import pickle
#insert path to the downloaded svm model here
model_path = r"D:\AK\Career\Python\ProdigyInfotech\Task03_SVM\PRODIGY_ML_03\dogsvscats_svm_pickle.pkl" 

# load the downloaded model here
with open(model_path, 'rb') as file: 
    model_dl=pickle.load(file)

# testing the model against the test folder given in the dataset
def testfunc(model, test_dir):
    input_arr = []
    
    for img in os.listdir(test_dir):
        img_array = imread(os.path.join(test_dir, img)) #
        img_rsz = resize(img_array, (150,150,3))
        input_arr.append(img_rsz.flatten())
    df = pd.DataFrame(input_arr)
    #df = df.sample(n=(.2*len(df)), random_state=1) # use this if you want to sample the data
    scaling = mms(feature_range=(-1,1)).fit(df)
    test_input = scaling.transform(df)
    preds = model.predict(test_input)
    return preds

#classifying using the trained model with images of cats and dogs
# the default directory is the test directory from the repository.
# any directory with images can be given.
test_dir = r"D:\AK\Career\Python\ProdigyInfotech\Task03_SVM\PRODIGY_ML_03\dogs-vs-cats\test"
classer = testfunc(model_dl, test_dir)
print(classer)

# Training the model by reading the images and converting image data into arrays
def trainfunc(imagedir):
    Categories = ['Cats', 'Dogs']
    input_arr = []
    output_arr = []
    imagedir = r"D:\AK\Career\Python\ProdigyInfotech\Task03_SVM\PRODIGY_ML_03\dogs-vs-cats\train"
    for i in Categories:  # i here is the folder for the category i.e "Cats" for cat images and "Dogs" for dog images
        print(f"Loading... Category {i}")
        path= os.path.join(imagedir, i) #joins the 2 paths 
        for img in os.listdir(path): #listdir returns a list of all files in the spec path
            img_array = imread(os.path.join(path, img)) # the data of the image is gathered
            img_rsz = resize(img_array, (150,150,3)) # the image is resized to maintain uniform dimensions
            input_arr.append(img_rsz.flatten()) # makes array a 1D
            output_arr.append(Categories.index(i)) # gives value "0" for cats and "1" for dogs
        print(f"Loaded Category {i} successfully")

# converting the python arrays into numpy arrays to make them efficient
    flat_data = np.array(input_arr)  
    target = np.array(output_arr)

# converting into a dataframe to join both arrays together for train test split
    df = pd.DataFrame(flat_data)
    df['Target'] = target
    df.shape
    #df = df.sample(n=2000, random_state=1)

    x = df.iloc[:,:-1] #input data
    y = df.iloc[:,-1]  #output data

# train test split of the data
    xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.20, random_state=77,stratify=y)

# scaling to normalise the values
    scaling = mms(feature_range=(-1,1)).fit(xtrain)
    xtrain = scaling.transform(xtrain)
    xtest = scaling.transform(xtest)

# parameters used to train the SVM by. The higher number of values the longer
# time taken by the SVM to learn but also higher the accuracy. 
# used the following for training in my laptop
# for 'C' and 'gamma' values can also use other float values 
# for 'kernel' can also include 'linear', 'rbf' etc
    parameter_grid = {'C' : [0.1], 'gamma' :[0.001], 'kernel' : ['poly']}

    svc = svm.SVC(probability=True)

    model = GridSearchCV(svc, parameter_grid)

    model.fit(xtrain, ytrain)

    ypred = model.predict(xtest)  

    acc = accuracy_score(ypred, ytest)

    print(f"Model Accuracy is {acc*100}%")

    print(classification_report(ytest, ypred, target_names=['cat','dog']))
    
    return model

# put location of training directory here
traindir = r"D:\AK\Career\Python\ProdigyInfotech\Task03_SVM\PRODIGY_ML_03\dogs-vs-cats\train"

model = trainfunc(traindir)
preds = testfunc(model, test_dir)
print(preds)
# to avoid retraining the model everytime saved the model using following code
model_pkl_file = r"D:\New folder\dogsvscats_svm_pickle.pkl"
# saving the model with pickle
with open(model_pkl_file, 'wb') as file: 
    pickle.dump(model, file)
