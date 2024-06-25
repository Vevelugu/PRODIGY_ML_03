''' 
 I have trained the SVM model on images of cats and dogs and saved the model
 to a pickle file on github.com. 
 To test the model, download the raw pickle file from "Vevelugu/PRODIGY_ML_03" and put it in downloads.
 To make it as convinient as possible to test this code I have included a mechanism to allow the user to
 run the code by simply downloading the zip file of the  repository and extracting it in the downloads 
 folder itself. However, for this to work the user shouldn't have changed the location or name of the
 downloads folder and the system should have english locale. The method won't work otherwise and the user
 should paste the path to the folders in question in the certain places required.
 Will explain further at the code in comments. 
 
 The testfunc() function can be used with the downloaded model directly and is hence put here first 
 The trainfunc() function which was used to train the model is below. It can be used to train 
 another model. 
 '''
# importing necessary libraries
import pandas as pd
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from pathlib import Path
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler as mms
import pickle


#insert path to the downloaded svm model here if this doesn't work by itself
model_file = r"dogsvscats_svm_pickle.pkl" 

downloads_path = str(os.path.join(Path.home(), "Downloads"))
model_path = str(os.path.join(downloads_path, model_file))

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

'''
 classifying using the trained model with images of cats and dogs
 the default directory is the test directory from the downloads folder(download the zip from repo).
 any directory with images can be given. 
 '''

test_folder = r"PRODIGY_ML_03-main\dogs-vs-cats\test"
test_dir = os.path.join(downloads_path, test_folder)
classer = testfunc(model_dl, test_dir)
print(classer)

# Training the model by reading the images and converting image data into arrays
def trainfunc(imagedir):
    Categories = ['Cats', 'Dogs']
    input_arr = []
    output_arr = []
    #imagedir = r"D:\AK\Career\Python\ProdigyInfotech\Task03_SVM\PRODIGY_ML_03\dogs-vs-cats\train"
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

 #parameters used to train the SVM by. The higher number of values the longer
 #time taken by the SVM to learn but also higher the accuracy. 
 #used the following for training in my laptop
 
    parameter_grid = {'C' : [1,10,100], 'gamma' :[0.001,0.01,0.1], 'kernel' : ['rbf', 'linear','poly']}
#for practicality only used a few of these values for actual training

    svc = svm.SVC(probability=True)

    model = GridSearchCV(svc, parameter_grid) #had intended to use this but my pc didn't allow
    
    print("Model is being trained")

    model.fit(xtrain, ytrain)
    
    print(f"Model has been trained\nTesting")

    ypred = model.predict(xtest)  
    
    print("Model has been tested")

    acc = accuracy_score(ypred, ytest)

    print(f"Model Accuracy is {acc*100}%")

    print(classification_report(ytest, ypred, target_names=['cat','dog']))
    
    return model

# put location of training directory here if this doesn't work
train_folder = r"PRODIGY_ML_03-main\dogs-vs-cats\train"
traindir = str(os.path.join(downloads_path, train_folder))
model = trainfunc(traindir)
preds = testfunc(model, test_dir)
print(preds)







# to avoid retraining the model everytime saved the model using following code
model_pkl_file = r"D:\New folder\dogsvscats_svm_pickle.pkl"
# saving the model with pickle
with open(model_pkl_file, 'wb') as file: 
    pickle.dump(model, file)
