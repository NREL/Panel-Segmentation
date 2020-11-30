from panel_segmentation import panel_train as ptrain
import numpy as np
import os
#Set the current working directory as the /examples/ folder. 
os.chdir("./examples")
os.getcwd()

batch_size= 16
no_epochs =  10
learning_rate = 1e-5
    
paneltrain = ptrain.TrainPanelSegmentationModel(batch_size, no_epochs, learning_rate)   

#Use the images/masks from the examples folder
train_data_path = "./Train/Images/"
train_mask_path = "./Train/Masks/"
val_data_path = "./Validate/Images/"
val_mask_path = "./Validate/Masks/"

#Read in the images as 4D numpy arrays 
train_data = paneltrain.loadImagesToNumpyArray(train_data_path)
train_mask = paneltrain.loadImagesToNumpyArray(train_mask_path)
val_data = paneltrain.loadImagesToNumpyArray(val_data_path)
val_mask = paneltrain.loadImagesToNumpyArray(val_mask_path)

#Train the segmentation model
[mod,results] = paneltrain.trainSegmentation(train_data, train_mask, val_data, val_mask)   
#Display the training statistics
paneltrain.trainingStatistics(results, 1)

[mod,results] = paneltrain.trainPanelClassifier("./Train_Classifier/", 
                                                "./Validate_Classifier/")
paneltrain.trainingStatistics(results, 1)