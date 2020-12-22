"""
Test suite for panel segmentation code. 
"""
import os

#Import other packages
import pytest
import pandas as pd
import numpy as np
from panel_segmentation import panel_train as pt
from tensorflow.keras.preprocessing import image as imagex
import tensorflow as tf
import tensorflow.keras.backend as K

def test_load_images_to_numpy_array():
    """
    Test the loadImagesToNumpyArray() function.

    Returns
    -------
    None.

    """
    #Clear the tensorflow.keras session (just in case)
    K.clear_session()
    #Variables
    batch_size= 64
    no_epochs =  10
    learning_rate = 1e-5
    train_ps = pt.TrainPanelSegmentationModel(batch_size, no_epochs, learning_rate)
    image_file_path = "./panel_segmentation/examples/Train/Images/"
    img_np_array = train_ps.loadImagesToNumpyArray(image_file_path)
    #Check the numpy array dimensions
    assert img_np_array.shape == (80, 640, 640, 3)


def test_train_panel_classifier():
    """
    Test the trainPanelClassifier() function.

    Returns
    -------
    None.

    """
    #Clear the tensorflow.keras session (just in case)
    K.clear_session()
    #Variables
    batch_size= 16
    no_epochs =  1
    learning_rate = 1e-5
    train_classifier = pt.TrainPanelSegmentationModel(batch_size, no_epochs, 
                                              learning_rate)
    #Train the classifier model
    [mod,results] = train_classifier.trainPanelClassifier(TRAIN_PATH = "./panel_segmentation/examples/Train_Classifier/", 
                                                          VAL_PATH = "./panel_segmentation/examples/Validate_Classifier/",
                                                          model_file_path = './panel_segmentation/tests/classifier.h5')
    #Delete the model 
    os.remove('./panel_segmentation/tests/classifier.h5')
    #Assert the mod and results types.
    assert (str(type(mod)) == "<class 'tensorflow.python.keras.engine.training.Model'>") & \
            (str(type(results)) == "<class 'tensorflow.python.keras.callbacks.History'>")
    
def _test_train_segmentation():
    """
    Test the trainSegmentation() function.

    Returns
    -------
    None.

    """
    #Clear the tensorflow.keras session (just in case)
    K.clear_session()
    #Variables
    batch_size= 16
    no_epochs =  1
    learning_rate = 1e-5
    train_seg = pt.TrainPanelSegmentationModel(batch_size, no_epochs, learning_rate)
    #Use the images/masks from the examples folder
    train_data_path = "./panel_segmentation/examples/Train/Images/"
    train_mask_path = "./panel_segmentation/examples/Train/Masks/"
    val_data_path = "./panel_segmentation/examples/Validate/Images/"
    val_mask_path = "./panel_segmentation/examples/Validate/Masks/"
    #Read in the images as 4D numpy arrays 
    train_data = train_seg.loadImagesToNumpyArray(train_data_path)
    train_mask = train_seg.loadImagesToNumpyArray(train_mask_path)
    val_data = train_seg.loadImagesToNumpyArray(val_data_path)
    val_mask = train_seg.loadImagesToNumpyArray(val_mask_path)
    #Train the segmentation model
    [mod,results] = train_seg.trainSegmentation(train_data = train_data, 
                                                train_mask = train_mask, 
                                                val_data = val_data, 
                                                val_mask = val_mask, 
                                                model_file_path = './panel_segmentation/tests/semantic_segmentation.h5')
    #Delete the model 
    os.remove('./panel_segmentation/tests/semantic_segmentation.h5')
    #Make assertions about model mod and the results
    assert (str(type(mod)) == "<class 'tensorflow.python.keras.engine.training.Model'>") & \
            (str(type(results)) == "<class 'tensorflow.python.keras.callbacks.History'>") & \
            (list(results.history.keys()) == ['loss', 'accuracy', 'diceCoeff', 'val_loss', 'val_accuracy', 'val_diceCoeff']) & \
            (len(results.history['loss']) == 1)

"""
test_load_images_to_numpy_array()
test_train_panel_classifier()
test_train_segmentation()
"""