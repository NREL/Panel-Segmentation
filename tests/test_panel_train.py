"""
Test suite for panel segmentation code. 
"""
import os

#Set the current working directory as Panel-Segmentation
os.chdir(os.path.dirname(os.path.dirname( __file__ )))

#Import other packages
import pytest
import pandas as pd
import numpy as np
from panel_segmentation import panel_train as pt
from tensorflow.keras.preprocessing import image as imagex
import tensorflow as tf

def assert_isinstance(obj, klass):
    assert isinstance(obj, klass), f'got {type(obj)}, expected {klass}'


@pytest.fixture()
def test_load_images_to_numpy_array():
    """
    Test the loadImagesToNumpyArray() function.

    Returns
    -------
    None.

    """    
    #Variables
    batch_size= 16
    no_epochs =  10
    learning_rate = 1e-5
    train_ps = pt.TrainPanelSegmentationModel(batch_size, no_epochs, learning_rate)
    image_file_path = "./examples/Train/Images/"
    img_np_array = train_ps.loadImagesToNumpyArray(image_file_path)
    #Check the numpy array dimensions
    assert img_np_array.shape == (36, 640, 640, 3)


@pytest.fixture()
def test_train_segmentation():
    """
    Test the trainSegmentation() function.

    Returns
    -------
    None.

    """    
    #Variables
    batch_size= 16
    no_epochs =  4
    learning_rate = 1e-5
    train_ps = pt.TrainPanelSegmentationModel(batch_size, no_epochs, learning_rate)
    #Use the images/masks from the examples folder
    train_data_path = "./examples/Train/Images/"
    train_mask_path = "./examples/Train/Masks/"
    val_data_path = "./examples/Validate/Images/"
    val_mask_path = "./examples/Validate/Masks/"
    #Read in the images as 4D numpy arrays 
    train_data = train_ps.loadImagesToNumpyArray(train_data_path)
    train_mask = train_ps.loadImagesToNumpyArray(train_mask_path)
    val_data = train_ps.loadImagesToNumpyArray(val_data_path)
    val_mask = train_ps.loadImagesToNumpyArray(val_mask_path)
    #Train the segmentation model
    [mod,results] = train_ps.trainSegmentation(train_data, train_mask, 
                                               val_data, val_mask)
    #Make assertions about model mod and the results
    assert (type(mod) == tf.python.keras.engine.functional.Functional) & \
            (type(results) == tf.python.keras.callbacks.History) & \
            (list(results.history.keys()) == ['loss', 'accuracy', 'diceCoeff']) & \
            (len(results.history['loss']) == 4)

@pytest.fixture()
def test_train_panel_classifier():
    """
    Test the trainPanelClassifier() function.

    Returns
    -------
    None.

    """    
    #Variables
    batch_size= 16
    no_epochs =  4
    learning_rate = 1e-5
    train_ps = pt.TrainPanelSegmentationModel(batch_size, no_epochs, learning_rate)
    #Train the classifier model
    [mod,results] = train_ps.trainPanelClassifier("./Train_Classifier/", 
                                                "./Validate_Classifier/")
    #Assert the mod and results types.
    assert (type(mod) == tf.python.keras.engine.functional.Functional) & \
            (type(results) == tf.python.keras.callbacks.History) 
