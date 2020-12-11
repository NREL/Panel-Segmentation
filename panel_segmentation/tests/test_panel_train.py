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
    image_file_path = "./examples/Train/Images/"
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
    [mod,results] = train_classifier.trainPanelClassifier("./examples/Train_Classifier/", 
                                                "./examples/Validate_Classifier/")
    #Assert the mod and results types.
    assert (type(mod) == tf.python.keras.engine.functional.Functional) & \
            (type(results) == tf.python.keras.callbacks.History) 