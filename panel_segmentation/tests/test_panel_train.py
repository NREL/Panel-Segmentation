"""
Test suite for panel segmentation code.
"""
import os
from panel_segmentation import panel_train as pt
import tensorflow.keras.backend as K
import torch

def testLoadImagesToNumpyArray():
    # Clear the tensorflow.keras session (just in case)
    K.clear_session()
    # Variables
    batch_size = 2
    no_epochs = 1
    learning_rate = 1e-5
    train_ps = pt.TrainPanelSegmentationModel(
        batch_size, no_epochs, learning_rate)
    image_file_path = "./panel_segmentation/examples/Train/Images/"
    img_np_array = train_ps.loadImagesToNumpyArray(image_file_path)
    # Check the numpy array dimensions
    assert img_np_array.shape == (2, 640, 640, 3)


def testTrainPanelClassifier():
    # Clear the tensorflow.keras session (just in case)
    K.clear_session()
    # Variables
    batch_size = 4
    no_epochs = 1
    learning_rate = 1e-5
    train_classifier = pt.TrainPanelSegmentationModel(batch_size, no_epochs,
                                                      learning_rate)
    # Train the classifier model
    [mod, results] = train_classifier.trainPanelClassifier(
        train_path="./panel_segmentation/examples/Train_Classifier/",
        val_path="./panel_segmentation/examples/Validate_Classifier/",
        model_file_path='./panel_segmentation/tests/classifier.h5')
    # Delete the model
    os.remove('./panel_segmentation/tests/classifier.h5')
    # Assert the mod and results types.
    assert (str(type(mod)) ==
            "<class 'tensorflow.python.keras.engine.training.Model'>") & \
        (str(type(results)) ==
         "<class 'tensorflow.python.keras.callbacks.History'>")


def testTrainSegmentation():
    # Clear the tensorflow.keras session (just in case)
    K.clear_session()
    # Variables
    batch_size = 2
    no_epochs = 1
    learning_rate = .1
    train_seg = pt.TrainPanelSegmentationModel(
        batch_size, no_epochs, learning_rate)
    # Use the images/masks from the examples folder
    train_data_path = "./panel_segmentation/examples/Train/Images/"
    train_mask_path = "./panel_segmentation/examples/Train/Masks/"
    val_data_path = "./panel_segmentation/examples/Validate/Images/"
    val_mask_path = "./panel_segmentation/examples/Validate/Masks/"
    # Read in the images as 4D numpy arrays
    train_data = train_seg.loadImagesToNumpyArray(train_data_path)
    train_mask = train_seg.loadImagesToNumpyArray(train_mask_path)
    val_data = train_seg.loadImagesToNumpyArray(val_data_path)
    val_mask = train_seg.loadImagesToNumpyArray(val_mask_path)
    # Train the segmentation model
    [mod, results] = \
        train_seg.trainSegmentation(
            train_data,
            train_mask,
            val_data,
            val_mask,
            './panel_segmentation/tests/semantic_segmentation.h5')
    # Delete the model
    os.remove('./panel_segmentation/tests/semantic_segmentation.h5')
    # Make assertions about model mod and the results
    assert (str(type(mod)) ==
            "<class 'tensorflow.python.keras.engine.training.Model'>") & \
        (str(type(results)) ==
         "<class 'tensorflow.python.keras.callbacks.History'>") & \
        (list(results.history.keys()) ==
         ['loss', 'accuracy', 'diceCoeff', 'val_loss',
          'val_accuracy', 'val_diceCoeff']) & \
        (len(results.history['loss']) == 1)


def testTrainMountingConfigClassifier():
    # Variables
    batch_size = 20
    no_epochs = 1
    learning_rate = .1
    train_mask_path = \
        "./panel_segmentation/examples/Train_Mount_Object_Detection"
    val_data_path = \
        "./panel_segmentation/examples/Validate_Mount_Object_Detection"
    train_seg = pt.TrainPanelSegmentationModel(batch_size, no_epochs,
                                               learning_rate)
    model = train_seg.trainMountingConfigClassifier(train_path=train_mask_path,
                                                    val_path=val_data_path,
                                                    device=torch.device('cpu'))
    assert (str(type(model)) == "<class 'detecto.core.Model'>")
