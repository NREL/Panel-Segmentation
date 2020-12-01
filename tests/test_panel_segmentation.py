"""
Test suite for panel segmentation code. 
"""
import os

#Set the current working directory as Panel-Segmentation
os.chdir(os.path.dirname(os.path.dirname( __file__ )))

import pytest
import pandas as pd
import numpy as np
from panel_segmentation import panel_detection as pan_det
from tensorflow.keras.preprocessing import image as imagex
import PIL


def assert_isinstance(obj, klass):
    assert isinstance(obj, klass), f'got {type(obj)}, expected {klass}'


def test_generate_satellite_image():
    """
    Test the generateSatelliteImage() function.

    Returns
    -------
    None.

    """    
    with pytest.raises(ValueError):
        #Variables
        latitude = 39.7407
        longitude = -105.1694    
        google_maps_api_key =  "Wrong_API_key"     
        file_name_save = "./examples/Panel_Detection_Examples/sat_img.png"
        #Create an instance of the PanelDetection() class.
        pc = pan_det.PanelDetection(model_file_path = './panel_segmentation/VGG16Net_ConvTranpose_complete.h5', 
                                    classifier_file_path = './panel_segmentation/VGG16_classification_model.h5')
        pc.generateSatelliteImage(latitude, longitude, 
                                  file_name_save, google_maps_api_key)
    

def test_has_panels():
    """
    Test the hasPanels() function.

    Returns
    -------
    Assert that the value returned is a boolean
    """
    #Pick the file name to read 
    img_file = "./examples/Panel_Detection_Examples/sat_img.png"
    #Create an instance of the PanelDetection() class.
    pc = pan_det.PanelDetection(model_file_path = './panel_segmentation/VGG16Net_ConvTranpose_complete.h5', 
                                classifier_file_path = './panel_segmentation/VGG16_classification_model.h5')
    #Read in the image
    x = imagex.load_img(img_file, 
                        color_mode='rgb', 
                        target_size=(640,640))
    x = np.array(x)
    #Assert that the returned value is a boolean
    panel_loc = pc.hasPanels(x)
    assert_isinstance(panel_loc, bool)
    

def test_mask_generator():
    """
    Test the testSingle() function.

    Returns
    -------
    None.

    """
    #Pick the file name to read 
    img_file = "./examples/Panel_Detection_Examples/sat_img.png"
    #Create an instance of the PanelDetection() class.
    pc = pan_det.PanelDetection(model_file_path = './panel_segmentation/VGG16Net_ConvTranpose_complete.h5', 
                                classifier_file_path = './panel_segmentation/VGG16_classification_model.h5')
    #Read in the image
    #Read in the image
    x = imagex.load_img(img_file, 
                        color_mode='rgb', 
                        target_size=(640,640))
    x = np.array(x)
    #Mask the satellite image
    res = pc.testSingle(x.astype(float), test_mask=None,  model =None)
    #Assert that the 'res' variable is a numpy array and the dimensions.
    assert (type(res) == np.ndarray) & (res.shape == (640, 640))


def test_crop_panels():
    """
    Test the cropPanels() function.

    Returns
    -------
    None.

    """
    #Pick the file name to read 
    img_file = "./examples/Panel_Detection_Examples/sat_img.png"
    #Create an instance of the PanelDetection() class.
    pc = pan_det.PanelDetection(model_file_path = './panel_segmentation/VGG16Net_ConvTranpose_complete.h5', 
                                classifier_file_path = './panel_segmentation/VGG16_classification_model.h5')
    #Read in the image
    x = imagex.load_img(img_file, 
                        color_mode='rgb', 
                        target_size=(640,640))
    x = np.array(x)
    #Mask the satellite image
    res = pc.testSingle(x.astype(float), test_mask=None,  model =None)
    #Crop the panels 
    new_res = pc.cropPanels(x, res)
    #Assert that the 'new_res' variable is a numpy array and the dimensions.
    assert (type(new_res) == np.ndarray) & (new_res.shape == (1, 640, 640, 3))

    
def test_estimate_az():
    """
    Test the detectAzimuth() function.

    Returns
    -------
    None.

    """
    #Pick the file name to read 
    img_file = "./examples/Panel_Detection_Examples/sat_img.png"
    #Create an instance of the PanelDetection() class.
    pc = pan_det.PanelDetection(model_file_path = './panel_segmentation/VGG16Net_ConvTranpose_complete.h5', 
                                classifier_file_path = './panel_segmentation/VGG16_classification_model.h5')
    #Read in the image
    x = imagex.load_img(img_file, 
                        color_mode='rgb', 
                        target_size=(640,640))
    x = np.array(x)
    #Mask the satellite image
    res = pc.testSingle(x.astype(float), test_mask=None,  model =None)
    #Crop the panels 
    new_res = pc.cropPanels(x, res)
    az = pc.detectAzimuth(new_res)
    #Assert that the azimut returned is an int instance
    assert_isinstance(az, int)


def test_plot_az():
    """
    Test the detectAzimuth() function.

    Returns
    -------
    None.

    """
    #Pick the file name to read 
    img_file = "./examples/Panel_Detection_Examples/sat_img.png"
    #Create an instance of the PanelDetection() class.
    pc = pan_det.PanelDetection(model_file_path = './panel_segmentation/VGG16Net_ConvTranpose_complete.h5', 
                                classifier_file_path = './panel_segmentation/VGG16_classification_model.h5')
    #Read in the image
    x = imagex.load_img(img_file, 
                        color_mode='rgb', 
                        target_size=(640,640))
    x = np.array(x)
    #Mask the satellite image
    res = pc.testSingle(x.astype(float), test_mask=None,  model =None)
    #Crop the panels 
    new_res = pc.cropPanels(x, res)
    pc.plotEdgeAz(new_res, 10, 1, 
                  save_img_file_path = "./tests/",
                  plot_show = True)
    #Open the image and assert that it exists
    im = PIL.Image.open("./tests/crop_mask_az_0.png")
    assert (type(im) == PIL.PngImagePlugin.PngImageFile)


def test_cluster_panels():
    """
    Test the clusterPanels() function.

    Returns
    -------
    None.

    """
    #Pick the file name to read 
    img_file = "./examples/Panel_Detection_Examples/sat_img.png"
    #Create an instance of the PanelDetection() class.
    pc = pan_det.PanelDetection(model_file_path = './panel_segmentation/VGG16Net_ConvTranpose_complete.h5', 
                                classifier_file_path = './panel_segmentation/VGG16_classification_model.h5')
    #Read in the image
    x = imagex.load_img(img_file, 
                        color_mode='rgb', 
                        target_size=(640,640))
    x = np.array(x)
    #Mask the satellite image
    res = pc.testSingle(x.astype(float), test_mask=None,  model =None)    
    #Use the mask to isolate the panels
    new_res = pc.cropPanels(x, res)
    number_arrays = 5
    clusters = pc.clusterPanels(new_res, res, 
                                number_arrays)
    azimuth_list = []
    for ii in np.arange(clusters.shape[0]):
        az = pc.detectAzimuth(clusters[ii][np.newaxis,:])
        azimuth_list.append(az)
    assert all(isinstance(x, int) for x in azimuth_list)
