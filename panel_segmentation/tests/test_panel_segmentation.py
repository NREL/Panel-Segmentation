"""
Test suite for panel segmentation code. 
"""
import os

import pytest
import pandas as pd
import numpy as np
from panel_segmentation import panel_detection as pan_det
from tensorflow.keras.preprocessing import image as imagex
import PIL
import h5py
import os


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
        print(os.getcwd())
        model_file_path = os.path.abspath('./panel_segmentation/VGG16Net_ConvTranpose_complete.h5')
        assert os.path.exists(model_file_path)
        classifier_file_path = os.path.abspath('./panel_segmentation/VGG16_classification_model.h5')
        assert os.path.exists(classifier_file_path)
        mounting_classifier_model_path = os.path.abspath('./panel_segmentation/object_detection_model.pth')
        assert os.path.exists(mounting_classifier_model_path)
        #Create an instance of the PanelDetection() class.
        pc = pan_det.PanelDetection(model_file_path = model_file_path, 
                                    classifier_file_path = classifier_file_path,
                                     mounting_classifier_file_path = mounting_classifier_model_path)
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
    img_file = "./panel_segmentation/examples/Panel_Detection_Examples/sat_img.png"
    #Create an instance of the PanelDetection() class.
    pc = pan_det.PanelDetection(model_file_path = './panel_segmentation/VGG16Net_ConvTranpose_complete.h5', 
                                classifier_file_path = './panel_segmentation/VGG16_classification_model.h5',
                                mounting_classifier_file_path = './panel_segmentation/object_detection_model.pth')
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
    img_file = "./panel_segmentation/examples/Panel_Detection_Examples/sat_img.png"
    #Create an instance of the PanelDetection() class.
    pc = pan_det.PanelDetection(model_file_path = './panel_segmentation/VGG16Net_ConvTranpose_complete.h5', 
                                classifier_file_path = './panel_segmentation/VGG16_classification_model.h5',
                                mounting_classifier_file_path = './panel_segmentation/object_detection_model.pth')
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
    img_file = "./panel_segmentation/examples/Panel_Detection_Examples/sat_img.png"
    #Create an instance of the PanelDetection() class.
    pc = pan_det.PanelDetection(model_file_path = './panel_segmentation/VGG16Net_ConvTranpose_complete.h5', 
                                classifier_file_path = './panel_segmentation/VGG16_classification_model.h5',
                                mounting_classifier_file_path = './panel_segmentation/object_detection_model.pth')
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
    img_file = "./panel_segmentation/examples/Panel_Detection_Examples/sat_img.png"
    #Create an instance of the PanelDetection() class.
    pc = pan_det.PanelDetection(model_file_path = './panel_segmentation/VGG16Net_ConvTranpose_complete.h5', 
                                classifier_file_path = './panel_segmentation/VGG16_classification_model.h5',
                                mounting_classifier_file_path = './panel_segmentation/object_detection_model.pth')
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
    assert_isinstance(az, float)
    
    
def test_classify_mounting_config():
    """
    Test the classifyMountingConfiguration() function.

    Returns
    -------
    None.

    """
    #Pick the file name to read 
    img_file = "./panel_segmentation/examples/Panel_Detection_Examples/sat_img.png"
    #Create an instance of the PanelDetection() class.
    pc = pan_det.PanelDetection(model_file_path = './panel_segmentation/VGG16Net_ConvTranpose_complete.h5', 
                                classifier_file_path = './panel_segmentation/VGG16_classification_model.h5',
                                mounting_classifier_file_path = './panel_segmentation/object_detection_model.pth')
    (scores, labels, boxes) = pc.classifyMountingConfiguration(img_file,
                                                               acc_cutoff = .65,
                                                               file_name_save = None)
    
    
    

def test_plot_az():
    """
    Test the detectAzimuth() function.

    Returns
    -------
    None.

    """
    #Pick the file name to read 
    img_file = "./panel_segmentation/examples/Panel_Detection_Examples/sat_img.png"
    #Create an instance of the PanelDetection() class.
    pc = pan_det.PanelDetection(model_file_path = './panel_segmentation/VGG16Net_ConvTranpose_complete.h5', 
                                classifier_file_path = './panel_segmentation/VGG16_classification_model.h5',
                                mounting_classifier_file_path = './panel_segmentation/object_detection_model.pth')
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
                  save_img_file_path = "./panel_segmentation/tests/",
                  plot_show = True)
    #Open the image and assert that it exists
    im = PIL.Image.open("./panel_segmentation/tests/crop_mask_az_0.png")
    assert (type(im) == PIL.PngImagePlugin.PngImageFile)


def test_cluster_panels():
    """
    Test the clusterPanels() function.

    Returns
    -------
    None.

    """
    #Pick the file name to read 
    img_file = "./panel_segmentation/examples/Panel_Detection_Examples/sat_img.png"
    #Create an instance of the PanelDetection() class.
    pc = pan_det.PanelDetection(model_file_path = './panel_segmentation/VGG16Net_ConvTranpose_complete.h5', 
                                classifier_file_path = './panel_segmentation/VGG16_classification_model.h5',
                                mounting_classifier_file_path = './panel_segmentation/object_detection_model.pth')
    #Read in the image
    x = imagex.load_img(img_file, 
                        color_mode='rgb', 
                        target_size=(640,640))
    x = np.array(x)
    #Mask the satellite image
    res = pc.testSingle(x.astype(float), test_mask=None,  model =None)    
    #Use the mask to isolate the panels
    new_res = pc.cropPanels(x, res)
    n, clusters = pc.clusterPanels(new_res)
    azimuth_list = []
    for ii in np.arange(clusters.shape[0]):
        az = pc.detectAzimuth(clusters[ii][np.newaxis,:])
        azimuth_list.append(az)
    assert all(isinstance(float(x), float) for x in azimuth_list)
    
    
def test_run_site_analysis_pipeline():
    """
    Test the runSiteAnalysisPipeline() function.

    Returns
    -------
    None.

    """
    #Pick the file name to read 
    img_file = "./panel_segmentation/examples/Panel_Detection_Examples/sat_img.png"
    latitude = 39.7407
    longitude = -105.1694
    google_maps_api_key =  "Wrong_API_key"  
    #Create an instance of the PanelDetection() class.
    pc = pan_det.PanelDetection(model_file_path = './panel_segmentation/VGG16Net_ConvTranpose_complete.h5', 
                                classifier_file_path = './panel_segmentation/VGG16_classification_model.h5',
                                mounting_classifier_file_path = './panel_segmentation/object_detection_model.pth')
    site_analysis_dict = pc.runSiteAnalysisPipeline(latitude,
                                                    longitude,
                                                    google_maps_api_key,
                                                    file_name_save_img = None, 
                                                    file_name_save_mount = None,
                                                    file_path_save_azimuth = None)