"""
Test suite for panel segmentation code.
"""
import os
import pytest
import numpy as np
from panel_segmentation import panel_detection as pan_det
from tensorflow.keras.preprocessing import image as imagex
import PIL

img_file = "./panel_segmentation/examples/Panel_Detection_Examples/sat_img.png"

def assert_isinstance(obj, klass):
    assert isinstance(obj, klass), f'got {type(obj)}, expected {klass}'
    
    
@pytest.fixture()
def panelDetectionClass():
    '''Generate an instance of the PanelDetection() class to run unit
    tests on.'''
    # Create an instance of the PanelDetection() class.
    pc = pan_det.PanelDetection(
        './panel_segmentation/VGG16Net_ConvTranpose_complete.h5',
        './panel_segmentation/VGG16_classification_model.h5',
        './panel_segmentation/object_detection_model.pth')
    return pc


@pytest.fixture()
def satelliteImg():
    '''Load in satellite image as fixture.'''
    # Read in the image
    x = imagex.load_img(img_file,
                        color_mode='rgb',
                        target_size=(640, 640))
    x = np.array(x)
    return x


def testGenerateSatelliteImage():
    with pytest.raises(ValueError):
        # Variables
        latitude = 39.7407
        longitude = -105.1694
        google_maps_api_key = "Wrong_API_key"
        file_name_save = "./examples/Panel_Detection_Examples/sat_img.png"
        model_file_path = os.path.abspath(
            './panel_segmentation/VGG16Net_ConvTranpose_complete.h5')
        assert os.path.exists(model_file_path)
        classifier_file_path = os.path.abspath(
            './panel_segmentation/VGG16_classification_model.h5')
        assert os.path.exists(classifier_file_path)
        mounting_classifier_model_path = os.path.abspath(
            './panel_segmentation/object_detection_model.pth')
        assert os.path.exists(mounting_classifier_model_path)
        # Create an instance of the PanelDetection() class.
        pc = pan_det.PanelDetection(
            model_file_path=model_file_path,
            classifier_file_path=classifier_file_path,
            mounting_classifier_file_path=mounting_classifier_model_path)
        pc.generateSatelliteImage(latitude, longitude,
                                  file_name_save, google_maps_api_key)


def testHasPanels(panelDetectionClass, satelliteImg):
    # Assert that the returned value is a boolean
    panel_loc = panelDetectionClass.hasPanels(satelliteImg)
    assert_isinstance(panel_loc, bool)


def testTestSingle(panelDetectionClass, satelliteImg):
    # Mask the satellite image
    res = panelDetectionClass.testSingle(satelliteImg.astype(float),
                                         test_mask=None,
                                         model=None)
    # Assert that the 'res' variable is a numpy array and the dimensions.
    assert_isinstance(res, np.ndarray)
    assert (res.shape == (640, 640))


def testCropPanels(panelDetectionClass, satelliteImg):
    # Mask the satellite image
    res = panelDetectionClass.testSingle(satelliteImg.astype(float),
                                         test_mask=None,
                                         model=None)
    # Crop the panels
    new_res = panelDetectionClass.cropPanels(satelliteImg, res)
    # Assert that the 'new_res' variable is a numpy array and the dimensions.
    assert_isinstance(new_res, np.ndarray)
    assert (new_res.shape == (1, 640, 640, 3))


def testDetectAzimuth(panelDetectionClass, satelliteImg):
    # Mask the satellite image
    res = panelDetectionClass.testSingle(satelliteImg.astype(float),
                                         test_mask=None,
                                         model=None)
    # Crop the panels
    new_res = pc.cropPanels(satelliteImg, res)
    az = pc.detectAzimuth(new_res)
    # Assert that the azimuth returned is a float instance
    assert_isinstance(az, float)


def testClassifyMountingConfiguration(panelDetectionClass,
                                      satelliteImg):
    (scores, labels, boxes) = \
        panelDetectionClass.classifyMountingConfiguration(
            img_file,
            acc_cutoff=.65,
            file_name_save=None)
    # Verify that we return 4 different labels, each
    # one associated with a carport installation
    assert (len(labels) == 4) & (len(scores) == 4) & (len(boxes) == 4)
    assert (all([label == 'carport-fixed' for label in labels]))
    # Assert that all scores associated with the labels are above .65
    assert(all([score > 0.65 for score in scores]))


def testPlotEdgeAz(panelDetectionClass, satelliteImg):
    # Mask the satellite image
    res = panelDetectionClass.testSingle(satelliteImg.astype(float),
                                         test_mask=None,
                                         model=None)
    # Crop the panels
    new_res = panelDetectionClass.cropPanels(satelliteImg, res)
    panelDetectionClass.plotEdgeAz(new_res, 10, 1,
                                   save_img_file_path=\
                                       "./panel_segmentation/tests/",
                                   plot_show=True)
    # Open the image and assert that it exists
    im = PIL.Image.open("./panel_segmentation/tests/crop_mask_az_0.png")
    assert_isinstance(im, PIL.PngImagePlugin.PngImageFile)


def testClusterPanels(panelDetectionClass, satelliteImg):
    # Mask the satellite image
    res = pc.testSingle(satelliteImg.astype(float),
                        test_mask=None,  model=None)
    # Use the mask to isolate the panels
    new_res = pc.cropPanels(satelliteImg, res)
    n, clusters = pc.clusterPanels(new_res)
    azimuth_list = []
    for ii in np.arange(clusters.shape[0]):
        az = pc.detectAzimuth(clusters[ii][np.newaxis, :])
        azimuth_list.append(az)
    assert all(isinstance(float(x), float) for x in azimuth_list)


def testRunSiteAnalysisPipeline(panelDetectionClass):
    site_analysis_dict = panelDetectionClass.runSiteAnalysisPipeline(
        file_name_save_img=img_file,
        file_name_save_mount=None,
        file_path_save_azimuth=None,
        generate_image=False)
    # Assert that a dictionary is returned with specific
    # attributes
    assert (type(site_analysis_dict) == dict)
    assert (all([label == 'carport-fixed' for label in
              site_analysis_dict["mounting_type"]]))
    assert (sorted(site_analysis_dict['associated_azimuths']) ==
            [90.0, 91.0, 161.0, 179.0])
