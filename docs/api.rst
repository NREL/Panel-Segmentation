.. currentmodule:: panel_segmentation

API Reference
=============
These classes and functions perform analyses used in the Panel-Segmentation project.

Main Panel Segmentation Pipeline
--------------------------------
Main panel segmentation pipeline used to detect solar panels in satellite imagery, using the pre-trained models provided in the Panel-Segmentation package.

Panel Detection
~~~~~~~~~~~~~~~
Generates satellite images and runs DL and CV routines on images to get array azimuth and mounting type/configuration.

.. autosummary::
   :toctree: generated/
   :caption: Panel Detection

   panel_detection.PanelDetection
   panel_detection.PanelDetection.generateSatelliteImage
   panel_detection.PanelDetection.classifyMountingConfiguration
   panel_detection.PanelDetection.diceCoeff
   panel_detection.PanelDetection.diceCoeffLoss
   panel_detection.PanelDetection.testBatch
   panel_detection.PanelDetection.testSingle
   panel_detection.PanelDetection.hasPanels
   panel_detection.PanelDetection.detectAzimuth
   panel_detection.PanelDetection.cropPanels
   panel_detection.PanelDetection.plotEdgeAz
   panel_detection.PanelDetection.clusterPanels
   panel_detection.PanelDetection.runSiteAnalysisPipeline

Panel Training
~~~~~~~~~~~~~~
Deep learning model training and development utilities.

.. autosummary::
   :toctree: generated/
   :caption: Panel Training

   panel_train.TrainPanelSegmentationModel
   panel_train.TrainPanelSegmentationModel.loadImagesToNumpyArray
   panel_train.TrainPanelSegmentationModel.diceCoeff
   panel_train.TrainPanelSegmentationModel.diceCoeffLoss
   panel_train.TrainPanelSegmentationModel.trainSegmentation
   panel_train.TrainPanelSegmentationModel.trainPanelClassifier
   panel_train.TrainPanelSegmentationModel.trainMountingConfigClassifier
   panel_train.TrainPanelSegmentationModel.trainingStatistics


Utilities
~~~~~~~~~
Helper functions and utilities.

.. autosummary::
   :toctree: generated/
   :caption: Utilities

   utils.generateSatelliteImage
   utils.generateAddress
   utils.generateSatelliteImageryGrid
   utils.visualizeSatelliteImageryGrid
   utils.splitTifToPngs
   utils.locateLatLonGeotiff
   utils.translateLatLongCoordinates
   utils.getInferenceBoxLatLonCoordinates
   utils.binaryMaskToPolygon
   utils.convertMaskToLatLonPolygon
   utils.convertPolygonToGeojson

LiDAR
-----
LiDAR data processing utilities.

.. TODO: ADD LIDAR FUNCTIONS TO THE BELOW AFTER LIDAR PR IS MERGED

.. Point Cloud Data (PCD) Processing
.. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. Processing functions for Point Cloud Data (PCD) files.

.. .. autosummary::
..    :toctree: generated/
..    :caption: Point Cloud Data (PCD) Processing
   
..    lidar.pcd_data.PCD

.. Plane Segmentation
.. ~~~~~~~~~~~~~~~~~~
.. Plane segmentation functions that segments planes from point cloud data.

.. .. autosummary::
..    :toctree: generated/
..    :caption: Plane Segmentation
   
..    lidar.plane_segmentation.planeSegmentation

.. USGS LiDAR API
.. ~~~~~~~~~~~~~~
.. API for downloading, processing, and filtering USGS LiDAR data.

.. .. autosummary::
..    :toctree: generated/
..    :caption: USGS LiDAR API
   
..    lidar.usgs_lidar_api.USGSLidarAPI

Models
======

The following deep learning models are included in the Panel-Segmentation package:

Panel Detection Models
----------------------
* **VGG16_classification_model.h5**: This is the DL classifier model, which identifies if a solar array is detected in an image.
* **VGG16Net_ConvTranpose_complete.h5**: This is the DL instance segmentation model, which identifies solar arrays in the image on a pixel-by-pixel basis.
* **object_detection_model.pth**: This is the DL object detection model, which detects and classifies solar array mounting configuration.
* **sol_searcher_config.py**: This is the configuration file for the DL object detection sol-searcher model.
* **sol_searcher_model.pth**: This is the checkpoint file for the DL object detection sol-searcher model, which searches for solar panels in satellite imagery. This model is trained on 3783 images from Google Maps imagery of the Austin, TX area from November 2023 and Denver, CO area from June 2023 with a resolution of 0.2986 meters per pixel. The architecture of the model is RTMDet-X with a mAP-50 score of 0.884.

Extreme Weather: Hail Models
----------------------------
* **hail_config.py**: This is the configuration file for the DL instance segmentation hail model.
* **hail_model.pth**: This is the checkpoint file for the DL instance segmentation hail model, which detects hail on solar arrays in satellite imagery. This model is trained on 1883 images from Google Maps imagery of the Austin, TX area from November 2023 with a resolution of 0.0746 meters per pixel. The architecture of the model is RTMDet-Ins-X with a mAP-50 score of 0.859.

Extreme Weather: Hurricane Models
---------------------------------
* **pre_hurricane_config.py**: This is the configuration file for the DL instance segmentation pre-hurricane model.
* **pre_hurricane_model.pth**: This is the checkpoint file for the DL instance segmentation pre-hurricane model, which detects solar arrays in pre-hurricane satellite imagery. This model is trained on 1883 images from Google Maps imagery of various areas before hurricane impact with a resolution of 0.0746 meters per pixel. Many of these images includes Puerto Rico. The architecture of this model is RTMDet-Ins-l with a mAP-50 score of 0.942.
* **post_hurricane_config.py**: This is the configuration file for the DL instance segmentation post-hurricane model.
* **post_hurricane_model.pth**: This is the checkpoint file for the DL instance segmentation post-hurricane model, which detects solar arrays in post-hurricane satellite imagery. This model is trained on 863 images from NOAA post-Hurricane Maria satellite imagery of the Puerto Rico area with a resolution of 0.0746 meters per pixel. The architecture of this model is Mask-RCNN X-101-64x4d-FPN with a mAP-50 score of 0.844.
