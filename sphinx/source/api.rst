.. currentmodule:: panel_segmentation

#############
API reference
#############


Classes
=======

These classes perform analyses used in the Panel-Segmentation project.

.. autosummary::
   :toctree: generated/

   panel_detection.PanelDetection
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
   panel_detection.PanelDetection.classifyMountingConfiguration

   panel_train.TrainPanelSegmentationModel
   panel_train.TrainPanelSegmentationModel.loadImagesToNumpyArray
   panel_train.TrainPanelSegmentationModel.diceCoeff
   panel_train.TrainPanelSegmentationModel.diceCoeffLoss
   panel_train.TrainPanelSegmentationModel.trainSegmentation
   panel_train.TrainPanelSegmentationModel.trainPanelClassifier
   panel_train.TrainPanelSegmentationModel.trainingStatistics
   panel_train.TrainPanelSegmentationModel.trainMountingConfigClassifier

Models
========

The following deep learning models are included in the Panel-Segmentation package.

.. autosummary::
   :toctree: generated/

   VGG16_classification_model.h5
   VGG16Net_ConvTranpose_complete.h5
