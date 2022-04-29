
.. py:currentmodule:: panel_segmentation

Change Log
==========
Version 0.0.2 (April 29, 2022)
--------------------------------

Added the mounting object detection algorithm to detect and classify the mounting
configuration of solar installations in satellite imagery. Updated several of the Github
workflows to provide more rigorous testing protocols (requirements.txt check and flake8 check).

Documentation
~~~~~~~~~~~~~
- Updated Sphinx Documentation to account for new functions.
- Updated the Jupyter notebooks to reflect pipeline changes: adding in the mounting configuration detection classifier and running it on satellite imagery.

Scripts
~~~~~~~~~~~~~
- Add the function trainMountingConfigClassifier() to the TrainPanelSegmentationModel() class.
- Add the functions runSiteAnalysisPipeline() and classifyMountingConfiguration() to the PanelDetection() class.
- Add unit testing for the trainMountingConfigClassifier(), runSiteAnalysisPipeline(), and trainMountingConfigClassifier() functions.

Other Changes
~~~~~~~~~~~~~
- Add Github workflow checks to include requirements.txt checks and flake8 checks.


Version 0.0.1 (October 27, 2020)
--------------------------------

Created initial package Panel-Segmentation for public release. 

Documentation
~~~~~~~~~~~~~
- Add Sphinx documentation. 
- Add commenting for each of the functions in the package.
- Add example Jupyter notebooks for panel detection and training the classifier and segmentation models.

Scripts
~~~~~~~~~~~~~
- Add PanelDetection class, where the user can generate a satellite image and run it through the pre-generated models.
- Add TrainPanelSegmentationModel() class, where the user can independently train segmentation and classifier models.
- Add unit testing for the PanelDetection() and TrainPanelSegmentationModel() classes, stored in the /tests/ folder. The pytest package was used.

Other Changes
~~~~~~~~~~~~~
- Add versioneer and setup scripts to perform pip installs of the package.
