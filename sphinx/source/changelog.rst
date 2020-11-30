
.. py:currentmodule:: panel_segmentation

Change Log
==========

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
