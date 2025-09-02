
Getting Started
===============

This page documents how to install the Panel Segmentation package and run 
automated metadata extraction for a PV array at a specified location. 
These instructions assume that you already have Anaconda and git installed. 


Installation Guide
-------------------
To install Panel-Segmentation, perform the following steps:

1. You must have Git large file storage (lfs) on your computer in order to download the deep learning models in this package. Go to the following site to download Git lfs: 
    
    https://git-lfs.github.com/

2. Once git lfs is installed, you can now install Panel-Segmentation on your computer. We are still working on making panel-segmentation available via PyPi, so entering the following in the command line will install the package locally on your computer:
    
    pip install git+https://github.com/NREL/Panel-Segmentation.git@master#egg=panel-segmentation

3. Panel-Segmentation requires the MMCV package, which can be tricky to install for CPU-only, and needs to be installed from source. To install MMCV for source, run the following in the command line:
    
    pip install git+https://github.com/open-mmlab/mmcv.git@v2.1.0

4. When initiating the PanelDetection() class, be sure to point your file paths to the model paths in your local Panel-Segmentation folder!

Please note that installations will likely take several minutes.
