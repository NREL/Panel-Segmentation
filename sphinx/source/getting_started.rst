
Getting Started
===============

This page documents how to install the Panel Segmentation package and run 
automated metadata extraction for a PV array at a specified location. 
These instructions assume that you already have Anaconda and git installed. 


Installing
----------

First, clone the `Panel-Segmentation <https://github.com/NREL/Panel-Segmentation>`_
repository to your computer with git. This will bring the source code of the 
package locally to your computer.

First, create a new conda environment and activate it (optional)::

    conda create -n panel-segmentation-dev python=3.7
    conda activate panel-segmentation-dev

Now you should change the working directory to the Panel-Segmentation repository folder
and install the package::

    pip install .

If you want to use the precise package versions used in the example notebooks,
you can use the ``requirements.txt`` file::

    pip install -r requirements.txt

Now you should be able to import `panel_segmentation` in a python terminal

.. code-block:: python

    import panel_segmentation

The recommended way of running the code is through a normal python terminal
so that the conda environment is kept clean, but if you want, you can install
spyder in the environment too.  Note that you'll have to start spyder in a
terminal with the conda environment activated for it to have access to the
packages we just installed.


Running a single system
-----------------------

A satellite image analysis is performed using the 
:py:class:`panel_detection.PanelDetection`
class. This class allows the user to generate a satellite image 
based on a set of latitude-longitude coordinates, run the satellite image
through a image segmentation model to determine presence of a solar on a 
pixel-by-pixel basis, cluster individual solar arrays in an 
image, and estimate the azimuth of each detected solar array.

.. code-block:: python

    from panel_segmentation.panel_detection import PanelDetection
    from panel_segmentation import panel_detection as pseg
    import numpy as np
    from tensorflow.keras.preprocessing import image as imagex
    import matplotlib.pyplot as plt
    
    #Example latitude-longitude coordinates to run the analysis on.
    latitude = 39.7407
    longitude = -105.1694
    google_maps_api_key =  "YOUR API KEY HERE"    
    file_name_save = "sat_img.png"
    
    #CREATE AN INSTANCE OF THE PANELDETECTION CLASS TO RUN THE ANALYSIS
    panelseg = pseg.PanelDetection(model_file_path ='./panel_segmentation/VGG16Net_ConvTranpose_complete.h5', 
                                    classifier_file_path ='./panel_segmentation/VGG16_classification_model.h5')

    #GENERATE A SATELLITE IMAGE USING THE ASSOCIATED LAT-LONG COORDS AND THE GOOGLE
    #MAPS API KEY
    img = panelseg.generateSatelliteImage(latitude, longitude,
                                       file_name_save,
                                       google_maps_api_key)
    #Show the generated satellite image
    plt.imshow(img)
    
    #LOAD THE IMAGE AND DECLARE AS A NUMPY ARRAY
    x = imagex.load_img(file_name_save, 
                        color_mode='rgb', 
                        target_size=(640,640))
    x = np.array(x)
    
    #USE CLASSIFIER MODEL TO DETERMINE IF A SOLAR ARRAY HAS BEEN DETECTED IN THE 
    #IMAGE
    panel_loc = panelseg.hasPanels(x)
    
    #Mask the satellite image
    res = panelseg.testSingle(x.astype(float), test_mask=None,  model =None)    
    #Use the mask to isolate the panels
    new_res = panelseg.cropPanels(x, res)
    plt.imshow(new_res.reshape(640,640,3))
    
    #check azimuth 
    az = panelseg.detectAzimuth(new_res)
    
    #plot edges + azimuth 
    panelseg.plotEdgeAz(new_res,10, 1,
                         save_img_file_path = './')
        
    #PERFORM AZIMUTH ESTIMATION FOR MULTIPLE CLUSTERS
    #Cluster panels in an image. The image to be passed are the "isolated panels", 
    #mask and number of clusters 
    
    number_arrays = 5
    clusters = panelseg.clusterPanels(new_res, res, 
                                      number_arrays)
    
    for ii in np.arange(clusters.shape[0]):
        az = panelseg.detectAzimuth(clusters[ii][np.newaxis,:])
        print(az)
     
     
    #Then we can find azimuth for each cluster
    panelseg.plotEdgeAz(clusters, 10, 1,
                         save_img_file_path = './')
    
    


