"""
Main/driver script 
"""

from panel_segmentation import panel_detection as pseg
import numpy as np
from tensorflow.keras.preprocessing import image as imagex
import matplotlib.pyplot as plt

#from scipy import misc

if __name__ == "__main__":
    #Example latitude-longitude coordinates to run the analysis on.
    latitude = 39.7407
    longitude = -105.1694
    google_maps_api_key =  "AIzaSyC5K2ywbBieNx-UWYl6e3Y9g5dH4JxrnpY"    
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
    
    file_name_save = "C:/Users/kperry/Downloads/image_33.98123_-116.517359_0.png"
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
    
    number_arrays = 4
    clusters = panelseg.clusterPanels(new_res, res, 
                                      number_arrays)
    
    for ii in np.arange(clusters.shape[0]):
        az = panelseg.detectAzimuth(clusters[ii][np.newaxis,:])
        print(az)
     
     
    #Then we can find azimuth for each cluster
    panelseg.plotEdgeAz(clusters, 10, 1,
                         save_img_file_path = './')
    
    
    
    
    
