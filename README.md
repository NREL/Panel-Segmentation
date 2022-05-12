# Panel Segmentation

This repo contains the scripts for automated metadata extraction of solar PV installations, 
using satellite imagery coupled with computer vision techniques. In this package, the user
can perform the following actions:
- Automatically generate a satellite image using a set of lat-long coordinates, and a Google 
Maps API key. To get a Google Maps API key, go to the following site and set up an account:
https://developers.google.com/maps/documentation/javascript/get-api-key?utm_source=google&utm_medium=cpc&utm_campaign=FY20-Q3-global-demandgen-displayonnetworkhouseads-cs-GMP_maps_contactsal_saf_v2&utm_content=text-ad-none-none-DEV_c-CRE_460848633529-ADGP_Hybrid%20%7C%20AW%20SEM%20%7C%20BKWS%20~%20Google%20Maps%20API%20Key-KWID_43700035216023629-kwd-298247230705-userloc_1014524&utm_term=KW_google%20maps%20api%20key-ST_google%20maps%20api%20key&gclid=Cj0KCQjwit_8BRCoARIsAIx3Rj7XZb01kt1iLH3zzxGODvmM62g0K4ujEMpla5pL1p057tQXmp6MXpsaAscrEALw_wcB
- Determine the presence of a solar array in the satellite image (boolean True/False), using a 
classification model (VGG16_classification_model.h5).
- Perform image segmentation on the satellite image, to locate the solar array(s) in the 
image on a pixel-by-pixel basis, using an image segmentation model (VGG16Net_ConvTranpose_complete.h5).
- Using connected components clustering, isolate individual solar arrays in the masked image.
- Perform azimuth estimation on each solar array cluster in the masked image.
- Using an object detection model (Faster R-CNN Resnet 50 trained via transfer learning), detect
and classify mounting type and configuration of solar installations in satellite imagery. This includes
classification of fixed tilt, single-axis trackers, and double-axis trackers, as well as the rooftop, 
ground, and carport mounting configurations.

To install Panel-Segmentation, perform the following steps:

1. You must have Git large file storage (lfs) on your computer in order to download the deep learning models in this package. Go to the following site to download Git lfs: 

https://git-lfs.github.com/

2. Once git lfs is installed, you can now install Panel-Segmentation on your computer. We are still working on making panel-segmentation availble via PyPi, so entering the following in the command line will install the package locally on your computer:

pip install git+https://github.com/NREL/Panel-Segmentation.git@master#egg=panel-segmentation








