# Panel Segmentation

This repo contains the scripts for automated metadata extraction of solar PV installations, 
using satellite imagery coupled with computer vision techniques. In this package, the user
can perform the following actions:
-Automatically generate a satellite image using a set of lat-long coordinates, and a Google 
Maps API key. To get a Google Maps API key, go to the following site and set up an account:
https://developers.google.com/maps/documentation/javascript/get-api-key?utm_source=google&utm_medium=cpc&utm_campaign=FY20-Q3-global-demandgen-displayonnetworkhouseads-cs-GMP_maps_contactsal_saf_v2&utm_content=text-ad-none-none-DEV_c-CRE_460848633529-ADGP_Hybrid%20%7C%20AW%20SEM%20%7C%20BKWS%20~%20Google%20Maps%20API%20Key-KWID_43700035216023629-kwd-298247230705-userloc_1014524&utm_term=KW_google%20maps%20api%20key-ST_google%20maps%20api%20key&gclid=Cj0KCQjwit_8BRCoARIsAIx3Rj7XZb01kt1iLH3zzxGODvmM62g0K4ujEMpla5pL1p057tQXmp6MXpsaAscrEALw_wcB
- Determine the presence of a solar array in the satellite image (boolean True/False), using a 
classification model (VGG16_classification_model.h5).
- Perform image segmentation on the satellite image, to locate the solar array(s) in the 
image on a pixel-by-pixel basis, using an image segmentation model (VGG16Net_ConvTranpose_complete.h5).
- Using spectral clustering, isolate individual solar arrays in the masked image.
- Perform azimuth estimation on each solar array cluster in the masked image.





