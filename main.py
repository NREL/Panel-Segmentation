from panel_segmentation.panel_detection import PanelDetection
import os


lat = 37.44648725156889
long = -122.26145462317662
key = "example_key"
print(os.getcwd())
p = PanelDetection(
    model_file_path='./panel_segmentation/VGG16_classification_model.h5',
    classifier_file_path='./panel_segmentation/VGG16_classification_model.h5')
# p = PanelDetection()
p.get_classification_and_score_from_long_and_lat(lat, long, key)
