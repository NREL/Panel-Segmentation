from panel_segmentation.panel_detection import PanelDetection

lat = 37.44648725156889
long = -122.26145462317662
key = "example_key"
p = PanelDetection()
p.get_classification_and_score_from_long_and_lat(lat, long, key)