
"""
Test suite for utils code.
"""
import pytest
import numpy as np
from panel_segmentation import utils

img_file = "./panel_segmentation/examples/Panel_Detection_Examples/sat_img.png"


def testGenerateSatelliteImageTypeErrors():
    """
    Tests if TypeErrors is rasied for incorrect variable types.
    """
    # Proper variable types
    lat = 34.2897
    lon = -23.290
    filename = "an_image.png"
    api_key = "123456789ABCD"
    zoom = 19
    # Tests for float latitude
    with pytest.raises(TypeError):
        utils.generateSatelliteImage(1, lon, filename, api_key, zoom)
    # Tests for float longitude
    with pytest.raises(TypeError):
        utils.generateSatelliteImage(lat, "-12.34", filename, api_key, zoom)
    # Tests for int zoom level
    with pytest.raises(TypeError):
        utils.generateSatelliteImage(lat, lon, 290, api_key, zoom)
    # Tests for str file_name_save
    with pytest.raises(TypeError):
        utils.generateSatelliteImage(lat, lon, filename, 19.3, zoom)
    # Tests for str google_maps_api_key
    with pytest.raises(TypeError):
        utils.generateSatelliteImage(lat, lon, filename, api_key, 4.78)