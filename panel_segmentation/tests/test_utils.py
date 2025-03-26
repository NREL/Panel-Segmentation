"""
Test suite for utils code.
"""

from panel_segmentation import utils
import os
from PIL import Image
import requests
import pytest
import numpy as np
from mpl_toolkits.axes_grid1.axes_grid import ImageGrid
import json

example_path = os.path.join("panel_segmentation", "examples", "utils_examples")


@pytest.fixture
def satellite_image_params():
    """
    Contains the proper variable types for running GenerateSatelliteImage
    and generate_address functions.
    """
    # Proper variable types
    lat = 37.18385
    lon = -113.70663
    file_name_save = os.path.join(example_path,
                                  "satellite_img")
    api_key = "123456789ABC"
    zoom = 19
    return lat, lon, file_name_save, api_key, zoom


@pytest.fixture
def satellite_image_grid_params():
    """
    Contains the proper variable types for running
    generate_satellite_imagery_grid function.
    """
    # Proper variable types
    nw_lat = 37.18385
    nw_lon = -113.70663
    se_lat = 37.17385
    se_lon = -113.69663
    api_key = "123456789ABC"
    file_save_folder = example_path
    zoom = 19
    lat_lon_dist = 0.01
    num_allowed_img = 2
    return nw_lat, nw_lon, se_lat, se_lon, api_key, file_save_folder, zoom, \
        lat_lon_dist, num_allowed_img


@pytest.fixture
def vis_satellite_img_grid_params():
    """
    Contains the proper variable types for running
    visualize_satellite_imagery_grid function.
    """
    # Proper variable types
    grid_location_list = [
        {'file_name': '37.18385_-113.70663.png', 'latitude': 37.18385,
            'lon': -113.70663, 'grid_x': 0, 'grid_y': 0},
        {'file_name': '37.17385_-113.70663.png', 'latitude': 37.17385,
            'lon': -113.70663, 'grid_x': 1, 'grid_y': 0},
        {'file_name': '37.18385_-113.69663.png', 'latitude': 37.18385,
            'lon': -113.69663, 'grid_x': 0, 'grid_y': 1},
        {'file_name': '37.18385_-113.68663.png', 'latitude': 37.18385,
            'lon': -113.68663, 'grid_x': 0, 'grid_y': 2}
    ]
    file_save_folder = os.path.join(example_path,
                                    "satellite_grid")
    return grid_location_list, file_save_folder


@pytest.fixture
def split_tif_to_pngs_params():
    """
    Contains the proper variable types for running split_tif_to_png
    function
    """
    geotiff_file = os.path.join(example_path,
                                "40.1072_-75.0137.tif")
    meters_per_pixel = 0.152401
    meters_png_image = 2500.0
    file_save_folder = example_path
    return geotiff_file, meters_per_pixel, meters_png_image, file_save_folder


@pytest.fixture
def locate_lat_lon_geotiff_params():
    """
    Contains the proper variable types for running locate_lat_lon_geotiff
    function
    """
    geotiff_file = os.path.join(example_path, "40.1072_-75.0137.tif")
    latitude = 40.1072
    longitude = -75.0137
    file_name_save = os.path.join(example_path, "40.1072_-75.0137.png")
    pixel_resolution = 600
    return geotiff_file, latitude, longitude, file_name_save, pixel_resolution


@pytest.fixture
def translate_lat_long_coords_params():
    """
    Contains the proper variable types for running
    translate_lat_long_coordinates function
    """
    latitude = 40.1072
    longitude = -75.0137
    lat_translation_meters = 100.0
    long_translation_meters = -200.0
    return latitude, longitude, lat_translation_meters, long_translation_meters


@pytest.fixture
def get_inference_box_lat_lon_coords_params():
    """
    Contains the proper variable types for running
    get_inference_box_lat_lon_coordinates function.
    """
    box = [15.0, 0.0, 485.0, 500.0]
    img_center_lat = 40.1072
    img_center_lon = -75.0137
    image_x_pixels = 600
    image_y_pixels = 600
    zoom_level = 18
    return box, img_center_lat, img_center_lon, image_x_pixels, \
        image_y_pixels, zoom_level


@pytest.fixture
def convert_mask_to_lat_lon_polygon_params():
    """
    Contains the proper variable types for running
    convert_mask_to_lat_lon_polygon function.
    """
    mask = np.zeros((4, 4))
    mask[1:3, 1:3] = 1
    img_center_lat = 40.1072
    img_center_lon = -75.0137
    image_x_pixels = 600
    image_y_pixels = 600
    zoom_level = 18
    return mask, img_center_lat, img_center_lon, image_x_pixels, \
        image_y_pixels, zoom_level


def test_generate_satellite_image_type_errors(satellite_image_params):
    """
    Tests if TypeErrors is rasied for incorrect variable types.
    """
    lat, lon, file_name_save, api_key, zoom = satellite_image_params
    # Tests for float latitude type
    with pytest.raises(TypeError,
                       match="latitude variable must be of type float."):
        utils.generateSatelliteImage(1, lon, file_name_save, api_key, zoom)
    # Tests for float longitude type
    with pytest.raises(TypeError,
                       match="longitude variable must be of type float."):
        utils.generateSatelliteImage(
            lat, "-12.34", file_name_save, api_key, zoom)
    # Tests for int zoom level type
    with pytest.raises(TypeError,
                       match="zoom_level variable must be of type int."):
        utils.generateSatelliteImage(lat, lon, file_name_save, api_key, True)
    # Tests for str file_name_save type
    with pytest.raises(
            TypeError,
            match="file_name_save variable must be of type string."):
        utils.generateSatelliteImage(lat, lon, 19.3, api_key, zoom)
    # Tests for str google_maps_api_key type
    with pytest.raises(
            TypeError,
            match="google_maps_api_key variable must be of type string."):
        utils.generateSatelliteImage(lat, lon, file_name_save, 4, zoom)


def test_generate_satellite_image_error_response(satellite_image_params,
                                                 mocker):
    """
    Tests if requests response returns the correct output for non-200
    status code.
    """
    lat, lon, file_name_save, api_key, zoom = satellite_image_params
    # Simulate mocked status code response of 404
    mocked_requests_get = mocker.patch("requests.get")
    mocked_response = mocker.Mock(spec=requests.Response)
    mocked_response.status_code = 404
    mocked_requests_get.return_value = mocked_response
    error_string = ("Response status code 404: "
                    "Image not pulled successfully from API.")
    # Test is ValueErroor is raised if status code is 404
    with pytest.raises(ValueError, match=error_string):
        utils.generateSatelliteImage(lat, lon, file_name_save, api_key, zoom)


def test_generate_satellite_image_200_response(satellite_image_params, mocker):
    """
    Tests if requests response returns the correct output for 200 status code.
    """
    lat, lon, file_name_save, api_key, zoom = satellite_image_params
    # Simulate mocked response of code 200 and returned image data
    mocked_requests_get = mocker.patch("requests.get")
    mocked_response = mocker.Mock(spec=requests.Response)
    mocked_response.status_code = 200
    mocked_response.content = b"satellite_img"
    mocked_requests_get.return_value = mocked_response
    # Mock opened file when open() is called
    mocked_file = mocker.mock_open()
    mocker.patch("panel_segmentation.utils.open", mocked_file)
    # Mock opened PIL Image
    mocked_img = mocker.Mock(spec=Image.Image)
    mocker.patch("PIL.Image.open", return_value=mocked_img)
    # Test function with mocked parameters
    actual_pulled_img = utils.generateSatelliteImage(
        lat, lon, file_name_save, api_key, zoom)
    # Aseert that the file is correctly written with mocked content
    mocked_file().write.assert_called_once_with(b"satellite_img")
    # Assert the pulled image returned the mocked image
    assert actual_pulled_img == mocked_img


def test_generate_address_type_errors(satellite_image_params):
    """
    Tests if TypeErrors is rasied for incorrect variable types.
    """
    lat, lon, _, api_key, _ = satellite_image_params
    # Tests for float latitude type
    with pytest.raises(TypeError,
                       match="latitude variable must be of type float."):
        utils.generate_address(1, lon, api_key)
    # Tests for float longitude type
    with pytest.raises(TypeError,
                       match="longitude variable must be of type float."):
        utils.generate_address(lat, "-12.34", api_key)
    # Tests for str google_maps_api_key type
    with pytest.raises(
            TypeError,
            match="google_maps_api_key variable must be of type string."):
        utils.generate_address(lat, lon, 4)


def test_generate_address_error_response(satellite_image_params, mocker):
    """
    Tests if requests response returns the correct output for non-200
    status code.
    """
    lat, lon, _, api_key, _ = satellite_image_params
    # Simulate mocked status code response of 404
    mocked_requests_get = mocker.patch("requests.get")
    mocked_response = mocker.Mock(spec=requests.Response)
    mocked_response.status_code = 404
    mocked_requests_get.return_value = mocked_response
    error_string = ("Response status code 404: "
                    "Address not pulled successfully from API.")
    # Test is ValueErroor is raised if status code is 404
    with pytest.raises(ValueError, match=error_string):
        utils.generate_address(lat, lon, api_key)


def test_generate_address_image_200_response(satellite_image_params, mocker):
    """
    Tests if requests response returns the correct output for 200 status code.
    """
    lat, lon, _, api_key, _ = satellite_image_params
    expected_address = "875 Coyote Gulch Ct, Ivins, UT 84738, USA"
    # Simulate mocked response of code 200 and returned str address
    mocked_requests_get = mocker.patch("requests.get")
    mocked_response = mocker.Mock(spec=requests.Response)
    mocked_response.status_code = 200
    mocked_response.json.return_value = {"results": [
        {"formatted_address": expected_address}
    ]}
    mocked_requests_get.return_value = mocked_response
    # Test function with mocked parameters
    actual_address = utils.generate_address(lat, lon, api_key)
    assert actual_address == expected_address
    # Assert address is a string
    assert isinstance(actual_address, str)


def test_generate_satellite_imagery_grid_type_errors(
        satellite_image_grid_params):
    """
    Tests if TypeErrors is rasied for incorrect variable types.
    """
    nw_lat, nw_lon, se_lat, se_lon, api_key, file_save_folder, zoom, \
        lat_lon_dist, num_allowed_img = satellite_image_grid_params
    # Tests for float northwest_latitude type
    with pytest.raises(
            TypeError,
            match="northwest_latitude variable must be of type float."):
        utils.generate_satellite_imagery_grid(1, nw_lon, se_lat, se_lon,
                                              api_key, zoom, file_save_folder,
                                              lat_lon_dist, num_allowed_img)
    # Tests for float northwest_longitude type
    with pytest.raises(
            TypeError,
            match="northwest_longitude variable must be of type float."):
        utils.generate_satellite_imagery_grid(nw_lat, "13", se_lat, se_lon,
                                              api_key, zoom, file_save_folder,
                                              lat_lon_dist, num_allowed_img)
    # Tests for float southeast_latitude type
    with pytest.raises(
            TypeError,
            match="southeast_latitude variable must be of type float."):
        utils.generate_satellite_imagery_grid(nw_lat, nw_lon, True, se_lon,
                                              api_key, file_save_folder, zoom,
                                              lat_lon_dist, num_allowed_img)
    # Tests for float southeast_longitude type
    with pytest.raises(
            TypeError,
            match="southeast_longitude variable must be of type float."):
        utils.generate_satellite_imagery_grid(nw_lat, nw_lon, se_lat, 21,
                                              api_key, file_save_folder, zoom,
                                              lat_lon_dist, num_allowed_img)
    # Tests for str google_maps_api_key type
    with pytest.raises(
            TypeError,
            match="google_maps_api_key variable must be of type string."):
        utils.generate_satellite_imagery_grid(nw_lat, nw_lon, se_lat, se_lon,
                                              123, file_save_folder, zoom,
                                              lat_lon_dist, num_allowed_img)
    # Tests for str file_save_folder type
    with pytest.raises(
            TypeError,
            match="file_save_folder variable must be of type string."):
        utils.generate_satellite_imagery_grid(nw_lat, nw_lon, se_lat, se_lon,
                                              api_key, 902, zoom,
                                              lat_lon_dist, num_allowed_img)
    # Tests for int zoom_level type
    with pytest.raises(
            TypeError,
            match="zoom_level variable must be of type int."):
        utils.generate_satellite_imagery_grid(nw_lat, nw_lon, se_lat, se_lon,
                                              api_key, file_save_folder, False,
                                              lat_lon_dist, num_allowed_img)
    # Tests for float lat_lon_distance type
    with pytest.raises(
            TypeError,
            match="lat_lon_distance variable must be of type float."):
        utils.generate_satellite_imagery_grid(nw_lat, nw_lon, se_lat, se_lon,
                                              api_key, file_save_folder, zoom,
                                              "lat_lon_dist", num_allowed_img)
    # Tests for int number_allowed_images_taken type
    with pytest.raises(
            TypeError,
            match="number_allowed_images_taken variable must be of type int."):
        utils.generate_satellite_imagery_grid(nw_lat, nw_lon, se_lat, se_lon,
                                              api_key, file_save_folder, zoom,
                                              lat_lon_dist, 9.1)


def test_generate_satellite_imagery_grid_output(
        satellite_image_grid_params, mocker):
    """
    Tests if generate_satellite_imagery_grid returns the correct output for
    sateliite images.
    """
    nw_lat, nw_lon, se_lat, se_lon, api_key, file_save_folder, zoom, \
        lat_lon_dist, num_allowed_img = satellite_image_grid_params

    # Mock satellite file pull from generateSatelliteImage function
    mocked_pull = mocker.patch(
        "panel_segmentation.utils.generateSatelliteImage")

    actual_output = utils.generate_satellite_imagery_grid(
        nw_lat, nw_lon, se_lat, se_lon, api_key, file_save_folder, zoom,
        lat_lon_dist, num_allowed_img
    )
    expected_output = [
        {'file_name': '37.18385_-113.70663.png', 'latitude': 37.18385,
            'lon': -113.70663, 'grid_x': 0, 'grid_y': 0},
        {'file_name': '37.17385_-113.70663.png', 'latitude': 37.17385,
            'lon': -113.70663, 'grid_x': 1, 'grid_y': 0},
        {'file_name': '37.18385_-113.69663.png', 'latitude': 37.18385,
            'lon': -113.69663, 'grid_x': 0, 'grid_y': 1},
        {'file_name': '37.18385_-113.68663.png', 'latitude': 37.18385,
            'lon': -113.68663, 'grid_x': 0, 'grid_y': 2}
    ]
    # Assert that mocked expected output equals to actual output
    assert actual_output == expected_output
    # Assert that the output is a list
    assert isinstance(actual_output, list)
    # Assert that the call count is equal to the numbers of files pulled
    assert mocked_pull.call_count == len(expected_output)


def test_visualize_satellite_imagery_grid_type_errors(
        vis_satellite_img_grid_params):
    """
    Tests if TypeErrors is rasied for incorrect variable types.
    """
    grid_location_list, file_save_folder = vis_satellite_img_grid_params
    # Tests for grid_location_list list type
    with pytest.raises(
            TypeError,
            match="grid_location_list variable must be of type list."):
        utils.visualize_satellite_imagery_grid(
            {'file_name': '37.17385_-113.70663.png'}, file_save_folder)
    # Tests for values within grid_location_list are dict type
    with pytest.raises(
            TypeError,
            match="grid_location_list must be a list of dictionaries."):
        utils.visualize_satellite_imagery_grid([1, 2, 3, 4],
                                               file_save_folder)
    # Tests for file_str_folder str type
    with pytest.raises(
            TypeError, match="file_save_folder variable must be of type str."):
        utils.visualize_satellite_imagery_grid(grid_location_list,
                                               False)


def test_visualize_satellite_imagery_grid_output(
        vis_satellite_img_grid_params):
    """
    Tests if visualize_satellite_imagery_grid returns the correct output.
    """
    grid_location_list, file_save_folder = vis_satellite_img_grid_params
    grid = utils.visualize_satellite_imagery_grid(grid_location_list,
                                                  file_save_folder)
    # Assert grid is of ImageGrid (plot of gridded satellite images)
    assert isinstance(grid, ImageGrid)


def test_split_tif_to_pngs_type_errors(
        split_tif_to_pngs_params):
    """
    Tests if TypeErrors is rasied for incorrect variable types.
    """
    geotiff_file, meters_per_pixel, meters_png_image, \
        file_save_folder = split_tif_to_pngs_params
    # Tests for geotiff_file str type
    with pytest.raises(
            TypeError, match="geotiff_file variable must be of type str."):
        utils.split_tif_to_pngs(["a_geotiff_file.tif"], meters_per_pixel,
                                meters_png_image, file_save_folder)
    # Tests for meters_per_pixel float type
    with pytest.raises(
            TypeError,
            match="meters_per_pixel variable must be of type float."):
        utils.split_tif_to_pngs(geotiff_file, 871,
                                meters_png_image, file_save_folder)
    # Tests for meters_png_image float type
    with pytest.raises(
            TypeError,
            match="meters_png_image variable must be of type float."):
        utils.split_tif_to_pngs(geotiff_file, meters_per_pixel,
                                "201", file_save_folder)
    # Tests for file_save_folder str type
    with pytest.raises(
            TypeError, match="file_save_folder variable must be of type str."):
        utils.split_tif_to_pngs(geotiff_file, meters_per_pixel,
                                meters_png_image, False)


def test_split_tif_to_pngs_type_output(split_tif_to_pngs_params, mocker):
    """
    Tests if split_tif_to_pngs_type function returns None.
    """
    geotiff_file, meters_per_pixel, meters_png_image, \
        file_save_folder = split_tif_to_pngs_params
    actual_output = utils.split_tif_to_pngs(
        geotiff_file, meters_per_pixel,
        meters_png_image, file_save_folder)
    # Delete the produced image
    os.remove(os.path.join(file_save_folder,
                           "40.0990931_-75.0039022.png"))
    assert actual_output is None


def test_locate_lat_lon_geotiff_image_type_errors(
        locate_lat_lon_geotiff_params):
    """
    Tests if TypeErrors is rasied for incorrect variable types.
    """
    geotiff_file, latitude, longitude, file_name_save, \
        pixel_resolution = locate_lat_lon_geotiff_params
    # Tests for geotiff_file str type
    with pytest.raises(
            TypeError, match="geotiff_file variable must be of type str."):
        utils.locate_lat_lon_geotiff(-231, latitude, longitude,
                                     file_name_save, pixel_resolution)
    # Tests for latitude float type
    with pytest.raises(
            TypeError, match="latitude variable must be of type float."):
        utils.locate_lat_lon_geotiff(geotiff_file, [3219, 31], longitude,
                                     file_name_save, pixel_resolution)
    # Tests for longitude float type
    with pytest.raises(
            TypeError, match="longitude variable must be of type float."):
        utils.locate_lat_lon_geotiff(geotiff_file, latitude, "1093",
                                     file_name_save, pixel_resolution)
    # Tests for file_name_save str type
    with pytest.raises(
            TypeError, match="file_name_save variable must be of type str."):
        utils.locate_lat_lon_geotiff(geotiff_file, latitude, longitude,
                                     True, pixel_resolution)
    # Tests for pixel_resolution int type
    with pytest.raises(
            TypeError, match="pixel_resolution variable must be of type int."):
        utils.locate_lat_lon_geotiff(geotiff_file, latitude, longitude,
                                     file_name_save, 678.1)


def test_locate_lat_lon_geotiff_image_output(locate_lat_lon_geotiff_params):
    """
    Tests if the output returns an image output for when latitude and longitude
    is within bounds to capture the png image.
    """
    geotiff_file, latitude, longitude, file_name_save, \
        pixel_resolution = locate_lat_lon_geotiff_params
    actual_output = utils.locate_lat_lon_geotiff(
        geotiff_file, latitude, longitude, file_name_save, pixel_resolution)
    # Assert that actual_output returns an image
    assert isinstance(actual_output, Image.Image)


def test_locate_lat_lon_geotiff_none_output(locate_lat_lon_geotiff_params,
                                            capsys):
    """
    Tests if the output returns None for when latitude and longitude is out of
    bounds.
    """
    geotiff_file, _, _, file_name_save, \
        pixel_resolution = locate_lat_lon_geotiff_params
    # Make the latitude and longitude out of image bounds
    latitude = 39.0662
    longitude = -121.1606
    actual_output = utils.locate_lat_lon_geotiff(
        geotiff_file, latitude, longitude, file_name_save, pixel_resolution)
    captured = capsys.readouterr()
    print_msg = ("""Latitude-longitude coordinates are not within bounds of
                the image, no PNG captured...\n""")
    # Assert that print_msg gets printed
    assert captured.out == print_msg
    # Assert that actual output returns None
    assert actual_output is None


def test_translate_lat_long_coordinates_type_errors(
        translate_lat_long_coords_params):
    """
    Tests if TypeErrors is rasied for incorrect variable types.
    """
    latitude, longitude, lat_translation_meters, \
        long_translation_meters = translate_lat_long_coords_params
    # Tests for latitude float type
    with pytest.raises(
            TypeError, match="latitude variable must be of type float."):
        utils.translate_lat_long_coordinates(91, longitude,
                                             lat_translation_meters,
                                             long_translation_meters)
    # Tests for longitude float type
    with pytest.raises(
            TypeError, match="longitude variable must be of type float."):
        utils.translate_lat_long_coordinates(latitude, "191",
                                             lat_translation_meters,
                                             long_translation_meters)
    # Tests for lat_translation_meters float type
    with pytest.raises(
            TypeError,
            match="lat_translation_meters variable must be of type float."):
        utils.translate_lat_long_coordinates(latitude, longitude,
                                             True, long_translation_meters)
    # Tests for long_translation_meters float type
    with pytest.raises(
            TypeError,
            match="long_translation_meters variable must be of type float."):
        utils.translate_lat_long_coordinates(latitude, longitude,
                                             lat_translation_meters, ["5"])


def test_translate_lat_long_coordinates_type_output(
        translate_lat_long_coords_params):
    """
    Tests the if the output returns float values.
    """
    latitude, longitude, lat_translation_meters, \
        long_translation_meters = translate_lat_long_coords_params
    actual_lat, actual_lon = utils.translate_lat_long_coordinates(
        latitude, longitude, lat_translation_meters, long_translation_meters)
    # Assert actual_lat is a float
    assert isinstance(actual_lat, float)
    # Assert actual_lon is a float
    assert isinstance(actual_lon, float)


def test_get_inference_box_lat_lon_coordinates_type_errors(
        get_inference_box_lat_lon_coords_params):
    """
    Tests if TypeErrors is rasied for incorrect variable types.
    """
    box, img_center_lat, img_center_lon, image_x_pixels, \
        image_y_pixels, zoom_level = get_inference_box_lat_lon_coords_params
    # Tests for box list type
    with pytest.raises(
            TypeError, match="box variable must be of type list."):
        utils.get_inference_box_lat_lon_coordinates(
            124, img_center_lat, img_center_lon, image_x_pixels,
            image_y_pixels, zoom_level)
    # Tests for img_center_lat float type
    with pytest.raises(
            TypeError, match="img_center_lat variable must be of type float."):
        utils.get_inference_box_lat_lon_coordinates(
            box, "341", img_center_lon, image_x_pixels,
            image_y_pixels, zoom_level)
    # Tests for img_center_lon float type
    with pytest.raises(
            TypeError, match="img_center_lon variable must be of type float."):
        utils.get_inference_box_lat_lon_coordinates(
            box, img_center_lat, 82, image_x_pixels,
            image_y_pixels, zoom_level)
    # Tests for image_x_pixels int type
    with pytest.raises(
            TypeError, match="image_x_pixels variable must be of type int."):
        utils.get_inference_box_lat_lon_coordinates(
            box, img_center_lat, img_center_lon, 51.0,
            image_y_pixels, zoom_level)
    # Tests for image_y_pixels int type
    with pytest.raises(
            TypeError, match="image_y_pixels variable must be of type int."):
        utils.get_inference_box_lat_lon_coordinates(
            box, img_center_lat, img_center_lon, image_x_pixels,
            "152", zoom_level)
    # Tests for zoom_level int type
    with pytest.raises(
            TypeError, match="zoom_level variable must be of type int."):
        utils.get_inference_box_lat_lon_coordinates(
            box, img_center_lat, img_center_lon, image_x_pixels,
            image_y_pixels, [51])


def test_get_inference_box_lat_lon_coordinates_output(
        get_inference_box_lat_lon_coords_params):
    """
    Tests if the output returns a tuple type with float values.
    """
    box, img_center_lat, img_center_lon, image_x_pixels, \
        image_y_pixels, zoom_level = get_inference_box_lat_lon_coords_params
    actual_output = utils.get_inference_box_lat_lon_coordinates(
        box, img_center_lat, img_center_lon, image_x_pixels,
        image_y_pixels, zoom_level)
    # Assert actual output is a tuple
    assert isinstance(actual_output, tuple)
    # Asert that the length of actual_output is 2 values (latitude, longitude)
    assert len(actual_output) == 2
    # Assert that the values in actual_output are floats
    for val in actual_output:
        assert isinstance(val, float)


def test_binary_mask_to_polygon_type_errors():
    """
    Tests if TypeErrors is rasied for incorrect variable types.
    """
    mask = [[0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0]]
    # Tests for mask np.ndarray type
    with pytest.raises(
            TypeError, match="mask variable must be of type numpy.ndarray"):
        utils.binary_mask_to_polygon(mask)


def test_binary_mask_to_polygon_output():
    """
    Tests if the output returns a list of coordinates for a polygon.
    """
    # Generate a square binary mask of 2x2 block (1,1) to (2,2) in a 4x4 matrix
    mask = np.zeros((4, 4))
    mask[1:3, 1:3] = 1
    polygon_contours = utils.binary_mask_to_polygon(mask)
    # Assert that polygon_contours is a list
    assert isinstance(polygon_contours, list)
    # Assert that polygon_contours is of length 4 (a square) from mask
    assert len(polygon_contours) == 4
    for coordinates in polygon_contours:
        # Assert that the coordinates inside polygon_coutours is a tuple
        assert isinstance(coordinates, tuple)
        # Assert that the coordinates is a tuple is length 2 (x,y)
        assert len(coordinates) == 2


def test_convert_mask_to_lat_lon_polygon_type_errors(
        convert_mask_to_lat_lon_polygon_params):
    """
    Tests if TypeErrors is rasied for incorrect variable types.
    """
    mask, img_center_lat, img_center_lon, image_x_pixels, \
        image_y_pixels, zoom_level = convert_mask_to_lat_lon_polygon_params
    # Tests for mask variable np.ndarray type
    with pytest.raises(
            TypeError, match="mask variable must be of type numpy.ndarray"):
        utils.convert_mask_to_lat_lon_polygon(
            [(0, 0, 0), (1, 1, 0)], img_center_lat, img_center_lon,
            image_x_pixels, image_y_pixels, zoom_level)
    # Tests for img_center_lat float type
    with pytest.raises(
            TypeError, match="img_center_lat variable must be of type float."):
        utils.convert_mask_to_lat_lon_polygon(
            mask, [4231], img_center_lon, image_x_pixels,
            image_y_pixels, zoom_level)
    # Tests for img_center_lon float type
    with pytest.raises(
            TypeError, match="img_center_lon variable must be of type float."):
        utils.convert_mask_to_lat_lon_polygon(
            mask, img_center_lat, False, image_x_pixels,
            image_y_pixels, zoom_level)
    # Tests for image_x_pixels int type
    with pytest.raises(
            TypeError, match="image_x_pixels variable must be of type int."):
        utils.convert_mask_to_lat_lon_polygon(
            mask, img_center_lat, img_center_lon, True,
            image_y_pixels, zoom_level)
    # Tests for image_y_pixels int type
    with pytest.raises(
            TypeError, match="image_y_pixels variable must be of type int."):
        utils.convert_mask_to_lat_lon_polygon(
            mask, img_center_lat, img_center_lon, image_x_pixels,
            -2.0, zoom_level)
    # Tests for zoom_level int type
    with pytest.raises(
            TypeError, match="zoom_level variable must be of type int."):
        utils.convert_mask_to_lat_lon_polygon(
            mask, img_center_lat, img_center_lon, image_x_pixels,
            image_y_pixels, "18")


def test_convert_mask_to_lat_lon_polygon_output(
        convert_mask_to_lat_lon_polygon_params):
    """
    Tests if the output returns a list of coordinates for a polygon.
    """
    mask, img_center_lat, img_center_lon, image_x_pixels, \
        image_y_pixels, zoom_level = convert_mask_to_lat_lon_polygon_params
    polygon_coord_list = utils.convert_mask_to_lat_lon_polygon(
        mask, img_center_lat, img_center_lon, image_x_pixels,
        image_y_pixels, zoom_level)
    # Assert that polygon_coord_list is a list
    assert isinstance(polygon_coord_list, list)
    # Assert that polygon_contours is of length 4 (a square) from mask
    assert len(polygon_coord_list) == 4
    for coordinates in polygon_coord_list:
        # Assert that the coordinates inside polygon_coutours is a tuple
        assert isinstance(coordinates, tuple)
        # Assert that the coordinates is a tuple is length 2 (x,y)
        assert len(coordinates) == 2


def test_convert_polygon_to_geojson_type_errors():
    """
    Tests if TypeErrors is rasied for incorrect variable types.
    """
    polygon_coord_list = ((0, 0), (1, 0), (1, 1), (0, 1), (0, 0))
    # Tests for polygon_coord_list list type
    with pytest.raises(
            TypeError, match="polygon_coord_list variable must be of type" +
            " list"):
        utils.convert_polygon_to_geojson(polygon_coord_list)


def test_convert_polygon_to_geojson_output():
    """
    Tests if the output returns a GeoJSON string.
    """
    polygon_coord_list = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    geojson_poly_coord_list = [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0],
                                [0.0, 1.0], [0.0, 0.0]]]
    geojson_poly = utils.convert_polygon_to_geojson(polygon_coord_list)
    geojson_dict = json.loads(geojson_poly)
    # Assert that geojson_poly is a string
    assert isinstance(geojson_poly, str)
    # Assert that type is a FeatureCollection
    assert geojson_dict["type"] == "FeatureCollection"
    # Assert that the feature is a Polygon
    feature = geojson_dict["features"][0]
    assert feature["geometry"]["type"] == "Polygon"
    # Assert that the feature returns the polygon_coord_list in geojson format
    assert feature["geometry"]["coordinates"] == geojson_poly_coord_list
