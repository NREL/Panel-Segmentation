"""
Utility functions that are used when dealing with deep learning models.
"""

import requests
from PIL import Image
import numpy as np
import os
import time
import random
import rasterio
from rasterio.warp import transform
import math
import cv2
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import geopandas
from pyproj import Transformer

# meter/pixel zoom level data taken from the following source:
# https://support.plexearth.com/hc/en-us/articles/6325794324497-Understanding-Zoom-Level-in-Maps-and-Imagery

meter_pixel_zoom_dict = {9: 305.8,
                         10: 152.9,
                         11: 76.4,
                         12: 38.22,
                         13: 19.11,
                         14: 9.56,
                         15: 4.78,
                         16: 2.39,
                         17: 1.2,
                         18: 0.5972,
                         19: 0.256,
                         20: 0.1493,
                         21: 0.0746,
                         22: 0.0373,
                         23: 0.0187}


def generateSatelliteImage(latitude, longitude,
                           file_name_save, google_maps_api_key,
                           zoom_level=18):
    """
    Generates satellite image via Google Maps, using a set of lat-long
    coordinates.

    Parameters
    -----------
    latitude: float
        Latitude coordinate of the site.
    longitude: float
        Longitude coordinate of the site.
    file_name_save: string
        File path that we want to save
        the image to, where the image is saved as a PNG file.
    google_maps_api_key: string
        Google Maps API Key for automatically pulling satellite images. For
        more information, see here:
        https://developers.google.com/maps/documentation/maps-static/start
    zoom_level: int, default 18
        Zoom level of the image. Set to 18 as default, as that's what's used
        for the original panel-segmentation models.

    Returns
    -------
    Figure
        Figure of the satellite image
    """
    # Check input variable for types
    if not isinstance(latitude, float):
        raise TypeError("latitude variable must be of type float.")
    if not isinstance(longitude, float):
        raise TypeError("longitude variable must be of type float.")
    if not isinstance(zoom_level, int) or isinstance(zoom_level, bool):
        raise TypeError("zoom_level variable must be of type int.")
    if not isinstance(file_name_save, str):
        raise TypeError("file_name_save variable must be of type string.")
    if not isinstance(google_maps_api_key, str):
        raise TypeError("google_maps_api_key variable must be "
                        "of type string.")
    # Build up the lat_long string from the latitude-longitude coordinates
    lat_long = str(latitude) + ", " + str(longitude)
    # get method of requests module
    # return response object
    r = requests.get(
        "https://maps.googleapis.com/maps/api/staticmap?maptype"
        "=satellite&center=" + lat_long +
        "&zoom=" + str(zoom_level) + "&size=40000x40000&key=" +
        google_maps_api_key,
        verify=False)
    # Raise an exception if image is not successfully returned
    if r.status_code != 200:
        raise ValueError("Response status code " +
                         str(r.status_code) +
                         ": Image not pulled successfully from API.")
    # wb mode is stand for write binary mode
    with open(file_name_save, 'wb') as f:
        f.write(r.content)
        # close method of file object
        # save and close the file
        f.close()
    # Read in the image and return it via the console
    return Image.open(file_name_save)


def generateAddress(latitude, longitude, google_maps_api_key):
    """
    Gets the address of a latitude, longitude coordinates using Google
    Geocoding API. Please note rates for running geocoding checks here:
        https://developers.google.com/maps/billing-and-pricing/pricing

    Parameters
    -----------
    latitude: float
        Latitude coordinate of the site.
    longitude: float
        Longitude coordinate of the site.
    google_maps_api_key: string
        Google Maps API Key for geocoding a site. For further information,
        see here:
        https://developers.google.com/maps/documentation/geocoding/overview

    Returns
    -----------
    address: str
        Address of given latitude, longitude coordinates.
    """
    # Ensure that the inputs are of the correct type
    if not isinstance(latitude, float):
        raise TypeError("latitude variable must be of type float.")
    if not isinstance(longitude, float):
        raise TypeError("longitude variable must be of type float.")
    if not isinstance(google_maps_api_key, str):
        raise TypeError("google_maps_api_key variable must be "
                        "of type string.")
    # return response object
    r = requests.get(
        "https://maps.googleapis.com/maps/api/geocode/json?latlng=" +
        str(latitude) + "," + str(longitude) + "&key=" + google_maps_api_key,
        verify=False)
    # Raise an exception if address is not successfully returned
    if r.status_code != 200:
        raise ValueError("Response status code " +
                         str(r.status_code) +
                         ": Address not pulled successfully from API.")
    data = r.json()
    address = data["results"][0]["formatted_address"]
    return address


def generateSatelliteImageryGrid(northwest_latitude, northwest_longitude,
                                 southeast_latitude, southeast_longitude,
                                 google_maps_api_key,
                                 file_save_folder,
                                 zoom_level=18,
                                 lat_lon_distance=0.00145,
                                 number_allowed_images_taken=6000):
    """
    Take satellite images via the Google Maps API in a grid fashion for a large
    area, and save the associated images to a folder. The associated images
    can then be used to feed into different models to assess a larger area.

    Please note rates for running the Google Maps API here:
        https://developers.google.com/maps/billing-and-pricing/pricing

    Parameters
    ----------
    northwest_latitude : float
        Latitude coordinate on the northwest corner of the area we wish
        to scan.
    northwest_longitude : float
        Longitude coordinate on the northwest corner of the area we wish
        to scan.
    southeast_latitude : float
        Latitude coordinate on the southeast corner of the area
        we wish to scan.
    southeast_longitude : float
        Longitude coordinate on the southeast corner of the area
        we wish to scan.
    google_maps_api_key : str
        API key for Google Maps API
    file_save_folder : str
        Folder path for which to save all of the images
    zoom_level: int, default 18
        Zoom level of the image.
    lat_lon_distance : float, default 0.00145
        Distance to traverse between images, in terms of lat-long
        degree distance. For example, default distance of 0.00145 degrees
        equates to approx. 161 meters.
    number_allowed_images_taken : int, default 6000
        Number of allowed images for the Google Maps API to take before
        stopping. If we pull too many images in one go, google may flag
        this as webscraping so it's advised to not pull too many images
        at once.

    Returns
    -------
    grid_location_list: list of dictionaries
        A list of dictionaries with metadata information about each grid
        location in the image with the keys "file_name", "latitude",
        "longitude", "grid_x", and "grid_y".

    """
    # Ensure that the inputs are of the correct type
    if not isinstance(northwest_latitude, float):
        raise TypeError("northwest_latitude variable must be of type float.")
    if not isinstance(northwest_longitude, float):
        raise TypeError("northwest_longitude variable must be of type float.")
    if not isinstance(southeast_latitude, float):
        raise TypeError("southeast_latitude variable must be of type float.")
    if not isinstance(southeast_longitude, float):
        raise TypeError("southeast_longitude variable must be of type float.")
    if not isinstance(google_maps_api_key, str):
        raise TypeError("google_maps_api_key variable must be "
                        "of type string.")
    if not isinstance(file_save_folder, str):
        raise TypeError("file_save_folder variable must be "
                        "of type string.")
    if not isinstance(zoom_level, int) or isinstance(zoom_level, bool):
        raise TypeError("zoom_level variable must be of type int.")
    if not isinstance(lat_lon_distance, float):
        raise TypeError("lat_lon_distance variable must be "
                        "of type float.")
    if not isinstance(number_allowed_images_taken, int) or \
            isinstance(number_allowed_images_taken, bool):
        raise TypeError("number_allowed_images_taken variable must be "
                        "of type int.")
    # Build the grid out
    start_lat, start_lon = northwest_latitude, northwest_longitude
    lat_list, lon_list = [start_lat], [start_lon]
    # Latitude list
    while (start_lat >= southeast_latitude):
        start_lat = start_lat - lat_lon_distance
        lat_list.append(start_lat)
    # Longitude list
    while (start_lon <= southeast_longitude):
        start_lon = start_lon + lat_lon_distance
        lon_list.append(start_lon)
    counter = 0
    coord_list = list()
    grid_location_list = list()
    grid_y = 0
    for lon in lon_list:
        grid_x = 0
        for lat in lat_list:
            coord_list.append((lat, lon))
            # For every coordinate, take a satellite image and save it
            file_name = (str(round(lat, 7)) + "_" + str(
                         round(lon, 7)) + ".png")
            file_save = os.path.join(file_save_folder,
                                     file_name)
            grid_location_list.append({"file_name": file_name,
                                       "latitude": lat,
                                       "longitude": lon,
                                       "grid_x": grid_x,
                                       "grid_y": grid_y})
            grid_x += 1
            if not os.path.exists(file_save):
                generateSatelliteImage(lat, lon,
                                       file_save,
                                       google_maps_api_key,
                                       zoom_level)
            else:
                print("File already pulled!")
                continue
            counter += 1
            time.sleep(random.randint(1, 5))
            if counter >= number_allowed_images_taken:
                break
        grid_y += 1
    return grid_location_list


def visualizeSatelliteImageryGrid(grid_location_list, file_save_folder):
    """
    Using the grid_location_list output from the
    generateSatelliteImageryGrid() function, visualize all of the images
    taken in a grid.

    Parameters
    ----------
    grid_location_list: List of dictionaries
        List of dictionaries directly outputed from the
        generateSatelliteImageryGrid() function.
    file_save_folder: Str
        Folder path where all of the outputed satellite images from the
        generateSatelliteImageryGrid() function are stored.

    Returns
    -------
    grid: Plot Object
        Plot of gridded satellite images.
    """
    # Ensure that the inputs are of the correct type
    if not isinstance(grid_location_list, list):
        raise TypeError("grid_location_list variable must be of type list.")
    if not isinstance(grid_location_list[0], dict):
        raise TypeError("grid_location_list must be a list of dictionaries.")
    if not isinstance(file_save_folder, str):
        raise TypeError("file_save_folder variable must be of type str.")
    # Get the max grid coordinates so we can build the appropriate matplotlib
    # gridded graphic
    x_max = max([x['grid_x'] for x in grid_location_list]) + 1
    y_max = max([x['grid_y'] for x in grid_location_list]) + 1
    fig = plt.figure(figsize=(x_max*4, y_max*4))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(x_max, y_max),
                     axes_pad=0.1,
                     )
    # Read in all of the imagery into the grid
    for file in grid_location_list:
        file_name = file['file_name']
        print(file_name)
        x_loc = file['grid_x']
        y_loc = file['grid_y']
        img = Image.open(os.path.join(file_save_folder, file_name))
        grid[(x_loc * y_max) + y_loc].imshow(img)
    return grid


def splitTifToPngs(geotiff_file, meters_per_pixel,
                   meters_png_image, file_save_folder):
    """
    Take a master GEOTIFF file, grid it, and convert it to a series of PNG
    files.

    Parameters
    ----------
    geotiff_file: str
        File name of the TIF file we want to grid images from.
    meters_per_pixel: float
        TIF file resolution in meters/pixel. This needs to be previously known
        for the TIF file, so we can grid images based on the number of meters
        each image represents (ie zoom level)
    meters_png_image: float
        Number of meters we want an individual image output to represent, in
        both the x- and y-direction
    file_save_folder: str
        Folder path where we write all of the gridded images from the master
        TIF file.

    Returns
    -------
    None.
    """
    # Ensure that the inputs are of the correct type
    if not isinstance(geotiff_file, str):
        raise TypeError("geotiff_file variable must be of type str.")
    if not isinstance(meters_per_pixel, float):
        raise TypeError("meters_per_pixel variable must be of type float.")
    if not isinstance(meters_png_image, float):
        raise TypeError("meters_png_image variable must be of type float.")
    if not isinstance(file_save_folder, str):
        raise TypeError("file_save_folder variable must be of type str.")
    # Ignore aux.xml files when creating the png
    os.environ["GDAL_PAM_ENABLED"] = "NO"
    with rasterio.open(geotiff_file) as img:
        # All tif measurements were similar so divide the measured meters
        # from ArcGIS by its pixels
        pixel_meter_conversion = int(meters_png_image/meters_per_pixel)
        # Get image dimensions
        img_width, img_height = img.width, img.height
        # Loop over dimensions to crop images in a grid
        # Start from top left and stop at bottom right
        for top in range(0, img_height, pixel_meter_conversion):
            for left in range(0, img_width, pixel_meter_conversion):

                # Create a Window and calculate the transform
                window = rasterio.windows.Window(left, top,
                                                 pixel_meter_conversion,
                                                 pixel_meter_conversion)
                transform = img.window_transform(window)

                # Skip images where all pixels are black
                if np.all(img.read(window=window) == 0):
                    continue

                # Create a new cropped raster to write to
                profile = img.profile
                profile.update({'height': pixel_meter_conversion,
                                'width': pixel_meter_conversion,
                                'transform': transform,
                                'driver': "PNG"})
                # Get center of cropped image
                center = pixel_meter_conversion/2
                # Check if crs is in  "EPSG:4326" lat-lon coordinates.
                # If it's not, transform it to "EPSG:4326" coordinates.
                # Otherwise, get coordinates directly
                lon, lat = rasterio.transform.xy(transform, center, center)
                if img.crs and not img.crs.is_geographic:
                    transformer = Transformer.from_crs(img.crs, "EPSG:4326",
                                                       always_xy=True)
                    lon, lat = transformer.transform(lon, lat)

                # Put coordinats in lat_lon format for file name
                lat_lon = f"{lat:.7f}_{lon:.7f}"
                file_path = os.path.join(file_save_folder, f"{lat_lon}.png")
                with rasterio.open(file_path, 'w',
                                   **profile) as png:
                    # Read the data from the window and write it as a
                    # png output
                    png.write(img.read(window=window))
    return


def locateLatLonGeotiff(geotiff_file, latitude, longitude,
                        file_name_save, pixel_resolution=300):
    """
    Locate a lat-lon coordinate in a GEOTIFF image, and then box that area
    and capture a PNG image of it.

    Parameters
    -----------
    geotiff_file: str
        File name of the TIF file we want to scan for a particular latitude-
        longitude coordinate.
    latitude: float
        Target latitude coordinate we want to find in the TIFF file.
    longitude: float
        Target longitude coordinate we want to find in the TIFF file.
    file_name_save: str
        Name of the file of the image taken of the target lat-lon
        coordinate and its surrounding area.
    pixel_resolution : int, default 300
        Number of pixels in the x- and y-direction of the resulting image.

    Returns
    -------
    image or None
        The cropped image is returned as a PIL Image Object if designated
        lat-lon coordinates can be located in the GEOTIFF file with the image
        center set as designated lat-lon. Otherwise, returns None if the input
        coordinates are outside the image bounds or if all regions of the
        image are all black pixels.

    """
    # Ensure that the inputs are of the correct type
    if not isinstance(geotiff_file, str):
        raise TypeError("geotiff_file variable must be of type str.")
    if not isinstance(latitude, float):
        raise TypeError("latitude variable must be of type float.")
    if not isinstance(longitude, float):
        raise TypeError("longitude variable must be of type float.")
    if not isinstance(file_name_save, str):
        raise TypeError("file_name_save variable must be of type str.")
    if not isinstance(pixel_resolution, int) or \
            isinstance(pixel_resolution, bool):
        raise TypeError("pixel_resolution variable must be of type int.")
    # Open the file and get its boundaries
    with rasterio.open(geotiff_file) as dat:
        bounds = dat.bounds
        # Check if crs is in "EPSG:4326" lat-lon coordinates.
        # If not, transform it to "EPSG:4326" coordinates.
        if dat.crs and not dat.crs.is_geographic:
            lon, lat = transform("EPSG:4326", dat.crs,
                                 [longitude], [latitude])
            longitude, latitude = lon[0], lat[0]
        # Check that the lat-lon is within the image. If so, go to it
        if ((longitude >= bounds.left) & (longitude <= bounds.right) &
                (latitude <= bounds.top) & (latitude >= bounds.bottom)):
            # Convert lon, lat to pixel row, col coords
            rows, cols = dat.index(longitude, latitude)
            # Get top left corner of window
            top, left = (rows - pixel_resolution //
                         2), (cols - pixel_resolution//2)
            # Create a cropping window
            window = rasterio.windows.Window(
                col_off=left,
                row_off=top,
                width=pixel_resolution,
                height=pixel_resolution
            )
            # Read data in the window
            clip = dat.read(window=window)
            # Skip image if all pixels are black
            if np.all(clip == 0):
                print("All pixels are black in area, skipping...")
                return None
            # Save metadata
            meta = dat.meta
            meta['width'], meta['height'] = pixel_resolution, pixel_resolution
            meta['transform'] = rasterio.windows.transform(
                window, dat.transform)
            with rasterio.open(file_name_save, 'w', **meta) as dst:
                # Read the data from the window and write it as a png output
                dst.write(clip)
            # Open the file again and re-write via pillow so it doesn't have
            # any strange format issues
            image = Image.open(file_name_save)
            image.save(file_name_save)
            return image
        else:
            print("""Latitude-longitude coordinates are not within bounds of
                the image, no PNG captured...""")
            return None


def translateLatLongCoordinates(latitude, longitude,
                                lat_translation_meters,
                                long_translation_meters):
    """
    Method to move any lat-lon coordinate by provided meters in lat and long
    direction, and return the new latitude-longitude coordinates.

    Parameters
    -----------
    latitude: float
        Latitude coordinate of the site.
    longitude: float
        Longitude coordinate of the site.
    lat_translation_meters: float
        Movement of point in meters in latitude direction.If the value is
        postive, translation moves up, if negative it moves down
    long_translation_meters: float
        Movement of point in meters in longitude direction. If the value is
        postive, translation moves left, if negative it moves right

    Returns
    -------
    (lat_new, long_new) : tuple
        The (latitude, longitude) coordinates, post-translation.
    """
    # Ensure that the inputs are of the correct type
    if not isinstance(latitude, float):
        raise TypeError("latitude variable must be of type float.")
    if not isinstance(longitude, float):
        raise TypeError("longitude variable must be of type float.")
    if not isinstance(lat_translation_meters, float):
        raise TypeError("lat_translation_meters variable must be of" +
                        " type float.")
    if not isinstance(long_translation_meters, float):
        raise TypeError("long_translation_meters variable must be of" +
                        " type float.")
    earth_radius = 6378.137
    # Calculate top, which is lat_translation_meters
    m_lat = (1 / ((2 * math.pi / 360) * earth_radius)) / 1000
    lat_new = latitude + (lat_translation_meters * m_lat)
    # Calculate sideways, which is long_translation_meters
    m_long = (1 / ((2 * math.pi / 360) * earth_radius)) / 1000
    long_new = longitude + (long_translation_meters * m_long) / \
        math.cos(latitude * (math.pi / 180))
    return lat_new, long_new


def getInferenceBoxLatLonCoordinates(box, img_center_lat, img_center_lon,
                                     image_x_pixels, image_y_pixels,
                                     zoom_level):
    """
    Get the latitude-longitude coordinates of the centroid of a box output
    from model inference, based on the image center location & zoom level.

    Parameters
    -----------
    box : list
        A list of float pixel values containing the coordinates of a bounding
        box in the format [xmin, ymin, xmax, ymax].
    img_center_lat : float
        Latitude coordinate of the image center.
    img_center_lon : float
        Longitude coordinate of the image center.
    image_x_pixels : int
        The x width of the image in pixels.
    image_y_pixels : int
        The y height of the image in pixels.
    zoom_level : int
        Zoom level of the image.

    Returns
    -------
    (box_lat, box_lon) : tuple
        The (latitude, longitude) coordinates of the centroid of a box.
    """
    # Ensure that the inputs are of the correct type
    if not isinstance(box, list):
        raise TypeError("box variable must be of type list.")
    if not isinstance(img_center_lat, float):
        raise TypeError("img_center_lat variable must be of type float.")
    if not isinstance(img_center_lon, float):
        raise TypeError("img_center_lon variable must be of type float.")
    if not isinstance(image_x_pixels, int) or isinstance(image_x_pixels, bool):
        raise TypeError("image_x_pixels variable must be of type int.")
    if not isinstance(image_y_pixels, int) or isinstance(image_y_pixels, bool):
        raise TypeError("image_y_pixels variable must be of type int.")
    if not isinstance(zoom_level, int) or isinstance(zoom_level, bool):
        raise TypeError("zoom_level variable must be of type int.")
    image_center_pixels_x, image_center_pixels_y = (image_x_pixels/2,
                                                    image_y_pixels/2)
    xmin, ymin, xmax, ymax = (float(box[0]), float(box[1]),
                              float(box[2]), float(box[3]))
    cx = int((xmin + xmax) / 2)
    cy = int((ymin + ymax) / 2)
    # Get the difference in meters between the main centroid and
    # label centroid, based on the image zoom level
    meter_pixel_conversion = meter_pixel_zoom_dict[zoom_level]
    lon_translation_meters = ((cx - image_center_pixels_x) *
                              meter_pixel_conversion)
    lat_translation_meters = ((image_center_pixels_y - cy) *
                              meter_pixel_conversion)
    box_lat, box_lon = translateLatLongCoordinates(
        latitude=img_center_lat,
        longitude=img_center_lon,
        lat_translation_meters=lat_translation_meters,
        long_translation_meters=lon_translation_meters)
    return (box_lat, box_lon)


def binaryMaskToPolygon(mask):
    """
    Convert a binary mask output (from a deep learning model) to a list of
    polygon coordinates, which can later be converted to latitude-longitude
    coordinates.

    Parameters
    -----------
    mask : nparray
        A binary mask output from a deep learning model, which can be converted
        to a polygon.

    Returns
    -------
    contours_new : list
        A list of (x,y) image pixel-coordinate tuples for a polygon.
    """
    # Ensure that the input are of the correct type
    if not isinstance(mask, np.ndarray):
        raise TypeError("mask variable must be of type numpy.ndarray")
    # Ensure the mask is binary
    binary_mask = (mask > 0).astype(np.uint8)
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    contours_new = contours[0]
    for idx in range(1, len(contours)):
        contours_new = np.concatenate((contours_new, contours[idx]), axis=0)
    contours_new = [tuple(x[0]) for x in contours_new]
    return contours_new


def convertMaskToLatLonPolygon(mask, img_center_lat,
                               img_center_lon,
                               image_x_pixels, image_y_pixels,
                               zoom_level):
    """
    Take an inference mask output from a model, and convert it to a polygon
    with listed latitude-longitude coordinates.

    Parameters
    -----------
    mask : nparray
        A binary mask output from a deep learning model, which can be converted
        to a polygon.
    img_center_lat : float
        Latitude coordinate of the image center.
    img_center_lon : float
        Longitude coordinate of the image center.
    image_x_pixels : int
        The x width of the image in pixels.
    image_y_pixels : int
        The y height of the image in pixels.
    zoom_level : int
        The zoom level of the image.

    Returns
    -------
    polygon_coord_list : list
        A list of (latitude, longitude) coordinates for a polygon.
    """
    # Ensure that the input are of the correct type
    if not isinstance(mask, np.ndarray):
        raise TypeError("mask variable must be of type numpy.ndarray")
    if not isinstance(img_center_lat, float):
        raise TypeError("img_center_lat variable must be of type float.")
    if not isinstance(img_center_lon, float):
        raise TypeError("img_center_lon variable must be of type float.")
    if not isinstance(image_x_pixels, int) or isinstance(image_x_pixels, bool):
        raise TypeError("image_x_pixels variable must be of type int.")
    if not isinstance(image_y_pixels, int) or isinstance(image_y_pixels, bool):
        raise TypeError("image_y_pixels variable must be of type int.")
    if not isinstance(zoom_level, int) or isinstance(zoom_level, bool):
        raise TypeError("zoom_level variable must be of type int.")
    # First convert the mask to a polygon (in pixel coordinates)
    polygon_coords = binaryMaskToPolygon(mask)
    x_center, y_center = image_x_pixels/2, image_y_pixels/2
    meter_pixel_conversion = meter_pixel_zoom_dict[zoom_level]
    polygon_coord_list = list()
    # Convert the polygon coords to lat-long coordinates
    for coord in polygon_coords:
        # Get distance changes in x- and y-directions in meters
        dx = -(x_center - coord[0]) * meter_pixel_conversion
        dy = (y_center - coord[1]) * meter_pixel_conversion
        new_coords = translateLatLongCoordinates(img_center_lat,
                                                 img_center_lon,
                                                 dy, dx)[::-1]
        polygon_coord_list.append(new_coords)
    return polygon_coord_list


def convertPolygonToGeojson(polygon_coord_list):
    """
    Take a list of lat-lon coordinates for a polygon and convert
    to GeoJSON format.

    Parameters
    -----------
    polygon_coord_list : list
        A list of (latitude, longitude) coordinates for a polygon.

    Returns
    -------
    geojson_poly: str
        A GeoJSON string representation of the polygon.
    """
    # Ensure that the input are of the correct type
    if not isinstance(polygon_coord_list, list):
        raise TypeError("polygon_coord_list variable must be of type list")
    shapely_poly = Polygon(polygon_coord_list)
    geojson_poly = geopandas.GeoSeries(shapely_poly).to_json()
    return geojson_poly