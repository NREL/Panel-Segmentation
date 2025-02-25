# -*- coding: utf-8 -*-
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
import math
import cv2


def generateSatelliteImage(latitude, longitude, zoom_level,
                           file_name_save, google_maps_api_key):
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

    Returns
    -------
        Figure of the satellite image
    """
    # Check input variable for types
    if type(latitude) != float:
        raise TypeError("latitude variable must be of type float.")
    if type(longitude) != float:
        raise TypeError("longitude variable must be of type float.")
    if type(zoom_level) != int:
        raise TypeError("zoom_level variable must be of type int.")
    if type(file_name_save) != str:
        raise TypeError("file_name_save variable must be of type string.")
    if type(google_maps_api_key) != str:
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


def generate_address(latitude, longitude, google_maps_api_key):
    """
    Gets the address of a latitude, longitude coordinates using Google
    Geocoding API.
    
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


def generate_satellite_imagery_grid(northwest_lat, northwest_lon,
                                    southeast_lat, southeast_lon,
                                    google_maps_api_key,
                                    zoom_level,
                                    file_save_folder,
                                    lat_lon_distance=.00145,
                                    number_allowed_images_taken=6000):
    """
    Take satellite images via the Google Maps API in a grid fashion for a large
    area, and save the associated images to a folder. The associated images
    can then be used to feed into different models to assess a larger area.

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
    lat_lon_distance : float
        Distance to traverse between images, in terms of lat-long
        degree distance. For example, default distance of 0.00145 degrees
        equates to approx. 161 meters.
    google_maps_api_key : str
        API key for Google Maps API
    file_save_folder : str
        Folder path for which to save all of the images
    number_allowed_images_taken : int, default 6000
        Number of allowed images for the Google Maps API to take before
        stopping. If we pull too many images in one go, google may flag
        this as webscraping so it's advised to not pull too many images
        at once.

    Returns
    -------
    None.

    """
    # Build the grid out
    start_lat, start_lon = northwest_lat, northwest_lon
    lat_list, lon_list = [start_lat], [start_lon]
    # Latitude list
    while (start_lat >= southeast_lat):
        start_lat = start_lat - lat_lon_distance
        lat_list.append(start_lat)
    # Longitude list
    while (start_lon <= southeast_lon):
        start_lon = start_lon + lat_lon_distance
        lon_list.append(start_lon)
    counter = 0
    coord_list = list()
    for lon in lon_list:
        for lat in lat_list:
            coord_list.append((lat, lon))
            # For every coordinate, take a satellite image and save it
            file_save = os.path.join(file_save_folder,
                                     str(round(lat, 7)) + "_" + str(
                                         round(lon, 7)) + ".png")
            if not os.path.exists(file_save):
                generateSatelliteImage(lat, lon, 
                                       zoom_level,
                                       file_save,
                                       google_maps_api_key)
            else:
                print("File already pulled!")
                continue
            counter += 1
            time.sleep(random.randint(1, 5))
            if counter >= number_allowed_images_taken:
                break
    return


def split_tif_to_pngs(tif_file, meters_per_pixel, 
                      number_meters_png_image, file_save_folder):
    """
    Take a master GEOTIFF file, grid it, and convert it to a series of PNG
    files.

    Parameters
    ----------
    tif_file: str
    meters_per_pixel: float
    number_meters_png_image: int
    file_save_folder: str

    Returns
    -------
    None.
    """
    # Ignore aux.xml files when creating the png
    os.environ["GDAL_PAM_ENABLED"] = "NO"
    with rasterio.open(tif_file) as img:
        # All tif measurements were similar so divide the measured meters
        # from ArcGIS by its pixels
        pixel_meter_conversion = int(number_meters_png_image/meters_per_pixel)
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
                # Get coordinates
                lon, lat = rasterio.transform.xy(transform, center, center)
                # Put coordinats in lat_lon format for file name
                lat_lon = f"{lat:.7f}_{lon:.7f}"
                with rasterio.open(f'{file_save_folder}{lat_lon}.png', 'w',
                                   **profile) as png:
                    # Read the data from the window and write it as a
                    # png output
                    png.write(img.read(window=window))
    return


def locate_lat_lon_geotiff(geotiff_file, latitude, longitude,
                           file_name_save, pixel_resolution=300):
    """
    Locate a lat-lon coordinate in a GEOTIFF image, and then box that area
    and capture a PNG image of it.

    Parameters
    -----------
    geotiff_file: str
    latitude: float
    longitude: float
    file_name_save: str
    pixel_resolution : int, default 300
    
    Returns
    -------
    image or None
    """
    # Open the file and get its boundaries
    dat = rasterio.open(geotiff_file)
    boundaries = dat.bounds
    # Check that the lat-lon is within the image. If so, go to it
    if ((longitude >= boundaries.left) & (longitude <= boundaries.right) &
        (latitude <= boundaries.top) & (latitude >= boundaries.bottom)):
        # Convert lon, lat to pixel row, col coords
        rows, cols = dat.index(longitude, latitude)
        # Get top left corner of window
        top, left = (rows - pixel_resolution//2), (cols - pixel_resolution//2)
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
        meta['transform'] = rasterio.windows.transform(window, dat.transform)
        with rasterio.open(file_name_save, 'w', **meta) as dst:
            # Read the data from the window and write it as a png output
            dst.write(clip)
        # Open the file again and re-write via pillow so it doesn't have any
        # strange format issues
        image = Image.open(file_name_save)
        image.save(file_name_save)
        return image
    else:
        print("""Latitude-longitude coordinates are not within bounds of
              the image, no PNG captured...""")
        return None


def translate_lat_long_coordinates(latitude, longitude, 
                                   lat_translation_meters,
                                   long_translation_meters):
    '''
    Method to move any lat-lon coordinate by provided meters in lat and long
    direction, and return the new latitude-longitude coordinates.
    Taken from the following source:
        https://stackoverflow.com/questions/7477003/calculating-new-longitude-latitude-from-old-n-meters
    
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
    lat_new : float
        New latitude coordinate, post-translation
    long_new : float
        New longitude coordinate, post-translation
    '''
    earth_radius = 6378.137
    # Calculate top, which is lat_translation_meters
    m_lat = (1 / ((2 * math.pi / 360) * earth_radius)) / 1000
    lat_new = latitude + (lat_translation_meters * m_lat)
    # Calculate sideways, which is long_translation_meters
    m_long = (1 / ((2 * math.pi / 360) * earth_radius)) / 1000
    long_new = longitude + (long_translation_meters * m_long) / \
        math.cos(latitude * (math.pi / 180))
    return lat_new, long_new


def binary_mask_to_polygon_opencv(mask):
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
        A list of (x,y) coordinates for a polygon
    """
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
