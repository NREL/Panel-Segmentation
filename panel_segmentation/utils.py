# -*- coding: utf-8 -*-
"""
Utility functions that are used by all of the processes!!!
"""

import folium
import zipfile
import io
from fastkml import kml
import requests
from PIL import Image
from shapely.geometry import Point, Polygon
import ast
from shapely import from_wkb, from_wkt
import pandas as pd
import numpy as np
import folium
import matplotlib.pyplot as plt
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
        Google Maps API Key for
        automatically pulling satellite images.

    Returns
    -----------
        Figure
        Figure of the satellite image
    """
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
        Google Maps API Key for
        automatically pulling satellite images.

    Returns
    -----------
    address: string
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


def generate_grid_satellite_imagery_google(northwest_lat, northwest_lon,
                                           southeast_lat, southeast_lon,
                                           google_maps_api_key, file_save_folder,
                                           lat_lon_distance=.00145,
                                           number_allowed_images_taken=6000):
    """
    Take satellite images via the Google Maps API in a grid fashion for a large
    area. This is for generating training/inference data for computer vision
    models.

    Parameters
    ----------
    northwest_lat : Latitude coordinate on the northwest corner of the area
        we wish to scan.
    northwest_lon : Longitude coordinate on the northwest corner of the area
        we wish to scan.
    southeast_lat : Latitude coordinate on the southeast corner of the area
        we wish to scan.
    southeast_lon : Longitude coordinate on the southeast corner of the area
        we wish to scan.
    lat_lon_distance : Distance to traverse between images, in terms of lat-long
        degree distance. For example, default distance of 0.00145 degrees equates
        to approx. 161 meters.
    google_maps_api_key : API key for Google Maps API
    file_save_folder : Folder path for which to save all of the images
        number_allowed_images_taken : Number of allowed images for the Google Maps
        API to take before stopping. This is set at 6000. If we pull too many images
        in one go, google flags this as webscraping so we need to not go over that
        6k limit on a daily basis if we don't want to get in trouble.

    Returns
    -------
    None.

    """
    # Build the grid out
    start_lat, start_lon = northwest_lat, northwest_lon
    lat_list = [start_lat]
    lon_list = [start_lon]

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
            # For every coordinate, take a satellite image and save it the austin
            # metro satellite imagery dataset
            file_save = os.path.join(file_save_folder,
                                     str(round(lat, 7)) + "_" + str(round(lon, 7)) + ".png")
            if not os.path.exists(file_save):
                generateSatelliteImage(lat, lon,
                                       file_save, google_maps_api_key, 19)
            else:
                print("file already pulled!")
                continue
            counter += 1
            time.sleep(random.randint(1, 5))
            if counter >= 6000:
                break
    return


def split_tif_to_pngs(tif_file, meters_per_pixel, number_meters_png_image, file_save_folder):
    """
    Take a master GEOTIFF file, grid it, and convert it to a series of PNG
    files.
    """
    # Ignore aux.xml files when creating the png
    os.environ["GDAL_PAM_ENABLED"] = "NO"
    with rasterio.open(tif_file) as img:
        # All tif measurements were similar so divide the measured meters
        # from ArcGIS by its pixels
        pixel_x = int(number_meters_png_image/meters_per_pixel)
        pixel_y = int(number_meters_png_image/meters_per_pixel)
        # Get image dimensions
        img_width = img.width
        img_height = img.height

        # Loop over dimensions to crop images in a grid
        # Start from top left and stop at bottom right
        for top in range(0, img_height, pixel_y):
            for left in range(0, img_width, pixel_x):

                # Create a Window and calculate the transform
                window = rasterio.windows.Window(left, top, pixel_x, pixel_y)
                transform = img.window_transform(window)

                # Skip images where all pixels are black
                if np.all(img.read(window=window) == 0):
                    continue

                # Create a new cropped raster to write to
                profile = img.profile
                profile.update({
                    'height': pixel_y,
                    'width': pixel_x,
                    'transform': transform,
                    'driver': "PNG"})

                # Get center of cropped image
                center_x = pixel_x/2
                center_y = pixel_y/2

                # Get coordinates
                lon, lat = rasterio.transform.xy(transform, center_y, center_x)
                # Put coordinats in lat_lon format for file name
                lat_lon = f"{lat:.7f}_{lon:.7f}"
                with rasterio.open(f'{file_save_folder}{lat_lon}.png', 'w', **profile) as png:
                    # Read the data from the window and write it as a png output
                    png.write(img.read(window=window))
    return


def auto_grid_area_heatmap(lat, lon, file_name_save, dist=1000, coors=12):
    """
    Autogrid an area based on a center point (lat, long). Good for making data
    for calling an API for heatmaps, etc. Create an associated folium graphic
    for the gridded area.

    lat: latitude of center point
    lon: longitude of center point
    file_name_save: name of path for which to save the final folium graphic.
    dist: in meters
    coors: number of coords in each direction
    """
    # Creating the offset grid
    mini, maxi = -dist*coors, dist*coors
    n_coord = coors*2+1
    axis = np.linspace(mini, maxi, n_coord)
    X, Y = np.meshgrid(axis, axis)
    # avation formulate for offsetting the latlong by offset matrices
    R = 6378137  # earth's radius
    dLat = X/R
    dLon = Y/(R*np.cos(np.pi*lat/180))
    latO = lat + dLat * 180/np.pi
    lonO = lon + dLon * 180/np.pi

    # stack x and y latlongs and get (lat,long) format
    output = np.stack([latO, lonO]).transpose(1, 2, 0)
    points = output.reshape(-1, 2)
    x = points[:, 0]
    y = points[:, 1]

    plt.scatter(x, y)  # <- plot all points
    plt.scatter(lat, lon, color='r')
    plt.show()

    df = pd.DataFrame()
    df['latitude'] = x
    df['longitude'] = y

    # Create a map to visualize the grid (in Folium)
    mapa = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()],
                      tiles="Cartodb Positron",
                      zoom_start=11)

    for lat, lon in zip(df['latitude'], df['longitude']):
        folium.CircleMarker([lat, lon],
                            color='b',
                            weight=.05,
                            fill=True,
                            fill_opacity=0.7,
                            fill_color='black',
                            radius=5,
                            ).add_to(mapa)
    wms = folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Esri Satellite',
        overlay=False,
        control=True
    )
    feature_group1 = folium.FeatureGroup(name='Topo')
    feature_group1.add_child(wms)
    mapa.add_child(feature_group1)
    mapa.add_child(folium.map.LayerControl())
    mapa.save(file_name_save)
    return


def locate_take_image_geotiff(geotiff_file, latitude, longitude,
                              file_name_save, pixel_resolution=300):
    """
    Locate a lat-lon coord in a GEOTIFF image, and then box that area
    and capture a PNG image of it.

    Returns
    -------
    None.

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
        top = rows - pixel_resolution//2
        left = cols - pixel_resolution//2
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
            return
        # Save metadata
        meta = dat.meta
        meta['width'], meta['height'] = pixel_resolution, pixel_resolution
        meta['transform'] = rasterio.windows.transform(window, dat.transform)
        with rasterio.open(file_name_save, 'w', **meta) as dst:
            # Read the data from the window and write it as a png output
            dst.write(clip)

        # Open the file again and re-write via pillow (to make its in the right
        # format for roboflow)
        image = Image.open(file_name_save)
        image.save(file_name_save)
        return
    else:
        print("Latitude-longitude coordinates are not within bounds of the image, no PNG captured...")
        return


def translate_latlong(lat, long, lat_translation_meters, long_translation_meters):
    '''
    Taken from the following source:
        https://stackoverflow.com/questions/7477003/calculating-new-longitude-latitude-from-old-n-meters
    method to move any lat,long point by provided meters in lat and long direction.
    params :
        lat,long: latitude and longitude in degrees as decimal values
        lat_translation_meters: movement of point in meters in lattitude direction.
                                positive value: up move, negative value: down move
        long_translation_meters: movement of point in meters in longitude direction.
                                positive value: left move, negative value: right move
    '''
    earth_radius = 6378.137
    # Calculate top, which is lat_translation_meters above
    m_lat = (1 / ((2 * math.pi / 360) * earth_radius)) / 1000
    lat_new = lat + (lat_translation_meters * m_lat)
    # Calculate right, which is long_translation_meters right
    m_long = (1 / ((2 * math.pi / 360) * earth_radius)) / \
        1000  # 1 meter in degree
    long_new = long + (long_translation_meters * m_long) / \
        math.cos(lat * (math.pi / 180))
    return lat_new, long_new


def binary_mask_to_polygon_opencv(mask):
    """
    Convert a binary mask output (from a deep learning model) to a list of
    polygon coordinates.
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
