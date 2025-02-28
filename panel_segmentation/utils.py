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
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt

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
                                       "lon": lon,
                                       "grid_x": grid_x,
                                       "grid_y": grid_y})
            grid_x += 1
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
        grid_y += 1
    return grid_location_list

def visualize_satellite_imagery_grid(grid_location_list, file_save_folder):
    """
    Using the grid_location_list output from the 
    generate_satellite_imagery_grid() function, visualize all of the images
    taken in a grid.
    
    Parameters
    ----------
    grid_location_list: List of dictionaries
        List of dictionaries directly outputed from the
        generate_satellite_imagery_grid() function.
    file_save_folder: Str
        Folder path where all of the outputed satellite images from the
        generate_satellite_imagery_grid() function are stored.
        

    Returns
    -------
    Plot
        Plot of gridded satellite images.
    """
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
        x_loc = file['grid_x']
        y_loc = file['grid_y']
        img = Image.open(os.path.join(file_save_folder, file_name))
        grid[(x_loc * y_max)+ y_loc].imshow(img)
    return grid

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


def get_inference_box_lat_lon_coordinates(box, img_center_lat, img_center_lon,
                                          image_x_pixels, image_y_pixels,
                                          zoom_level):
    """
    Get the latitude-longitude coordinates of the centroid of a box output
    from model inference, based on the image center location & zoom level.
    
    Parameters
    -----------
    box
    img_center_lat
    img_center_lon
    image_x_pixels
    image_y_pixels

    Returns
    -------
    """
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
    box_lat, box_lon = translate_lat_long_coordinates(
        latitude = img_center_lat,
        longitude = img_center_lon, 
        lat_translation_meters = lat_translation_meters,
        long_translation_meters = lon_translation_meters)
    return (box_lat, box_lon)

def binary_mask_to_polygon(mask):
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


def convert_mask_to_lat_lon_polygon(mask, img_center_lat, img_center_lon,
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
    img_center_lat, 
    img_center_lon,
    image_x_pixels, 
    image_y_pixels,
    zoom_level
        

    Returns
    -------
    None.
    """
    # First convert the mask to a polygon (in pixel coordinates)
    polygon_coords = binary_mask_to_polygon(mask)
    x_center, y_center = image_x_pixels/2, image_y_pixels/2
    meter_pixel_conversion = meter_pixel_zoom_dict[zoom_level]
    polygon_coord_list = list()
    # Convert the polygon coords to lat-long coordinates
    for coord in polygon_coords:
        # Get distance changes in x- and y-directions in meters
        dx = -(x_center - coord[0]) * meter_pixel_conversion
        dy = (y_center - coord[1]) * meter_pixel_conversion
        new_coords = translate_lat_long_coordinates(img_center_lat,
                                                    img_center_lon, 
                                                    dy, dx)[::-1] 
        polygon_coord_list.append(new_coords)
    return polygon_coord_list