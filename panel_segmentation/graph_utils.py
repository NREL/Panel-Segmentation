# -*- coding: utf-8 -*-
"""
Graphing utility functions that are used by all of the processes!!!
"""

import pandas as pd
import os
import numpy as np
import folium
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import geojsoncontour
import json
from scipy.ndimage import gaussian_filter
import simplekml
import folium_arrow_icon
import math
from PIL import Image 

def generate_geojson_heatmap(df, min_bin, max_bin, bin_size):
    """
    Generates a contour geojson heatmap of the max wind gust speed.
    Parameters:
        df (Pandas dataframe): pandas dataframe with latitude, longitude, and
            wind_gust (m/s) columns
        min_bin (float): lowest bin value for max wind gust speeds
        max_bin (float): highest bin value for max wind gust speeds
        bin_size (int): size of binning for max wind gust speeds
    Returns:
        windgust_geojson (dict): Geojson dictionary containing information
            about the max wind gust speed contours
        contour (object): matplotlib plot filled contours.
    """
    lat = df["latitude"]
    lon = df["longitude"]
    max_windgust = df["max_windgust"]

    # Create grid
    x = np.linspace(lon.min(), lon.max(), 100)
    y = np.linspace(lat.min(), lat.max(), 100)
    x_mesh, y_mesh = np.meshgrid(x, y)
    # Put data on grid
    z_mesh = griddata((lon, lat), max_windgust, (x_mesh, y_mesh), method="linear")
    # Make contour lines with defined bin
    bins = np.arange(min_bin, max_bin, bin_size)
    
    # Use Gaussian filter to smoothen the contour
    z_mesh = gaussian_filter(z_mesh, 0.8)

    contour = plt.contourf(x_mesh, y_mesh, z_mesh, levels=bins, cmap="RdYlBu_r") 
    # contour = plt.contourf(x_mesh, y_mesh, z_mesh) # Does autobinning
    
    # Convert contour lines to geojson
    windgust_geojson = geojsoncontour.contourf_to_geojson(contourf=contour)
    windgust_geojson = json.loads(windgust_geojson)

    return windgust_geojson, contour


def autogenerate_kml(df, file_save_name, color_column=None):
    """
    Read in a dataframe and automatically generate a KML file from it (for
    viewing on Google Earth).


    Parameters
    ----------
    df : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    kml = simplekml.Kml()
    for idx, row in df.iterrows():
        if color_column:
            color = row[color_column]
        lat = row['latitude']
        lon = row['longitude']
        pnt = kml.newpoint()
        pnt.coords = [(lon, lat)]
        if color:
            if color == 'red':
                pnt.style.iconstyle.color = 'ff0000ff'
            elif color == 'orange':
                pnt.style.iconstyle.color = 'ff008cff'
            elif color == 'yellow':
                pnt.style.iconstyle.color = 'ff00ffff'
            else:
                pnt.style.iconstyle.color = 'ffff0000'
        else:
            # If no color, just label it all blue
            pnt.style.iconstyle.color = 'ffff0000'
        pnt.style.labelstyle.scale = 2
        pnt.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/target.png'  
    # save KML to a file
    kml.save(file_save_name)
    return


def generate_wind_direction_quiver_plot_folium(df, wind_direction_column,
                                               file_save_name):
    """
    Generate a quiver plot for wind direction plotting on a grid
    in Folium.
    """
    df['radian'] = [((x+ 180) * math.pi /180) for x in
                    list(df[wind_direction_column])]
    mapb = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()],
                      tiles="OpenStreetMap",
                      zoom_start=10)
    for idx, row in df.iterrows():
        lat, lon = row['latitude'], row['longitude']
        radian =row['radian']
        folium.Marker(
            (lat,lon),
            icon=folium_arrow_icon.ArrowIcon(
                12, radian,
                head=folium_arrow_icon.ArrowIconHead(width=8, length=6),
                body=folium_arrow_icon.ArrowIconBody(width=3),
                color="black",
                border_width=1,
                border_color="black",
                anchor="mid"
                )
                ).add_to(mapb)

    mapb.add_child(folium.map.LayerControl())
    mapb.save(file_save_name)
    return


def reduce_image_quality_size(png_file_path, jpg_file_path):
    """
    Reduces the image quality until the image file size is less than 1MB.
    Embeded images in Folium displays if image size is less than 1MB.
    This is needed for large geotiff files.
    """
    # Convert png to jpg
    im = Image.open(png_file_path) 
    rgb_im = im.convert('RGB')
    quality=95
    # Loop until jpg file is less than 1MB
    while True:
        rgb_im.save(f'{jpg_file_path}', quality=quality, optimize=True)
        # Get size of image in kb
        im_size = os.path.getsize(jpg_file_path) / 1024
        if im_size < 1024:
            #print(f"Image quality: {quality}. JPG file path: {jpg_file_path}")
            break
        quality -=5
        if quality <=10:
            print(f"Cannot reduce {png_file_path} to less than 1MB in size.")
