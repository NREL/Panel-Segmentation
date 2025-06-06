# -*- coding: utf-8 -*-
"""
Hail detection inference pipeline (finalized). This is for the PVSC paper statistics
"""

# Only flag for single warning if there are repeats
import warnings
warnings.filterwarnings(action='once')

from panel_segmentation import utils
import glob
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
import os
from shapely.geometry import Polygon
import geopandas

example_images = glob.glob("C:/Users/kperry/Documents/extreme_weather_data/hail_satellite_imagery/*.jpg")

# Hail damage model
hail_model_path = "./panel_segmentation/models/hail_model.pt"
hail_model = YOLO(hail_model_path)

# Module only model
module_file_path = "C:/Users/kperry/Documents/source/repos/sol-searcher/mmdetection_pipeline/runs/segment/train5/weights/best.pt"
module_model = YOLO(module_file_path)

module_damage_list = list()

for file_name in example_images:
    # Run for the module masks specifically
    mod_result = module_model([file_name], overlap_mask=False)[0]
    mod_masks = mod_result.masks
    base_file_name = os.path.basename(mod_result.path)
    img_center_lat, img_center_lon = (float(base_file_name.split("_")[0]),
                                      float(base_file_name.split("_")[-1].replace(".jpg", "")))
    image_x_pixels, image_y_pixels = mod_result.orig_shape
    module_polys = list()
    for index in range(len(mod_masks)):
        try:
            segmentation_mask = mod_masks[index].data.cpu()[0].numpy()
            # Convert the mask to a polygon
            polygon_lat_lon_coords = utils.convertMaskToLatLonPolygon(segmentation_mask, 
                                                                      img_center_lat, 
                                                                      img_center_lon,
                                                                      image_x_pixels, 
                                                                      image_y_pixels,
                                                                      zoom_level=21)
            # Convert to a geoJSON
            shapely_poly = Polygon(polygon_lat_lon_coords)
            geojson_poly = geopandas.GeoSeries(shapely_poly).to_json()
            module_polys.append(shapely_poly)
        except: 
            pass
    # Now run the associated 
    hail_result = hail_model([file_name], overlap_mask=False)[0]
    hail_masks, hail_boxes = hail_result.masks, hail_result.boxes  
    categories = [int(box.cls[0]) for box in hail_boxes]
    # If hail damage is present, calculate the size of the damage area
    hail_damage_polys = list()
    if (hail_masks is not None) & (1 in categories):
        for index in range(len(hail_masks)):
            hail_segmentation_mask = hail_masks[index].data.cpu()[0].numpy()
            category = hail_boxes[index].cls[0]
            if category == 1:
                try:
                    # Convert the mask to a polygon
                    polygon_lat_lon_coords = utils.convertMaskToLatLonPolygon(hail_segmentation_mask, 
                                                                              img_center_lat, 
                                                                              img_center_lon,
                                                                              image_x_pixels, 
                                                                              image_y_pixels,
                                                                              zoom_level=21)
                    # Convert to a geoJSON
                    hail_shapely_poly = Polygon(polygon_lat_lon_coords)
                    hail_geojson_poly = geopandas.GeoSeries(hail_shapely_poly).to_json()
                    hail_damage_polys.append(hail_shapely_poly)
                except:
                    pass
    # now let's check for overlapping module + hail masks
    for module_poly in module_polys:
        total_dmg_amt = 0
        if module_poly.is_valid:
            # Loop through all of the hail damage polygons and get the
            # hail damage masks that overlap the module mask
            for hail_poly in hail_damage_polys:
                if hail_poly.is_valid:
                    if hail_poly.intersects(module_poly):
                        # Check if the hail polygon is 90% within the module polygon.
                        if ((hail_poly.intersection(module_poly).area/hail_poly.area) * 100) > 90:
                            dmg_pct = ((hail_poly.area/module_poly.area) * 100)
                            total_dmg_amt = total_dmg_amt + dmg_pct
        else:
            total_dmg_amt = 0
        # Record the total damage amount
        module_damage_list.append({"latitude": img_center_lat,
                                   "longitude": img_center_lon,
                                   "file_name": base_file_name,
                                   "module_polygon": module_poly,
                                   "damage_pct": total_dmg_amt})
        