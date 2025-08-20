import laspy
import numpy as np
import open3d as o3d
from pyproj import Transformer
import shapely
from shapely.ops import transform


class PCD:
    '''
    A class that reads, filters, and crops point cloud data from .laz file.
    '''

    def __init__(self, laz_file_path, polygon=None,
                 latitude=None, longitude=None):
        # File path of .laz data
        self.laz_file_path = laz_file_path
        # Shapely polygon in EPSG:4326 coordinates in .laz data
        self.polygon = polygon
        # EPSG:4326 latitude, longitude center points in .laz data
        self.latitude = latitude
        self.longitude = longitude
        # Only choose polygon or latitude, longitude center coordinates
        # Cannot be both shapely polygon and lat, lon
        if self.polygon and (self.latitude and self.longitude):
            raise ValueError(
                "Please input a shapely polygon or latitude, longitude" +
                " center coordinates to generate a bbox to get pcd data. " +
                "Do not enter both.")
        if not self.polygon and not (self.latitude and self.longitude):
            raise ValueError(
                "No shapely polygon or latitude, longitude center " +
                "coordinates were input. Please input a shapely polygon or " +
                "latitude, longitude center coordinates to generate a bbox " +
                "to get pcd data.")

    def readCropLaz(self, lat_lon_bbox_size=20):
        """
        Reads and crops LiDAR data from a .laz file based on a shapely polygon
        or latitude, longitude coordinate bounding box.

        Parameters:
        -----------
        lat_lon_bbox_size: int
            Size of bounding box (in meters) to crop the LiDAR data.
            Defaulted to 20. This is only needed when cropping with latitude,
            longitude coordinate point and not when using a shapely polygon.

        Returns:
        --------
        output_laz: laspy.LasData object
            Cropped and filtered LiDAR data that only contains the
            X,Y,Z points within the shapely polygon or latitude, longitude
            bouding box.
        """
        # Ensure that the inputs are of the correct type
        if not isinstance(lat_lon_bbox_size, int):
            raise TypeError(
                "lat_lon_bbox_size variable must be of type int.")
        # Process LiDAR data in chunks since .laz files are large
        CHUNK_SIZE = 1_000_000
        with laspy.open(self.laz_file_path) as f:
            filtered_points = []
            final_point_record = laspy.PackedPointRecord.empty(
                f.header.point_format)
            # Get crs of LiDAR data
            self.source_crs = f.header.parse_crs()
            # Check if crs has multiple components
            if self.source_crs.is_compound:
                # Get horizontal crs
                # Only need horizontal projection since Shapely polygon is 2D
                horizontal_crs = self.source_crs.sub_crs_list[0]
                dst_crs = f"EPSG:{horizontal_crs.to_epsg()}"
            else:
                dst_crs = f"EPSG:{self.source_crs.to_epsg()}"
            # Make coordinate transformer from EPSG:4326 to LiDAR crs
            self.transformer = Transformer.from_crs(
                "EPSG:4326", dst_crs, always_xy=True)
            # For shapely polygons
            if self.polygon:
                # Project polygons in latitude, longitude (in "EPSG:4326")
                # onto lidar coordinate system
                projected_poly = transform(
                    self.transformer.transform, self.polygon)
            # For latitude, longitude coordinate point
            elif self.latitude and self.longitude:
                # Project latitude, longitude coordinate points onto lidar
                # coordinate system
                x_mid, y_mid = self.transformer.transform(
                    self.longitude, self.latitude)
                # Make bbox with x_mid, y_mid as it's centers
                bbox_size = lat_lon_bbox_size / 2
                left, right = x_mid - bbox_size, x_mid + bbox_size
                bottom, top = y_mid - bbox_size, y_mid + bbox_size
            # Get [x, y] scale factors to match transformed lat long to
            # array values
            self.scales = f.header.scales
            self.offsets = f.header.offsets
            # Read values by chunks to avoid crashing
            for points in f.chunk_iterator(CHUNK_SIZE):
                # Resize coordinates with lidar's scale and offset
                x_coord = points["X"] * self.scales[0] + self.offsets[0]
                y_coord = points["Y"] * self.scales[1] + self.offsets[1]
                # Create bbox
                if self.polygon:
                    bbox = shapely.contains_xy(
                        projected_poly, x_coord, y_coord)
                elif self.latitude and self.longitude:
                    bbox = ((x_coord >= left) & (x_coord <= right) &
                            (y_coord >= bottom) & (y_coord <= top))
                # Filter out unnecessary classification and get only
                # unclassified or building classification
                # Classification info at:
                # https://www.usgs.gov/ngp-standards-and-specifications/lidar-base-specification-tables
                filter_class = (points.classification == 1) | (
                    points.classification == 6)
                filtered_points.append(points[bbox & filter_class].array)
            if len(filtered_points) == 0:
                print("Given latitude, longitude coordinates are not " +
                      "within the LiDAR scan bounds.")
            # Combine the points together and create a new LasData object
            concatenated_points = np.concatenate(filtered_points)
            final_point_record.array = concatenated_points
            output_laz = laspy.LasData(header=f.header)
            output_laz.points = final_point_record
        return output_laz

    def preprocessPcd(self, laz_data, nb_neighbors=10, std_ratio=0.5):
        """
        Preprocess pcd (point cloud data), which are 3D coordinate points
        in the LiDAR data, and filters out noisy/outlier data.

        Parameters:
        -----------
        laz_data: laspy.LasData
            LasData LiDAR object that contains X,Y,Z coordinates.
        nb_neighbors: int
            Number of neighbors for outlier removal.
            Defaulted to 10.
        std_ratio : float
            Standard deviation ratio threshold for outlier removal. The lower
            the std_ratio, the more aggresive it is in remove outliers.
            Defaulted to 0.5.

        Returns:
        --------
        inlier_cloud: o3d.geometry.PointCloud or None
            Preprocessed pcd with noise and outlier points removed.
        """
        # Ensure that the inputs are of the correct type
        if not isinstance(laz_data, laspy.LasData):
            raise TypeError(
                "laz_data variable must be a laspy.LasData object.")
        if not isinstance(nb_neighbors, int) or \
                isinstance(nb_neighbors, bool):
            raise TypeError("nb_neighbors variable must be of type int.")
        if not isinstance(std_ratio, float):
            raise TypeError("std_ratio variable must be of type float.")
        # Make LasData object into PointCloud object
        point_clouds = np.column_stack(
            [laz_data["X"], laz_data["Y"], laz_data["Z"]])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_clouds)
        try:
            # Remove nan and infinite points
            pcd = pcd.remove_non_finite_points()
            # Remove duplicate points
            pcd = pcd.remove_duplicated_points()
            # Estimate normal vectors if pcd has none
            if not pcd.has_normals():
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=0.1, max_nn=30))
            # Align normal vectors in an upward direction
            pcd.orient_normals_to_align_with_direction([0., 0., 1.])
            # Remove noisy/outlier points
            cl, ind = pcd.remove_statistical_outlier(
                nb_neighbors=nb_neighbors, std_ratio=std_ratio)
            inlier_cloud = pcd.select_by_index(ind)
        except Exception:
            print("PCD is empty.")
            inlier_cloud = None
        return inlier_cloud

    def visualizePolygonOnLidar(self, pcd_polygon, pcd_lat_lon):
        """
        Visualizes polygon on top of rooftop lidar data within a latitude,
        longitude bounding box. This only works if a polygon (in EPSG:4326)
        was originally provided during class initialization.

        Parameters:
        -----------
        pcd_polygon: o3d.geometry.PointCloud
            Preprocessed and cropped PointCloud within the shapely polygon.
        pcd_lat_lon: o3d.geometry.PointCloud
            Preprocessed and cropped PointCloud within the latitude, longitude
            bounding box.

        Returns:
        --------
        None.
        """
        # Needs both polygon and lat_lon to visualize them on top each other
        if not (pcd_polygon and pcd_lat_lon):
            raise ValueError(
                "Please provide both polygon PointCloud data and " +
                "latitude, longitude bbox PointCloud data.")
        # Ensure that the inputs are of the correct type
        if not isinstance(pcd_polygon, o3d.geometry.PointCloud):
            raise TypeError(
                "pcd_polygon variable must be a o3d.geometry.PointCloud " +
                "object.")
        if not isinstance(pcd_lat_lon, o3d.geometry.PointCloud):
            raise TypeError(
                "pcd_lat_lon variable must be a o3d.geometry.PointCloud " +
                "object.")
        # Get shapely polygon and project them ontop of lidar data
        transformed_coords = []
        for lon, lat in list(self.polygon.exterior.coords):
            x, y = self.transformer.transform(lon, lat)
            x_coord = (x - self.offsets[0]) / self.scales[0]
            y_coord = (y - self.offsets[1]) / self.scales[1]
            # Get average z coord
            z_coord = np.mean(np.asarray(pcd_polygon.points)[:, 2])
            transformed_coords.append([x_coord, y_coord, z_coord])

        # Make LineSet to outline polygon
        ls_poly = o3d.geometry.LineSet()
        # Make polygon lines to visuzalize the outline of the polygon
        poly_lines = [[i, i+1] for i in range(len(transformed_coords)-1)]
        ls_poly.lines = o3d.utility.Vector2iVector(poly_lines)
        ls_poly.points = o3d.utility.Vector3dVector(transformed_coords)
        # Change pcd colors. Blue is rooftop, red is shapely polygon
        pcd_lat_lon.paint_uniform_color([0, 0, 1])
        pcd_polygon.paint_uniform_color([1, 0, 0])
        ls_poly.paint_uniform_color([1, 0, 0])
        # Visualize on one map
        o3d.visualization.draw_geometries([ls_poly,  pcd_polygon, pcd_lat_lon])
