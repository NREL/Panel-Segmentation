import pytest
import os
from panel_segmentation.lidar.pcd_data import PCD
from shapely.wkt import loads
import laspy
import open3d as o3d
import numpy as np

example_path = os.path.join("panel_segmentation", "examples", "lidar_examples")


@pytest.fixture
def pcdParams():
    """
    Contains the proper variable types for running PCD class.
    """
    laz_file_path = os.path.join(
        example_path,
        "cropped_USGS_LPC_TX_Central_B1_2017_stratmap17_50cm_3097513c1" +
        "_LAS_2019.laz")
    lat = 30.1478
    lon = -97.74105
    poly = loads("""POLYGON ((-97.74106315632842 30.14782268767431,
                 -97.74096919534045 30.1478079931521,
                 -97.74096768230025 30.1478196557987,
                 -97.74099649481839 30.14782495927133,
                 -97.74099415225778 30.1478379364854,
                 -97.74103533818115 30.14784324186061,
                 -97.74103979247948 30.14783185545246,
                 -97.74105752041777 30.14783384726395,
                 -97.74106315632842 30.14782268767431))""")
    return laz_file_path, lat, lon, poly


@pytest.fixture
def pcdClassPolygon(pcdParams):
    """
    A PCD class with a shapely polygon.
    """
    laz_file_path, _, _, poly = pcdParams
    return PCD(laz_file_path, poly)


@pytest.fixture
def pcdClassLatLon(pcdParams):
    """
    A PCD class with a lat, lon center point.
    """
    laz_file_path, lat, lon, _ = pcdParams
    return PCD(laz_file_path, None, lat, lon)


@pytest.fixture
def lazData(pcdParams):
    """
    A sample laz data.
    """
    laz_file_path, _, _, _ = pcdParams
    # read in .laz file
    with laspy.open(laz_file_path) as laz_reader:
        laz_data = laz_reader.read()
    return laz_data


@pytest.fixture
def op3dData(lazData):
    """
    A sample o3d data from the sample laz data.
    """
    point_clouds = np.column_stack(
        [lazData["X"], lazData["Y"], lazData["Z"]])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_clouds)
    return pcd


def testInitPolyAndLatLonValueErrors(pcdParams):
    """
    Tests if a ValueError is raised when both shapely polygon and lat, lon are
    inputs.
    """
    laz_file_path, lat, lon, poly = pcdParams
    # Test if ValueError is raised
    with pytest.raises(
            ValueError,
            match="Please input a shapely polygon or latitude, longitude " +
            "center coordinates to generate a bbox to get pcd data. " +
            "Do not enter both."):
        PCD(laz_file_path, poly, lat, lon)


def testInitNoPolyOrLatLonValueErrors(pcdParams):
    """
    Tests if a ValueError is raised when no shapely polygon or lat, lon are
    inputs.
    """
    laz_file_path, _, _, _ = pcdParams
    # Test if ValueError is raised
    with pytest.raises(ValueError,
                       match="No shapely polygon or latitude, longitude " +
                       "center coordinates were input. Please input a " +
                       "shapely polygon or latitude, longitude center " +
                       "coordinates to generate a bbox to get pcd data."):
        PCD(laz_file_path, None, None, None)
    with pytest.raises(ValueError,
                       match="No shapely polygon or latitude, longitude " +
                       "center coordinates were input. Please input a " +
                       "shapely polygon or latitude, longitude center " +
                       "coordinates to generate a bbox to get pcd data."):
        PCD(laz_file_path)


def testReadCropLazTypeErrors(pcdClassPolygon):
    """
    Tests if TypeErrors are raised for incorrect variable types.
    """
    # Test if lat_lon_bbox is correct
    with pytest.raises(
            TypeError,
            match="lat_lon_bbox_size variable must be of type int."):
        pcdClassPolygon.readCropLaz(lat_lon_bbox_size=1224.3)


def testReadCropLazPolygon(pcdClassPolygon):
    """
    Tests if the returned laz data is a laspy.LasData object after running
    readCropLaz for a shapely polygon.
    """
    result = pcdClassPolygon.readCropLaz()
    # Assert that the result is a laspy.LasData object
    assert isinstance(result, laspy.LasData)


def testReadCropLazLatLon(pcdClassLatLon):
    """
    Tests if the returned laz data is a laspy.LasData object after running
    readCropLaz for a lat, lon center point.
    """
    result = pcdClassLatLon.readCropLaz()
    # Assert that the result is a laspy.LasData object
    assert isinstance(result, laspy.LasData)


def testPreprocessPcdTypeErrors(pcdClassPolygon, lazData):
    """
    Tests if TypeErrors are raised for incorrect variable types.
    """
    # Test if laz_data is correct
    with pytest.raises(
            TypeError,
            match="laz_data variable must be a laspy.LasData object."):
        pcdClassPolygon.preprocessPcd(4234)
    # Test if nb_neighbors is correct
    with pytest.raises(TypeError,
                       match="nb_neighbors variable must be of type int."):
        pcdClassPolygon.preprocessPcd(lazData, nb_neighbors=4234.6)
    # Test if std_ratio is correct
    with pytest.raises(TypeError,
                       match="std_ratio variable must be of type float."):
        pcdClassPolygon.preprocessPcd(lazData, std_ratio=1)


def testPreprocessPcdResult(pcdClassPolygon, lazData):
    """
    Tests if the returned pcd resultis an o3d.geometry.PointCloud object
    after running preprocessPcd.
    """
    result = pcdClassPolygon.preprocessPcd(lazData)
    # Assert that the result is a o3d.geometry.PointCloud object
    assert isinstance(result, o3d.geometry.PointCloud)


def testPreprocessPcdNoneResult(pcdClassPolygon, capsys):
    """
    Tests if the returned pcd result is None when the pcd is empty.
    """
    # Make an empty laspy.LasData object pcd
    empty_las = laspy.create()
    result = pcdClassPolygon.preprocessPcd(empty_las)
    # Assert that the result is None
    assert result is None
    # Test if the print message is printed
    captured = capsys.readouterr()
    print_msg = "PCD is empty.\n"
    assert print_msg in captured.out


def testVisualizePolygonOnLidarValueErrors(pcdClassPolygon, op3dData):
    """
    Tests if a ValueError is raised when both shapely polygon and lat, lon
    are not inputs when running visualizePolygonOnLidar.
    """
    # Test if ValueError is raised when no shapely polygon is input
    with pytest.raises(
            ValueError,
            match="Please provide both polygon PointCloud data and " +
            "latitude, longitude bbox PointCloud data."):
        pcdClassPolygon.visualizePolygonOnLidar(op3dData, None)
    # Test if ValueError is raised when no lat, lon is input
    with pytest.raises(ValueError,
                       match="Please provide both polygon PointCloud data " +
                       "and latitude, longitude bbox PointCloud data."):
        pcdClassPolygon.visualizePolygonOnLidar(None, op3dData)


def testVisualizePolygonOnLidarTypeErrors(pcdClassPolygon, op3dData):
    """
    Tests if TypeErrors are raised for incorrect variable types.
    """
    # Test if pcd_polygon is correct
    with pytest.raises(TypeError,
                       match="pcd_polygon variable must be a " +
                       "o3d.geometry.PointCloud object."):
        pcdClassPolygon.visualizePolygonOnLidar(834, op3dData)
    # Test if pcd_lat_lon is correct
    with pytest.raises(TypeError,
                       match="pcd_lat_lon variable must be a " +
                       "o3d.geometry.PointCloud object."):
        pcdClassPolygon.visualizePolygonOnLidar(op3dData, "asdk")


def testVisualizePolygonOnLidarResult(pcdClassPolygon, op3dData, mocker):
    """
    Tests if the returned None is returned after running
    visualizePolygonOnLidar.
    """
    # Mock pyproj transformer function
    mock_transformer = mocker.Mock()
    mock_transformer.transform.return_value = (10.0, 30.0)
    # Make sample pcd's transform, scales, and offsets attributes
    pcdClassPolygon.transformer = mock_transformer
    pcdClassPolygon.scales = [0.001, 0.001, 0.001]
    pcdClassPolygon.offsets = [0.0, 0.0, 0.0]
    # Mock o3d visualization
    mock_draw = mocker.patch('open3d.visualization.draw_geometries')
    mock_draw.return_value = None
    # Run visualizePolygonOnLidar function
    result = pcdClassPolygon.visualizePolygonOnLidar(op3dData, op3dData)
    # Assert that the result is None
    assert result is None
