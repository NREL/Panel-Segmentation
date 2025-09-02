from panel_segmentation.lidar.usgs_lidar_api import USGSLidarAPI
import pytest
import os
import shutil
import pandas as pd


@pytest.fixture()
def usgsLidarApiClass():
    """
    Generate an instance of the USGSLidarAPI() class to run unit
    tests on.
    """
    # Create an instance of the USGSLidarAPI() class.
    usgs_api = USGSLidarAPI()
    return usgs_api


@pytest.fixture
def sampleLatLon():
    """
    Contains a sample latitude, longitude value with its
    associated USGS dataset name.
    """
    dataset_name = ("USGS Lidar Point Cloud TX Central B1 2017" +
                    " stratmap17-50cm-3097513c1 LAS 2019")
    lat = 30.1478
    lon = -97.74105
    return dataset_name, lat, lon


@pytest.fixture
def sampleDatasetHtml():
    """
    Contains a sample format of the USGS project page, dataset, and
    metadata html files.
    """
    project_html = """
    <html>
        <body>
            <a href="USGS_LPC_TX_Central_B1_2017_LAS_2019/">
                USGS_LPC_TX_Central_B1_2017_LAS_2019
            </a>
        </body>
    </html>"""

    dataset_html = """
    <html>
        <body>
            <a href="browse/">browse/</a>
            <a href="laz/">laz/</a>
            <a href="metadata/">metadata/</a>
            <a href="0_file_download_links.txt">
                0_file_download_links.txt
            </a>
        </body>
    </html>"""

    metadata_html = """
    <html>
        <body>
            <a href="USGS_LPC_TX_Central_B1_2017_stratmap17_50cm_"
               "3097513c1_LAS_2019_meta.xml">
                USGS_LPC_TX_Central_B1_2017_stratmap17_50cm_
                3097513c1_LAS_2019_meta.xml
            </a>
        </body>
    </html>
    """
    return project_html, dataset_html, metadata_html


@pytest.fixture
def sampleXmlMetadata():
    """
    Contains a sample xml link and its metadata information.
    """
    xml_link = ("https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/" +
                "Elevation/LPC/Projects/" +
                "USGS_LPC_TX_Central_B1_2017_LAS_2019/" +
                "metadata/USGS_LPC_TX_Central_B1_2017_stratmap17_50cm_" +
                "3097513c1_LAS_2019_meta.xml")
    dataset_name = ("USGS Lidar Point Cloud TX Central B1 2017" +
                    " stratmap17-50cm-3097513c1 LAS 2019")
    laz_link = ("https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/" +
                "Elevation/LPC/Projects/USGS_LPC_TX_Central_B1_2017_LAS_2019" +
                "/laz/USGS_LPC_TX_Central_B1_2017" +
                "_stratmap17_50cm_3097513c1_LAS_2019.laz")
    bbox_west = -97.75019731999998
    bbox_east = -97.73417588599995
    bbox_north = 30.15643917600005
    bbox_south = 30.140498657000023
    scan_start_date = "2017/01/28"
    scan_end_date = "2017/03/22"
    metadata_dict = {"dataset_name": dataset_name,
                     "xml_link": xml_link,
                     "laz_link": laz_link,
                     "bbox_west": bbox_west,
                     "bbox_east": bbox_east,
                     "bbox_north": bbox_north,
                     "bbox_south": bbox_south,
                     "scan_start_date": scan_start_date,
                     "scan_end_date": scan_end_date}
    return metadata_dict


def testInitOutputFolderDoesNotExist(test_output_folder="test_output_folder"):
    """
    Tests that an output folder is created when initializing the USGSLidarAPI()
    class and that the base url is correct.
    """
    usgs_api = USGSLidarAPI(output_folder=test_output_folder)
    # Test if the output folder exists
    assert os.path.exists(test_output_folder)
    assert usgs_api.output_folder == test_output_folder
    # Test if the base url is correct
    assert usgs_api.base_url == ("https://rockyweb.usgs.gov/vdelivery/" +
                                 "Datasets/Staged/Elevation/LPC/Projects/")
    # Clean up output folder
    shutil.rmtree(test_output_folder)


def testAllXmlMetadataTypeErrors(usgsLidarApiClass):
    """
    Tests if TypeErrors are raised for incorrect variable types.
    """
    # Test for dataset_name type
    with pytest.raises(TypeError,
                       match="dataset_name variable must be of type string."):
        usgsLidarApiClass.getAllXmlMetadata(451236)
    # Test for thread_max_workers type
    with pytest.raises(
            TypeError,
            match="thread_max_workers variable must be of type int."):
        usgsLidarApiClass.getAllXmlMetadata("dataset_name", True)
    with pytest.raises(
            TypeError,
            match="thread_max_workers variable must be of type int."):
        usgsLidarApiClass.getAllXmlMetadata("dataset_name", "12")
    with pytest.raises(
            TypeError,
            match="log_output variable must be of type bool."):
        usgsLidarApiClass.getAllXmlMetadata("dataset_name", 12, "False")
    # Clean up output folder
    shutil.rmtree(usgsLidarApiClass.output_folder)


def testAllXmlMetadataFileExists(usgsLidarApiClass, capsys):
    """
    Tests if the output folder and file exists when running getAllXmlMetadata.
    If the file exists, the function returns a dataframe from the output
    parquet file.
    """
    # Create a test output folder and parquet file
    lidar_metadata_folder = os.path.join(
        usgsLidarApiClass.output_folder, "lidar_metadata")
    os.makedirs(lidar_metadata_folder, exist_ok=True)
    metadata_df = pd.DataFrame({"dataset_name": ["dataset_name"]})
    metadata_df.to_parquet(os.path.join(lidar_metadata_folder,
                                        "dataset_name.parquet"), index=False)
    # Test if test output folder exists
    result_df = usgsLidarApiClass.getAllXmlMetadata("dataset_name")
    assert os.path.exists(os.path.join(
        usgsLidarApiClass.output_folder, "lidar_metadata"))
    # Test if output parquet file exists
    assert os.path.exists(os.path.join(
        usgsLidarApiClass.output_folder,
        "lidar_metadata", "dataset_name.parquet"))
    # Assert that a dataframe is returned from the existing parquet file
    assert isinstance(result_df, pd.DataFrame)
    # Test if the print message is printed
    captured = capsys.readouterr()
    print_msg = ("dataset_name.parquet exists. " +
                 "Pulling metadata from parquet file.\n")
    assert captured.out == print_msg
    # Clean up output folder
    shutil.rmtree(usgsLidarApiClass.output_folder)


def testAllXmlMetadataFileDoesNotExist(usgsLidarApiClass, sampleDatasetHtml,
                                       sampleXmlMetadata, capsys, mocker):
    """
    Tests if a parquet file is created for a dataset that have not been
    pulled yet.
    """
    # Get a sample metadata dictionary returned from parsing an xml file
    metadata_dict = sampleXmlMetadata
    metadata_df = pd.DataFrame([metadata_dict])
    # Mock GET requests and get a response of sampleDatasetHtml
    _, _, metadata_html = sampleDatasetHtml
    mock_response = mocker.Mock()
    mock_response.content = metadata_html.encode()
    mocker.patch("requests.get", return_value=mock_response)
    # Mock running getOneXmlMetadata
    usgsLidarApiClass.getOneXmlMetadata = mocker.Mock(
        return_value=metadata_dict)
    # Mock using ThreadPoolExecutor
    mock_future = mocker.Mock()
    mock_future.result.return_value = metadata_dict
    mock_executor = mocker.Mock()
    mock_executor.submit.return_value = mock_future
    mock_executor.__enter__ = mocker.Mock(return_value=mock_executor)
    mock_executor.__exit__ = mocker.Mock(return_value=None)
    mocker.patch("concurrent.futures.ThreadPoolExecutor",
                 return_value=mock_executor)
    # Mock pandas dropping dups and sorting functions
    mocker.patch("pandas.DataFrame.drop_duplicates", return_value=metadata_df)
    mocker.patch("pandas.DataFrame.sort_values", return_value=metadata_df)
    # Test if getAllXmlMetadata returned a dataframe result with
    # mocked process
    dataset_name = metadata_df["dataset_name"].iloc[0]
    result_df = usgsLidarApiClass.getAllXmlMetadata(dataset_name)
    assert isinstance(result_df, pd.DataFrame)
    # Test if the print messages are printed
    captured = capsys.readouterr()
    pulling_file_print_msg = (f"{dataset_name}.parquet does not exist. " +
                              "Pulling metadata data from parquet file.\n")
    assert pulling_file_print_msg in captured.out
    generated_file_print_msg = f"{dataset_name}.parquet generated.\n"
    assert generated_file_print_msg in captured.out
    # Test if the output parquet file exists
    assert os.path.exists(os.path.join(
        usgsLidarApiClass.output_folder,
        "lidar_metadata", f"{dataset_name}.parquet"))
    # Clean up output folder
    shutil.rmtree(usgsLidarApiClass.output_folder)


def testGetOneXmlMetadataTypeErrors(usgsLidarApiClass, sampleXmlMetadata):
    """
    Tests if TypeErrors are raised for incorrect variable types.
    """
    full_xml_link = sampleXmlMetadata["xml_link"]
    # Test for full_xml_link type
    with pytest.raises(TypeError,
                       match="full_xml_link variable must be of type string."):
        usgsLidarApiClass.getOneXmlMetadata(816, True)
    with pytest.raises(TypeError,
                       match="log_output variable must be of type bool."):
        usgsLidarApiClass.getOneXmlMetadata(full_xml_link, "True")
    # Clean up output folder
    shutil.rmtree(usgsLidarApiClass.output_folder)


def testGetOneXmlMetadataResults(usgsLidarApiClass, sampleXmlMetadata, capsys):
    """
    Tests if a metadata dictionary is returned when running getOneXmlMetadata.
    """
    metadata_dict = sampleXmlMetadata
    # Tests if a dictionary is returned
    result_dict = usgsLidarApiClass.getOneXmlMetadata(
        metadata_dict["xml_link"])
    assert isinstance(result_dict, dict)
    # Test if the print message is printed
    captured = capsys.readouterr()
    print_msg = ("Extracting information from " +
                 f"{metadata_dict['dataset_name']}\n")
    assert captured.out == print_msg
    # Assert that the result returns the correct metadata information
    assert result_dict["dataset_name"] == metadata_dict["dataset_name"]
    assert result_dict["laz_link"] == metadata_dict["laz_link"]
    assert result_dict["bbox_west"] == metadata_dict["bbox_west"]
    assert result_dict["bbox_east"] == metadata_dict["bbox_east"]
    assert result_dict["bbox_north"] == metadata_dict["bbox_north"]
    assert result_dict["bbox_south"] == metadata_dict["bbox_south"]
    assert result_dict["scan_start_date"] == metadata_dict["scan_start_date"]
    assert result_dict["scan_end_date"] == metadata_dict["scan_end_date"]
    # Clean up output folder
    shutil.rmtree(usgsLidarApiClass.output_folder)


def testOneDatasetTypeErrors(usgsLidarApiClass):
    """
    Tests if TypeErrors are raised for incorrect variable types.
    """
    # Test for output_filename type
    with pytest.raises(TypeError,
                       match="current_url variable must be of type string."):
        usgsLidarApiClass.getOneDataset(4612, "a_path")
    with pytest.raises(TypeError,
                       match="current_path variable must be of type string."):
        usgsLidarApiClass.getOneDataset("a_url", 4612)
    with pytest.raises(TypeError,
                       match="log_output variable must be of type bool."):
        usgsLidarApiClass.getOneDataset("a_url", "a_path", "True")
    # Clean up output folder
    shutil.rmtree(usgsLidarApiClass.output_folder)


def testAllDatasetTypeErrors(usgsLidarApiClass):
    """
    Tests if TypeErrors are raised for incorrect variable types.
    """
    # Test for output_filename type
    with pytest.raises(
            TypeError,
            match="output_filename variable must be of type string."):
        usgsLidarApiClass.getAllDataset(4612)
    with pytest.raises(
            TypeError,
            match="log_output variable must be of type bool."):
        usgsLidarApiClass.getAllDataset("output_file.parquet", "True")
    # Clean up output folder
    shutil.rmtree(usgsLidarApiClass.output_folder)


def testOneDatasetResults(usgsLidarApiClass, sampleDatasetHtml, mocker):
    """
    Tests if a dictionary and list of tuples are returned when running
    getOneDataset.
    """
    # Mock requests and get a response of sampleDatasetHtml
    _, dataset_html, _ = sampleDatasetHtml
    mock_response = mocker.Mock()
    mock_response.content = dataset_html.encode()
    mocker.patch("requests.get", return_value=mock_response)
    # Test getOneDataset is a dictionary
    result_dict, subfolders = usgsLidarApiClass.getOneDataset(
        "current_url", "current_path")
    # Test if the result is a dictionary
    assert isinstance(result_dict, dict)
    # Test if the result is a list
    assert isinstance(subfolders, list)
    # Clean up output folder
    shutil.rmtree(usgsLidarApiClass.output_folder)


def testAllDatasetResults(usgsLidarApiClass, sampleDatasetHtml, mocker):
    """
    Tests if a dataframe is returned when running getAllDataset.
    """
    # Mock getOneDataset function to return an example data
    mock_dataset_dict = {"dataset_name": "test_dataset",
                         "deprecated_scans": False}
    mocker.patch.object(usgsLidarApiClass, "getOneDataset",
                        return_value=(mock_dataset_dict, []))
    # Test getAllDataset is a dataframe
    result_df = usgsLidarApiClass.getAllDataset()
    assert isinstance(result_df, pd.DataFrame)
    # Test that dataframe has two columns ("dataset_name" & "deprecated_scans")
    assert len(result_df.columns) == 2
    assert "dataset_name" in result_df.columns
    assert "deprecated_scans" in result_df.columns
    # Clean up output folder
    shutil.rmtree(usgsLidarApiClass.output_folder)


def testLocateLazFileByLatLonTypeErrors(usgsLidarApiClass, sampleLatLon):
    """
    Tests if TypeErrors are raised for incorrect variable types.
    """
    # Make sample data
    dataset_name, lat, lon = sampleLatLon
    metadata_df = pd.DataFrame({"dataset_name": [dataset_name],
                                "lat": [lat], "lon": [lon]})
    # Test for metadata_df type
    with pytest.raises(
            TypeError,
            match="metadata_df variable must be a pandas DataFrame."):
        usgsLidarApiClass.locateLazFileByLatLon(123456, 30.1478, -97.74105)
    # Test for latitude type
    with pytest.raises(TypeError,
                       match="latitude variable must be of type float."):
        usgsLidarApiClass.locateLazFileByLatLon(metadata_df, "412356", lon)
    # Test for longitude type
    with pytest.raises(TypeError,
                       match="longitude variable must be of type float."):
        usgsLidarApiClass.locateLazFileByLatLon(metadata_df, lat, True)
    # Clean up output folder
    shutil.rmtree(usgsLidarApiClass.output_folder)


def testLocateLazFileByLatLonNoneResults(usgsLidarApiClass,
                                         sampleXmlMetadata,
                                         mocker):
    """
    Tests if None is returned for a latitude, longitude that does not have any
    LiDAR data.
    """
    # Mock GET requests
    mocker.patch("requests.get")
    # Make sample metadata df
    metadata_df = pd.DataFrame([sampleXmlMetadata])
    # Test if locateLazFileByLatLon returns None for an out of bound lat, lon
    result = usgsLidarApiClass.locateLazFileByLatLon(
        metadata_df, 40.1478, -27.74105)
    assert result is None
    # Clean up output folder
    shutil.rmtree(usgsLidarApiClass.output_folder)


def testLocateLazFileByLatLonResults(usgsLidarApiClass, sampleLatLon,
                                     sampleXmlMetadata, mocker):
    """
    Tests if a string .laz link is returned when running locateLazFileByLatLon.
    """
    # Get sample lat, lon data within metadata_df
    _, lat, lon = sampleLatLon
    # Mock GET requests
    mocker.patch("requests.get")
    # Make sample metadata df
    metadata_df = pd.DataFrame([sampleXmlMetadata])
    # Test if locateLazFileByLatLon returns a string for an in bound lat, lon
    result = usgsLidarApiClass.locateLazFileByLatLon(metadata_df, lat, lon)
    assert isinstance(result, str)
    assert result == metadata_df["laz_link"].iloc[0]
    # Clean up output folder
    shutil.rmtree(usgsLidarApiClass.output_folder)


def testDownloadLazFileTypeErrors(usgsLidarApiClass):
    """
    Tests if TypeErrors are raised for incorrect variable types.
    """
    # Test for laz_link type
    with pytest.raises(TypeError,
                       match="laz_link variable must of type string."):
        usgsLidarApiClass.downloadLazFile(123456)
    # Clean up output folder
    shutil.rmtree(usgsLidarApiClass.output_folder)


def testDownloadLazFileExists(usgsLidarApiClass, sampleXmlMetadata,
                              mocker, capsys):
    """
    Tests if a string local .laz file path is returned when the .laz file
    already exists locally.
    """
    # Make a sample .laz file
    output_folder = os.path.join(usgsLidarApiClass.output_folder, "lidar_laz")
    os.makedirs(output_folder)
    laz_file_path = os.path.join(output_folder, "example.laz")
    with open(laz_file_path, "w") as f:
        f.write("an empty file")
    # Test if downloadLazFile returns the local laz file path
    result = usgsLidarApiClass.downloadLazFile("example.laz")
    assert result == laz_file_path
    assert isinstance(result, str)
    # Test if the print message is printed
    captured = capsys.readouterr()
    print_msg = "example.laz already downloaded.\n"
    assert captured.out == print_msg
    # Clean up output folder
    shutil.rmtree(output_folder)


def testDownloadLazFileDoesNotExist(usgsLidarApiClass, sampleXmlMetadata,
                                    mocker, capsys):
    """
    Tests if a .laz is downloaded and its local file path returned when
    the .laz file does not exists locally.
    """
    # Mock GET requests
    mocker.patch("requests.get")
    # Test if downloadLazFile downloads a .laz file
    result = usgsLidarApiClass.downloadLazFile(sampleXmlMetadata["laz_link"])
    assert isinstance(result, str)
    # Test if the .laz file has the correct file path
    laz_filename = os.path.basename(sampleXmlMetadata["laz_link"])
    local_laz_file_path = os.path.join(
        usgsLidarApiClass.output_folder, "lidar_laz", laz_filename)
    assert result == local_laz_file_path
    # Test if the print message is printed
    captured = capsys.readouterr()
    print_msg = (f"Downloaded {laz_filename}.\n")
    assert captured.out == print_msg
    # Clean up output folder
    shutil.rmtree(usgsLidarApiClass.output_folder)
