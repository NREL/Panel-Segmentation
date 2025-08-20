import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class USGSLidarAPI:
    '''
    A class that pulls LiDAR data from USGS.

    The dataset source is found at:
    https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/
    An interactive explorer is found at:
    https://apps.nationalmap.gov/lidar-explorer/#/
    '''

    def __init__(self, output_folder="data"):
        # Output folder to save the pulled LiDAR data
        self.output_folder = output_folder

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        # Base url where all dataset files are located
        self.base_url = ("https://rockyweb.usgs.gov/vdelivery/Datasets/" +
                         "Staged/Elevation/LPC/Projects/")

    def getAllXmlMetadata(self, dataset_name, thread_max_workers=12):
        """
        Gets xml formatted LiDAR metadata for a specific dataset name from
        the USGS website's metadata folder. Saves metadata information as
        a parquet file to save in memory to reference later.

        Parameters
        -----------
        dataset_name: string
            Name of USGS LiDAR dataset.
        thread_max_workers: int
            Maximum number of workers thread for parallelization.
            Defaulted to 12.

        Returns
        --------
        metadata_df: Pandas dataframe
            Pandas dataframe containing the LiDAR's metadata for a specific
            dataset. The dataframe "dataset_names", "xml_link", "laz_link",
            "bbox_west", "bbox_east", "bbox_north", "bbox_south",
            "scan_start_date", and "scan_end_date" columns.
        """
        # Ensure that the inputs are of the correct type
        if not isinstance(dataset_name, str):
            raise TypeError("dataset_name variable must be of type string.")
        if not isinstance(thread_max_workers, int) or \
                isinstance(thread_max_workers, bool):
            raise TypeError("thread_max_workers variable must be of type int.")
        # Check if metadata folder contains the metadata parquet for the
        # dataset
        metadata_folder = os.path.join(self.output_folder, "lidar_metadata")
        if not os.path.exists(metadata_folder):
            os.makedirs(metadata_folder)
        # Make output filename and filepath
        output_filename = dataset_name.replace(
            "/", "-").rstrip("-") + ".parquet"
        metadata_file_path = os.path.join(metadata_folder, output_filename)
        # If metadata parquet is already saved, get its data
        if os.path.exists(metadata_file_path):
            metadata_df = pd.read_parquet(metadata_file_path)
            print(f"{output_filename} exists." +
                  "Pulling metadata from parquet file.")
            return metadata_df
        # If metadata parquet is not already saved, generate one and save it
        else:
            print(f"{output_filename} does not exist. " +
                  "Pulling metadata data from parquet file.")
            # Make a master metadata list to keep track of lidar file metadata
            metadata_list = list()
            # Get lists of metadata xml files
            metadata_url = f"{self.base_url}{dataset_name}metadata/"
            response = requests.get(metadata_url)
            # xml files names are found in a href headers
            plain_text = response.content
            soup = BeautifulSoup(plain_text, "html.parser")
            xml_links = soup.find_all("a", href=lambda href: href and
                                      href.endswith(".xml"))
            full_xml_links = [
                f"{metadata_url}{link.get('href')}" for link in xml_links]
            metadata_list = []
            # Parallelize extracting xml
            # Without parallelization, it takes ~0.5s to read one xml file
            with ThreadPoolExecutor(max_workers=thread_max_workers) \
                    as executor:
                futures = []
                # Execute XML metadata requests to the thread pool and
                # get results
                for xml_link in full_xml_links:
                    future = executor.submit(
                        self.getOneXmlMetadata, xml_link)
                    futures.append(future)
                for future in as_completed(futures):
                    data = future.result()
                    metadata_list.append(data)
            # Make results into dataframe
            metadata_df = pd.DataFrame(
                metadata_list).drop_duplicates().sort_values(by="dataset_name")
            # Save to parquet
            metadata_df.to_parquet(metadata_file_path, index=False)
            print(f"{output_filename} generated.")
            time.sleep(15)
            return metadata_df

    def getOneXmlMetadata(self, full_xml_link):
        """
        Parses one xml file and gets the specific metadata information from
        that xml file.

        Parameters
        -----------
        full_xml_link: string
            Full xml url of the file for parsing.

        Returns
        --------
        metadata_dict: dictionary
            Dictionary containing the metadata information obtained from one
            xml file. The dictionary contains "dataset_names", "xml_link",
            "laz_link", "bbox_west", "bbox_east", "bbox_north", "bbox_south",
            "scan_start_date", and "scan_end_date" key names.
        """
        # Ensure that the inputs are of the correct type
        if not isinstance(full_xml_link, str):
            raise TypeError("full_xml_link variable must be of type string.")
        # Make persistent Retry HTTP sessions to fix connection issues
        session = requests.Session()
        retry = Retry(connect=6, backoff_factor=1)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        # Make GET resquest to full_xml_link and get its information
        xml_response = session.get(full_xml_link)
        root = ET.fromstring(xml_response.content)
        # Get data set title name
        title_name = root.find(".//title").text
        print(f"Extracting information from {title_name}")
        # Get latitude, longitude bounding box
        west = root.find(".//westbc").text
        east = root.find(".//eastbc").text
        north = root.find(".//northbc").text
        south = root.find(".//southbc").text
        # Get laz file download url
        laz_link = root.find(".//networkr").text
        # Get scan dates
        start_date = root.find(".//begdate").text
        end_date = root.find(".//enddate").text
        format_start_date = datetime.strptime(
            start_date, "%Y%m%d").strftime("%Y/%m/%d")
        format_end_date = datetime.strptime(
            end_date, "%Y%m%d").strftime("%Y/%m/%d")
        # Put all extracted information into metadata_dict
        metadata_dict = {
            "dataset_name": title_name,
            "xml_link": full_xml_link,
            "laz_link": laz_link,
            "bbox_west": float(west),
            "bbox_east": float(east),
            "bbox_north": float(north),
            "bbox_south": float(south),
            "scan_start_date": format_start_date,
            "scan_end_date": format_end_date
        }
        return metadata_dict

    def getAllDataset(
            self,
            output_filename="master_usgs_lidar_dataset_names.parquet"):
        """
        Pulls all the LiDAR dataset names by scanning folders in USGS
        Projects page that contains laz and metadata subfolders.
        This only needs to run once to generate a
        master_usgs_lidar_dataset_names.parquet that will be saved in memory.
        These datasets are located at:
        https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/

        Parameters
        -----------
        output_filename: string
            Path and filename for the output parquet file containing all
            dataset names found in USGS LiDAR website.
            Defaulted to "master_usgs_lidar_dataset_names.parquet".

        Returns
        --------
        dataset_df: Pandas dataframe
            Pandas dataframe containing all dataset names found in USGS LiDAR
            website. The dataframe contains "dataset_names" and
            "deprecated_scans" columns.
        """
        # Ensure that the inputs are of the correct type
        if not isinstance(output_filename, str):
            raise TypeError("output_filename variable must be of type string.")
        # Make a master list of all LiDAR USGS dataset names
        dataset_list = []
        # Keep track of which url that already ran
        url_list = [(self.base_url, "")]
        # Iterate through each parent folders until we get to the dataset
        # subfolders containing laz and metadata folders
        while url_list:
            current_url, current_path = url_list.pop()
            # Make a request to the url
            response = requests.get(current_url)
            # Dataset names are found in href headers with "/" at the end
            soup = BeautifulSoup(response.content, "html.parser")
            # Ignore certain href that are not dataset names
            links = soup.find_all("a", href=lambda href:
                                  href and
                                  href.endswith("/") and
                                  not href.startswith("https://") and
                                  href not in ["../", "./"])
            # Look for laz/ and metadata/ folders if it exists
            laz_links = [
                link for link in links if "laz/" in link["href"].lower()]
            metadata_links = [
                link for link in links if "metadata/" in link["href"].lower()]
            # If there exists laz and metadata folder
            # then add the dataset folder name to dataset list
            if laz_links and metadata_links:
                print("Dataset found: ", current_path)
                # Keep track of deprecated scans, which may have different xml
                # and lidar formats
                if "legacy/" in current_path:
                    dataset_list.append({"dataset_name": current_path,
                                        "deprecated_scans": True})
                else:
                    dataset_list.append({"dataset_name": current_path,
                                        "deprecated_scans": False})
            # Keep track of deprecated scans, which may not have metadata
            # in the metadata folders
            elif laz_links and not metadata_links:
                print("Dataset found: ", current_path)
                dataset_list.append({"dataset_name": current_path,
                                    "deprecated_scans": True})
            else:
                # Add subfolders url to the next iteration
                # to check if they contain laz and metadata folders
                for link in links:
                    subfolder_name = link["href"]
                    new_url = f"{current_url}{subfolder_name}"
                    if current_path:
                        new_path = f"{current_path}{subfolder_name}"
                    else:
                        new_path = subfolder_name
                    url_list.append((new_url, new_path))
        # Save to parquet file
        dataset_df = pd.DataFrame(dataset_list)
        dataset_df.to_parquet(os.path.join(self.output_folder,
                                           output_filename), index=False)
        return dataset_df

    def locateLazFileByLatLon(self, metadata_df, latitude, longitude):
        """
        Locates the latest .laz file link for the given latitude, longitude.

        Parameters
        -----------
        metadata_df: Pandas dataframe
            Pandas dataframe containing the LiDAR's metadata for all
            datasets. The dataframe contains "dataset_names", "xml_link",
            "laz_link", "bbox_west", "bbox_east", "bbox_north", "bbox_south",
            "scan_start_date", and "scan_end_date" columns.
        latitude: float
            The requested latitude coordinate.
        longitude: float
            The requested longitude coordinate.

        Returns
        --------
        laz_download_link: string or None
            Download link to the laz file containing the latitude, longitude
            coordinates. Otherwise, returns None if no file is found.
        """
        # Ensure that the inputs are of the correct type
        if not isinstance(metadata_df, pd.DataFrame):
            raise TypeError("metadata_df variable must be a pandas DataFrame.")
        if not isinstance(latitude, float):
            raise TypeError("latitude variable must be of type float.")
        if not isinstance(longitude, float):
            raise TypeError("longitude variable must be of type float.")
        # Filter metadata_df for bbox that contains lat, lon coordinates
        lat_lon_df = metadata_df[((metadata_df["bbox_south"] <= latitude) &
                                  (latitude <= metadata_df["bbox_north"]) &
                                  (metadata_df["bbox_west"] <= longitude) &
                                  (longitude <= metadata_df["bbox_east"]))]
        # Check if LiDAR data exists
        if len(lat_lon_df) > 0:
            # If there are multiple results, get the latest lidar dataset
            lat_lon_df["scan_end_date"] = pd.to_datetime(
                lat_lon_df["scan_end_date"])
            lat_lon_df = lat_lon_df.sort_values(
                "scan_end_date", ascending=False).iloc[[0]]
            # Get laz link
            laz_download_link = lat_lon_df["laz_link"].iloc[0]
        # If no LiDAR data exits, return None
        else:
            print(f"No LiDAR data at: {latitude}, {longitude}")
            laz_download_link = None
        return laz_download_link

    def downloadLazFile(self, laz_link, chunk_size=8192):
        """
        Downloads the .laz file from the USGS browser given the link.

        Parameters
        -----------
        laz_link: string
            Download link to the laz file
        chunk_size: int
            Byte size of chunks to download. Defaulted to 8192.

        Returns
        --------
        laz_file_path: string
            File path of the downloaded .laz file.
        """
        # Ensure that the inputs are of the correct type
        if not isinstance(laz_link, str):
            raise TypeError("laz_link variable must of type string.")
        # Check if laz download folder exists
        laz_folder = os.path.join(self.output_folder, "lidar_laz")
        if not os.path.exists(laz_folder):
            os.makedirs(laz_folder)
        # Create output laz filename
        laz_filename = laz_link.split("/")[-1]
        laz_file_path = os.path.join(laz_folder, laz_filename)
        # Check if file already exists
        if os.path.exists(laz_file_path):
            print(f"{laz_filename} already downloaded.")
            return laz_file_path
        # Download file if it does not exist
        else:
            # Prevents high traffic to website if downloading consecutive files
            time.sleep(10)
            # Stream download the file with chunking since files are large
            with requests.get(laz_link, stream=True) as r:
                r.raise_for_status()
                with open(laz_file_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
            print(f"Downloaded {laz_filename}.")
            return laz_file_path
