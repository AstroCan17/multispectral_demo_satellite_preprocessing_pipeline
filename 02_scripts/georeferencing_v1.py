# created by Cando17 on 2023-12-02, 20:14

import ee
from tkinter import filedialog
from datetime import datetime,timezone
from urllib.request import urlretrieve
import webbrowser
import os 
import tkinter as tk
import zipfile
import json
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
import pandas as pd
import requests
from pyorbital.orbital import tlefile,Orbital
from skyfield.api import EarthSatellite, Topos, load, utc
import math
import cv2
from dateutil.parser import parse
import rasterio as rio
import numpy as np
# import gdal
# from gdalconst import *
from osgeo import osr
from osgeo import gdal
from osgeo import gdalconst


################## Define the area of interest #####################
class getSatelliteInfo:
    def __init__(self):
        self.satellite_id = "Colombus_T2"
        self.lat = globals()['lat']
        self.lon = globals()['lon']
        if self.lat is None:
            self.lat = self.get_satellite_info()[0]
        else:
            pass
        if self.lon is None:    
            self.lon = self.get_satellite_info()[1]
        else:
            pass
        self.formatted_capture_time = self.read_capture_time()
        

    def read_metadata_caiman(self,orbit_path):
        # write a code to read the .txt file in orbit_path
        for file_name in os.listdir(orbit_path):
            if file_name.endswith('.txt'):
                with open(os.path.join(orbit_path, file_name), 'r') as f:
                    metadata = json.load(f)
        yield metadata
    
    def read_capture_time(self):
        path = r"D:\02_plans\Photo_DataBase.xlsx"
        df = pd.read_excel(path)
        ankara_row = df[df['Naming'] == 'Somewhere']
        capture_time = ankara_row['Capture Time (UTC)'].values[0]
        capture_time = pd.Timestamp(capture_time)
        capture_time_out = capture_time.strftime('%Y-%m-%d %H:%M:%S')
        data_search_time = capture_time.strftime('%Y-%m-%d')

        return capture_time_out,data_search_time
    
    

    def get_tle(self):
        # Celestrak'tan uydu TLE verilerini çeken fonksiyon
        url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"  # Celestrak TLE veri URL'si
        response = requests.get(url)

        if response.status_code == 200:
            lines = response.text.split('\n')
            for i in range(0, len(lines), 3):
                if self.satellite_id in lines[i]:
                    return lines[i:i+3]
        return None


    def get_satellite_info(self):
        # tle = getSatelliteInfo.get_tle(self.satellite_id)
        # Line1 = tle[1]
        # Line2 = tle[2]

        Line1 = "1 56210U 23054AJ  24133.57278421  .00022293  00000+0  69995-3 0  9995"
        Line2 = "2 56210  97.3638  30.2367 0008588 276.5063  83.5202 15.32993628 60623"
        time_str,data_search_time = getSatelliteInfo.read_capture_time(self)
        ts = load.timescale()
        time = ts.utc(datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=utc))
        # print("flag2")
        # print("Time: ", time)
        satellite_ = EarthSatellite(Line1, Line2)
        geometry = satellite_.at(time)
        subpoint = geometry.subpoint()
        elevation = subpoint.elevation
        altitude = float(format(elevation.km))
        pixelsize = 5.5 * math.pow(10, -9)  # Caiman pixel size in km
        focalL = 845 * math.pow(10, -6)  # Caiman focal length in km
        GSD = (altitude * pixelsize) / focalL
        mu = 3.986004418 * math.pow(10, 5)  # Standard Gravitational Parameter
        earthRadius = 6371  # in km
        TransV = math.sqrt(mu / (earthRadius + altitude))  # Translational velocity km/s
        AngularRofOrbit = TransV / (earthRadius + altitude)  # Angular Rate of Orbit rad/s
        GroundTrackV = earthRadius * AngularRofOrbit  # Ground Track Velocity km/s
        self.lat = subpoint.latitude.degrees
        self.lon = subpoint.longitude.degrees

        return self.lat, self.lon
    



class referenceDownload():
    def __init__(self):
        self.info = getSatelliteInfo()

        self.lat = self.info.lat
        self.lon = self.info.lon
        self.formatted_capture_time = self.info.formatted_capture_time

        

        # for item in globals().items():
        #     print(item)

        self.coreg_rgb_img = globals()['coreg_rgb_img']
        self.save_path = globals()['save_path']
        self.width = globals()['width']
        self.height = globals()['height']
        if self.width is not None:
            self.width = self.width
            self.height = self.width
        else:
            self.width,self.height = self.roi_size()
        self.selected_bands = ['B2','B3','B4']
        self. resolution_dict = {band: 10 if band in ["B2", "B3", "B4", "B8"] else (60 if band in ["B1", "B9", "B10"] else 20) for band in self.selected_bands}
        self.roi = self.create_roi()
        self.region = self.roi.toGeoJSONString()
        self.start_date = self.info.read_capture_time()[1]
        self.end_date = input("write an end_date for searching in format YYYY-MM-DD (leave blank for today's date): ")
        if self.end_date == "":
            self.end_date = datetime.today().strftime('%Y-%m-%d')
        self.image_name = self.sentinel2_collection_search()[1]
        # print("Image name: ", self.image_name) 
        self.image, self.crs = self.get_image_crs()

        self.download_dir = self.unzip_directory()


        
  

    def print_attributes(self):
        for attr, value in self.__dict__.items():
            print(f"{attr}: {value}")
    

    def roi_size(self): 


        gsd_pan = 3.25 #meters
        gsd_ms = gsd_pan*2

        dummy = self.coreg_rgb_img.copy()
        # dummy = cv2.imread(orbit_path_ms, cv2.IMREAD_UNCHANGED)
        dummy_ms_size = dummy.shape
        del dummy
        width, height = dummy_ms_size[1], dummy_ms_size[0]
        width = width*gsd_ms/1000 #in km 
        height = height*gsd_ms/1000 # in km 

        return width,height

    def create_roi(self):
        # Get width and height, defaulting to roi_size values if None
        width, height = (self.width, self.height) if (self.width and self.height) else referenceDownload.roi_size(self)
        width = self.width if self.width is not None else width
        height = self.height if self.height is not None else height

        # Get latitude and longitude, defaulting to get_satellite_info values if None
        lat, lon = (self.lat, self.lon) if (self.lat and self.lon) else getSatelliteInfo.get_satellite_info(self)
        lat = self.lat if self.lat is not None else lat
        lon = self.lon if self.lon is not None else lon

        print("inside of create_roi function")
        print(f"Latitude: {lat}")
        print(f"Longitude: {lon}")
        print(f"Width: {width}")
        print(f"Height: {height}")

        # Convert km to degrees
        width_in_degrees = width / 111.32
        height_in_degrees = height / 111.32

        # Define the ROI coordinates
        roi_coords_wgs84 = (
            lon - width_in_degrees / 2,
            lat - height_in_degrees / 2,
            lon + width_in_degrees / 2,
            lat + height_in_degrees / 2
        )
        roi = ee.Geometry.Rectangle(roi_coords_wgs84)
        return roi



    ################## Search for images and getting names #####################

    def sentinel2_collection_search(self):
        se2_l2 = ee.ImageCollection('COPERNICUS/S2_SR').filterDate(self.start_date,self.end_date).filterBounds(self.roi).filter(
        ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE",10))
       
        image_names = [image['id'] for image in se2_l2.getInfo()['features']]
        print(f"There are {len(image_names)} images in the collection")

        # print the images on the list by order
        for image_name in image_names:
            print(f" - {image_name}")
        
        answer  = input("Do you want to search again? (y/n): ")
        if answer == "y":
            # input_start = input("write a start_date for searching in format YYYY-MM-DD: ")
            input_end = input("write an end_date for searching in format YYYY-MM-DD (leave blank for today's date): ")
            self.end_date = input_end
            self.sentinel2_collection_search()
        else:
            

            print("Ok, we will continue")
        image_name = input("write an image name from the list above: ")

        return se2_l2,image_name


    ################## Getting image  #####################


    def get_image_crs(self):
        image = ee.Image(self.image_name)
        available_bands = image.bandNames().getInfo()
        
        image_all_bands = image.select(available_bands)
        print("Available bands: " + ", ".join(available_bands[:12]))
        
        # selected_bands = [band.strip() for band in input("Enter the bands you want to select (comma-separated): ").split(",")]
        selected_bands = self.selected_bands
        print("Selected bands: " + ", ".join(selected_bands))
        # choosen image
        image = image.select(selected_bands)
        
        print("Image is ready for the launch.")
        image_crs = image_all_bands.select(['B2','B3','B4'])
        crs = image_crs.projection().crs().getInfo()
        print("CRS:", crs)
        print("Sending...")

        return image,crs

    @staticmethod
    def unzip_directory():
        root = tk.Tk()
        root.withdraw()
        folder_selected = filedialog.askdirectory()
        # change working dir to the selected folder
        os.chdir(folder_selected)
        return folder_selected
    
    @staticmethod
    def unzipper(folder_selected):
        print("Unzipping...")
        for file_name in os.listdir(folder_selected):
            if file_name.endswith('.zip'):
                with zipfile.ZipFile(os.path.join(folder_selected, file_name), 'r') as zip_ref:
                    zip_ref.extractall(folder_selected)
                os.remove(os.path.join(folder_selected, file_name))
        print("Done")


    @staticmethod
    def get_metadata(image):
        metadata = image.select().getInfo()
        return metadata

    @staticmethod
    def save_metadata(metadata, folder_selected):
        with open(os.path.join(folder_selected, "metadata.txt"), "w") as f:
            f.write(json.dumps(metadata, sort_keys=True, indent=4))
        print("Metadata saved as metadata.txt in the selected folder")

    
    

    ##################### Download Functions ############################

    def download(self):
        """ Downloads the image from the Google Earth Engine servers. 
            This is the main downlaod function.

        Args:
            selected_bands (_type_): _description_
            image (_type_): _description_
            region (_type_): _description_
            crs (_type_): _description_
        """    
       


        print("Please Select a folder to download the image")
        folder_selected = self.download_dir
        metadata_dict = {}
        for bands in self.selected_bands:
            url = self.download_image(self.image.select(bands), self.region,self.crs)
            if url is None:
                raise ValueError("URL is not defined. Cannot download the image.")

            print(url)
            print("Downloading...")
            webbrowser.open(url)
            # usage
            metadata_dict[bands] = self.get_metadata(self.image.select(bands))
            self.save_metadata(metadata_dict[bands], folder_selected)
            # write a code to wait until downloading completed
            
        answer = input("Do you want to unzip the files? (y/n): ")
        if answer == "y":
            print("Unziping...")
            self.unzipper(folder_selected)
        else:
            print("Anyway, I will unzip the files...")
            self.unzipper(folder_selected)
              
        # self.image, self.crs, self.selected_bands

    def download_image(self,image, region, crs):
        """
        This is the download function linked to previous function.

        """ 

        resolution = 10
        try:
            if isinstance(image, ee.image.Image):
                print('Its Image')
            elif isinstance(image, ee.imagecollection.ImageCollection):
                print('Its ImageCollection')
                exit()
            url = image.getDownloadUrl({
                'scale': resolution,
                'crs': crs,
                'region': region
            })
            return url
        except:
            print("Could not download")




from osgeo import gdal,ogr,osr

class geoReferencing():
    def __init__(self):
        # self.folder_selected = os.getcwd()
        self.folder_sentinel = globals()['save_path'] + "/05_sentinel"
        self.colombus_raw_bands = globals()['save_path'] + "/02_denoised/16b_images"

        self.colombus_bands_path_dict = {}
        self.colombus_band_dict = {}
        self.sentinel_path_dict = {}
        self.crs = None
        base_path = globals()['save_path']
        self.output_path = os.path.abspath(os.path.join(base_path, "06_georeferenced"))
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        if os.path.exists(globals()['save_path'] + "/06_georeferenced"):
            # self.output_path = globals()['save_path'] + "/06_georeferenced"
            self.output_path = os.path.abspath(os.path.join(base_path, "06_georeferenced"))
        else:
            print("Creating a folder for georeferenced images")
            os.mkdir(globals()['save_path'] + "/06_georeferenced")
        # self.output_path = globals()['save_path'] + "/06_georeferenced"
        # self.output_path = os.path.abspath(os.path.join(base_path, "06_georeferenced"))
        

        if os.path.exists(globals()['save_path'] + "/02_denoised/16b_images/colombus_wiener_deconv_b9.tif"):
            self.colombus_nir_path = globals()['save_path'] + "/02_denoised/16b_images/colombus_wiener_deconv_b9.tif"
        else:
            self.colombus_nir = None
        self.output_file = os.path.join(self.output_path, f'registered_b4_v6.tif')


        

    def get_data(self):
        for attr, value in self.__dict__.items():
            print(f"{attr}: {value}")

    
    def get_colombus_raw_image(self):
        # Collect paths for all relevant bands
        for file_name in os.listdir(self.colombus_raw_bands):
            if file_name.endswith('.tif'):
                band_name = file_name.split('.')[0][-2:]
                if band_name in ['b2', 'b3', 'b4', 'b6', 'b9']:
                    self.colombus_bands_path_dict[band_name] = os.path.join(self.colombus_raw_bands, file_name)

        # Yield each band as it is read
        with rio.open(self.colombus_bands_path_dict.get('b2')) as src2, \
             rio.open(self.colombus_bands_path_dict.get('b3')) as src3, \
             rio.open(self.colombus_bands_path_dict.get('b4')) as src4, \
             rio.open(self.colombus_bands_path_dict.get('b6')) as src6:

            yield 'b2', src2.read(1)
            yield 'b3', src3.read(1)
            yield 'b4', src4.read(1)
            yield 'b6', src6.read(1)

            if 'b9' in self.colombus_bands_path_dict:
                with rio.open(self.colombus_bands_path_dict['b9']) as src9:
                    yield 'b9', src9.read(1)
            else:
                yield 'b9', None
    

    
    def get_sentinel_downloaded(self):
        for file_name in os.listdir(self.folder_sentinel):
            if file_name.endswith('.tif'):
                # split the second . and get the band name
                band_name = "S2AB_" + file_name.split('.')[1]
                sentinel_path = os.path.join(self.folder_sentinel, file_name)
                self.sentinel_path_dict[band_name] = sentinel_path
        
        src2 = gdal.Open(self.sentinel_path_dict.get('S2AB_B2'))
        src3 = gdal.Open(self.sentinel_path_dict.get('S2AB_B3'))
        src4 = gdal.Open(self.sentinel_path_dict.get('S2AB_B4'))
        
        self.crs = src2.GetProjection()
        self.sentinel_transform = src2.GetGeoTransform()
        
        yield 'S2AB_b2', src2.GetRasterBand(1).ReadAsArray()
        yield 'S2AB_b3', src3.GetRasterBand(1).ReadAsArray()
        yield 'S2AB_b4', src4.GetRasterBand(1).ReadAsArray()

    
        
    def band_registration(self):
        sentinel_bands = {name: band for name, band in self.get_sentinel_downloaded()}
        colombus_bands = {name: band for name, band in self.get_colombus_raw_image()}

        # Example usage of the retrieved bands
        sentinel_b4 = sentinel_bands.get('S2AB_b4')
        colombus_b6 = colombus_bands.get('b4')
        def normalize_8bit(channel):
            return (((channel - channel.min()) / (channel.max() - channel.min())) * 255).astype('uint8')
        
        sentinel_b4_8b = normalize_8bit(sentinel_b4)
        colombus_b6_8b = normalize_8bit(colombus_b6)

        def apply_clahe(channel):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(channel)

        sentinel_b4_8b_str = apply_clahe(sentinel_b4_8b)
        colombus_b6_8b_str = apply_clahe(colombus_b6_8b)
        print("Clahe applied to the channels")

        del sentinel_b4_8b, colombus_b6_8b

        sift = cv2.SIFT_create()
        print("SIFT object created")

        keypoints_sentinel, descriptors_sentinel = sift.detectAndCompute(sentinel_b4_8b_str, None)
        keypoints_colombus_b6, descriptors_colombus_b6 = sift.detectAndCompute(colombus_b6_8b_str, None)

        del sentinel_b4_8b_str, colombus_b6_8b_str

        matcher = cv2.FlannBasedMatcher()
        matches_b6 = matcher.match(descriptors_colombus_b6, descriptors_sentinel)
        matches_b6 = sorted(matches_b6, key=lambda x: x.distance)

        num_matches_pan = int(len(matches_b6) * 0.1)
        matches_b6 = matches_b6[:num_matches_pan]

        src_pts_b6 = np.float32([keypoints_colombus_b6[m.queryIdx].pt for m in matches_b6]).reshape(-1, 1, 2)
        dst_pts_b6 = np.float32([keypoints_sentinel[m.trainIdx].pt for m in matches_b6]).reshape(-1, 1, 2)
        #  the transformation matrix using RANSAC
        M_pan,mask_pan = cv2.findHomography(src_pts_b6, dst_pts_b6, cv2.RANSAC, 5.0)
        

        # Warp the colombus_b6 band to align with the sentinel band
        height, width = sentinel_b4.shape
        print("shaoe of the colombus_b4 before warping: ", colombus_b6.shape)
        warped_colombus_b6 = cv2.warpPerspective(colombus_b6, M_pan, (width, height))
        cv2.imwrite(self.output_file, warped_colombus_b6)
        print("shape of the colombus_b4 after warping: ", warped_colombus_b6.shape)
        print("colombus B6 band registered")
        # cv2.imwrite(os.path.join(self.output_path, 'registered_non_b6.tif'), warped_colombus_b6)
        self.create_bounding_box(warped_colombus_b6)



    def create_bounding_box(self, colombus_image):
        sentinel_ds = gdal.Open(self.sentinel_path_dict.get('S2AB_B4'))
        sentinel_transform = sentinel_ds.GetGeoTransform()
        sentinel_projection = sentinel_ds.GetProjection()

        # Sentinel image bounding box
        sentinel_bbox = [
            (sentinel_transform[0], sentinel_transform[3]),  # Top-left
            (sentinel_transform[0] + sentinel_transform[1] * sentinel_ds.RasterXSize, sentinel_transform[3]),  # Top-right
            (sentinel_transform[0] + sentinel_transform[1] * sentinel_ds.RasterXSize, sentinel_transform[3] + sentinel_transform[5] * sentinel_ds.RasterYSize),  # Bottom-right
            (sentinel_transform[0], sentinel_transform[3] + sentinel_transform[5] * sentinel_ds.RasterYSize)  # Bottom-left
        ]

        # Get the bounding box coordinates in pixel space
        min_x = min([coord[0] for coord in sentinel_bbox])
        max_x = max([coord[0] for coord in sentinel_bbox])
        min_y = min([coord[1] for coord in sentinel_bbox])
        max_y = max([coord[1] for coord in sentinel_bbox])

        # Set the GSD for the colombus image
        target_gsd = 6.5

        # Calculate the new geotransform for the colombus image
        new_transform = (
            min_x,  # Top left x
            target_gsd,  # W-E pixel resolution
            0,  # Rotation (0 if North is up)
            max_y,  # Top left y
            0,  # Rotation (0 if North is up)
            -target_gsd  # N-S pixel resolution (negative value)
        )

        # Create the output dataset
        driver = gdal.GetDriverByName('GTiff')
        out_dataset = driver.Create(
            self.output_file,
            int((max_x - min_x) / target_gsd),
            int((max_y - min_y) / target_gsd),
            1,  # Number of bands
            gdal.GDT_Byte
        )

        out_dataset.SetGeoTransform(new_transform)
        out_dataset.SetProjection(sentinel_projection)
        out_dataset.GetRasterBand(1).WriteArray(colombus_image)
        out_dataset.FlushCache()
        out_dataset = None
        print("colombus image placed into bounding box and georeferenced")

class reprojection():
    def __init__(self):
        self.warped_dir = geoReferencing().output_file
        print("reporjection initialized")
        print("warped_dir: ", self.warped_dir)
        self.georef_save_dir = os.path.abspath(os.path.join(geoReferencing().output_path,"georef_colombus_v4.tif"))
        
        self.folder_sentinel = globals()['save_path'] + "/05_sentinel"
        self.sentinel_path_dict = {}
        for filename in os.listdir(self.folder_sentinel):
            if filename.endswith('.tif'):
                band_name = "S2AB_"+filename.split('.')[1]
                sentinel_path = os.path.join(self.folder_sentinel, filename)
                self.sentinel_path_dict[band_name] = sentinel_path
        pass



    def GetGeoInfo(self,FileName):
        SourceDS = gdal.Open(FileName)
        GeoT = SourceDS.GetGeoTransform()
        Projection = osr.SpatialReference()
        Projection.ImportFromWkt(SourceDS.GetProjectionRef())
        return GeoT, Projection

    def CreateGeoTiff(self,Name, Array, driver, 
                    xsize, ysize, GeoT, Projection):
        DataType = gdal.GDT_Float32
        NewFileName = Name
        # Set up the dataset
        DataSet = driver.Create( NewFileName, xsize, ysize, 1, DataType )
                # the '1' is for band 1.
        DataSet.SetGeoTransform(GeoT)
        DataSet.SetProjection( Projection.ExportToWkt() )
        # Write the array
        DataSet.GetRasterBand(1).WriteArray( Array )
        return NewFileName

    def ReprojectCoords(self,x, y,src_srs,tgt_srs):
        trans_coords=[]
        transform = osr.CoordinateTransformation( src_srs, tgt_srs)
        x,y,z = transform.TransformPoint(x, y)
        return x, y
    


    def projection(self):
        # Load the raw image
        raw = cv2.imread(self.warped_dir, cv2.IMREAD_UNCHANGED)
        raw = np.array(raw, dtype=np.uint16)  # Ensure the data type is 16-bit to handle 12-bit data correctly
        
        # Get raw image dimensions
        y_size, x_size = raw.shape[:2]

        # Path to the Sentinel image
        RASTER_FN = self.sentinel_path_dict.get('S2AB_B4')

        # Open the Sentinel image using GDAL
        sentinel = gdal.Open(RASTER_FN)
        if not sentinel:
            raise FileNotFoundError(f"Could not open file: {RASTER_FN}")

        # Get the geotransformation information
        ulx, xres, xskew, uly, yskew, yres = sentinel.GetGeoTransform()

        # Calculate the new geotransformation for the raw image
        new_xres = 6.5
        new_yres = -6.5
        new_ulx = ulx
        new_uly = uly
        new_geo_transform = [new_ulx, new_xres, 0, new_uly, 0, new_yres]

        # Create a new dataset to save the georeferenced image
        driver = gdal.GetDriverByName('GTiff')
        output_file = self.georef_save_dir
        dst_ds = driver.Create(output_file, x_size, y_size, 1, gdal.GDT_UInt16)

        if dst_ds is None:
            raise Exception(f"Could not create file: {output_file}")

        # Write the raw image data to the new dataset
        dst_ds.GetRasterBand(1).WriteArray(raw)

        # Set the geotransformation and projection
        dst_ds.SetGeoTransform(new_geo_transform)
        dst_ds.SetProjection(sentinel.GetProjection())

        # Close the dataset
        dst_ds = None

        return output_file


    ################## Main function  #####################


def run_georef(save_path,coreg_rgb_img):
    # Initialize the Earth Engine module.
    try:
        ee.Initialize()
    except Exception as e:
        ee.Authenticate()
        ee.Initialize()
    ################# defining roi ##########################

    # lat_input, lon_input = input("write latitude and longitude for roi: ").split(',')
    
    lat_input, lon_input  = 39.925188042843686, 32.83705246084268
    lat = float(lat_input)
    lon = float(lon_input)

    width_input = 30
    height_input = 20

    # width_input = input("write a width (km) for roi: ")
    width = float(width_input)

    # height_input = input("write a height (km) for roi: ")
    height = float(height_input)


    globals().update(locals())
    print("Globals are updated...")

    georef = geoReferencing()

    georef.band_registration()
    # repro = reprojection()
    # repro.projection()

    # for item in georef.colombus_band_dict.items():
    #     print(item)

 



if __name__ == "__main__":
    run_georef()