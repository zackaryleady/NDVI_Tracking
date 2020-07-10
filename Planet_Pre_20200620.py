import os
import sys
import shutil
import logging
import argparse
import datetime
import json
from xml.dom import minidom
import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.merge import merge
from rasterio.features import dataset_features
import shapely
from shapely.geometry import shape, GeometryCollection
import arosics
from arosics import COREG_LOCAL


def create_logger(log_file):
    """ Zack's Generic Logger function to create onscreen and file logger

    Parameters
    ----------
    log_file: string
        `log_file` is the string of the absolute filepathname for writing the
        log file too which is a mirror of the onscreen display.

    Returns
    -------
    logger: logging object

    Notes
    -----
    This function is completely generic and can be used in any python code.
    The handler.setLevel can be adjusted from logging.INFO to any of the other
    options such as DEBUG, ERROR, WARNING in order to restrict what is logged.

    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # create console handler and set level to info
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    # create error file handler and set level to info
    handler = logging.FileHandler(log_file,  "w", encoding=None, delay="true")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def single_folder_2_planet_order_hierarchy(input_dir, target_dir):
    """ Reorganizes a folder with multiple planet images and metadata

    This function takes in a single input folder containing
    a mixture of planet images and metadata files and reorganizes the data into
    the following format target_dir --> each day has a date folder --> 
    in each day folder are individual folders for each image -->
    in each image folder is the image file and metadata. This is similiar to how
    Planet Explorer delivers orders via their web interface.

    Parameters
    ----------
    input_dir: str
        `input_dir` is the absolute or relative folder path containing the mixed planet data files
    
    target_dir: str
        `target_dir` is the absolute or relative folder path of where to build the
        planet order hierarchy folder structure

    Returns
    -------
    This function simply returns 0 if it completes for error checking purposes.

    Notes
    -----
    This is usually run after the merge_multi_folder_2_single_folder function.

    """
    for root, dirs, files in os.walk(os.path.normpath(input_dir), topdown=False):
        for file_name in files:
            # This selects only the actual image *.tif file
            if "AnalyticMS_clip" in file_name and file_name.endswith(".tif"):
                logging.info("Found: {}".format(file_name))
                file_name_split = file_name.split("_")
                # The date_id in the ISO format YYYYMMDD becomes a folder
                date_id = file_name_split[0]
                date_dir = os.path.join(target_dir, date_id)
                logging.info("Writing date directory as: {}".format(date_dir))
                if not os.path.isdir(date_dir):
                    os.mkdir(date_dir)
                # The img_id is formatted as YYYYMMDD_id_sat
                img_id = "_".join(file_name_split[0:3])
                logging.info("Writing image folder as: {}".format(img_id))
                img_dir = os.path.join(date_dir, img_id)
                if not os.path.isdir(img_dir):
                    os.mkdir(img_dir)
                logging.info("Writing to image directory: {}".format(img_dir))
                # loop for copying all metadata files as well with the same img_id
                for file_name2 in files:
                    if img_id in file_name2:
                        source_dir = os.path.join(input_dir, file_name2)
                        shutil.copy2(source_dir, img_dir) 
    return 0


def merge_multi_folder_2_single_folder(input_dir, merged_dir):
    """ Merges all files in an input_dir into a single folder

    This is a simple utility for getting all planet data files into a single folder from
    a possibly tiered directory tree / folder system.

    Parameters
    ----------
    input_dir: str
        `input_dir` is the folder pathname or directory tree that needs to be merged
    
    merged_dir: str
        `merged_dir` is the folder pathname that all the files from `input_dir` should be copied too
    
    Returns
    -------
    This function simply returns 0 if it completes for error checking purposes.

    Notes
    -----
    This is usually run before the single_folder_2_planet_order_hierarchy function.

    """
    for root, dirs, files in os.walk((os.path.normpath(input_dir)), topdown=False):
        for file_name in files:
            logging.info("Found: {}".format(file_name))
            source_dir = os.path.join(root, file_name)
            shutil.copy2(source_dir, merged_dir)
            logging.info("Wrote: {} to merged directory".format(file_name))
    return 0


def planet_filter_cloud_from_json(json_file):
    """ Helper function for planet data cloud filtering from JSON

    Parameters
    ----------
    json_file: str
        `json_file` is the absolute or relative path to the *.json metadata file
        associated with the planet image

    Returns
    -------
    cloud_value: float
        `cloud_value` is a percent value 0-100 of the cloud coverage in the planet image

    Notes
    -----
    The cloud value is a percent 0-100 that is returned as a float.
    This function is a helper function to planet_filter_cloud

    """
    with open(json_file) as f:
        json_obj = json.load(f)
        # The json_obj is read similiarly as a python dictionary
        # aka json_obj.get("properties").get("cloud_cover")
        cloud_value = json_obj["properties"]["cloud_cover"]
    f.close()
    return float(cloud_value)

def planet_filter_cloud_from_xml(xml_file):
    """ Helper function for planet data cloud filtering from XML

    Parameters
    ----------
    xml_file: str
        `xml_file` is the absolute or relative path to the *.xml metadata file
        associated with the planet image

    Returns
    -------
    cloud_value: float
        `cloud_value` is a percent value 0-100 of the cloud coverage in the planet image

    Notes
    -----
    The cloud value is a percent 0-100 that is returned as a float.
    This function is a helper function to planet_filter_cloud

    """
    xmldoc = minidom.parse(xml_file)
    cloud_value = xmldoc.getElementsByTagName("opt:cloudCoverPercentage")[0].firstChild.data
    return float(cloud_value)


def planet_filter_cloud(input_dir, target_dir, file_type='json', cloud_threshold=0.30):
    """ The main cloud filtering function for planet imagery data

    It is possible to set a cloud filter in the porder tool used to download planet imagery.
    However, it is recommended that the user use this post-download filtering function for
    greater flexibility in determining what imagery to use. This functions reads the
    planet determined cloud coverage from each image's metadata file (json or xml) and then
    only selects the images with less than or equal to the cloud_threshold. 

    Parameters
    ----------
    input_dir: str
        `input_dir` is the absolute or relative folder pathname for the directory
        to be cloud filtered based on planet imagery metadata
    
    file_type: str
        `file_type` is a keyword argument for which type of metadata file to look for
        the cloud coverage value for each planet image. Valid inputs are 'json' or 'xml'
        as these are the two types of metadata files available with planet imagery.
        The default is set to 'json'
    
    cloud_threshold: float
        `cloud_threshold` is a keyword argument for the percentage of cloud cover allowed
        in a planet image. Valid inputs are between 0-1 (i.e. 0.30 == 30%).
        The default is set to 0.30 or 30%
    
    Returns
    -------
    This function simply returns 0 if it completes for error checking purposes.

    Notes
    -----
    planet_filter_cloud_from_json is a helper function
    planet_filter_cloud_from_xml is a helper function

    """
    # if/else clause sets the file extension to search for and the helper function to use
    if file_type == 'json':
        file_ext = '.json'
        filter_func = planet_filter_cloud_from_json
    elif file_type == 'xml':
        file_ext = '.xml'
        filter_func = planet_filter_cloud_from_xml
    else:
        logging.error("The keyword argument file_type for planet_filter_cloud must be 'json' or 'xml' ")
    exception_roots = []
    num_imgs_below_cloud_threshold = 0
    for root, dirs, files in os.walk((os.path.normpath(input_dir)), topdown=False):
        for file_name in files:
                if file_name.endswith(file_ext) and 'manifest' not in file_name:
                    metadata_file = os.path.join(root, file_name)
                    cloud_percent = filter_func(metadata_file)
                    # Apply cloud threshold as filter on the metadata cloud percent
                    if cloud_percent <= cloud_threshold:
                        num_imgs_below_cloud_threshold += 1
                        file_name_split = file_name.split("_")
                        date_id = file_name_split[0]
                        date_dir = os.path.join(target_dir, date_id)
                        if not os.path.isdir(date_dir):
                            os.mkdir(date_dir)
                        img_id = '_'.join(file_name_split[0:3])
                        img_dir = os.path.join(date_dir, img_id)
                        try:
                            shutil.copytree(root, img_dir)
                        except FileExistsError:
                            exception_roots.append([root, img_dir])
                            pass
    for e in exception_roots:
        logging.info("Exception for directory: \n {}".format(e))
    logging.info("Total number of images that passed the cloud threshold: {}".format(num_imgs_below_cloud_threshold))
    return 0


def read_aoi_polygon(poly_file):
    """ Reads the Area of Interest Polygon

    Reads the Area of Interest Polygon from a *.geojson or *.json file into a Shapely geometry collection object.

    Parameters
    ----------
    poly_file: str
        `poly_file` is the absolute or relative file pathname for the area of interest polygon file.
        (i.e. the polygon geojson or json file you used to download planet imagery...porder)
        Valid file extensions are *.geojson or *.json only.

    Returns
    -------
    aoi_polygon: obj
        `aoi_polygon` is a GeometryCollection object from the Python Shapely library.
        It contains the coordinates of the AOI Polygon for calculating intersection and area.

    Notes
    -----
    Helper function to planet_filter_coverage_mosaic.
    This function is generic, it can be used to read any *.json or *.geojson polygon into a Shapely GeometryCollection.

    """
    # Error check the file is a *.json or an *.geojson file
    if not poly_file.endswith(".json") and not poly_file.endswith(".geojson"):
        logging.error("The Area of Interest Polygon is not a *.json or *.geojson file. It must be one or the other.")
    with open(poly_file) as f:
        features = json.load(f)["features"]
    # .buffer[0] deals with repeated coordinate tracing
    aoi_polygon = GeometryCollection([shape(feature["geometry"]).buffer(0) for feature in features])
    return aoi_polygon


def find_all_rasters_by_date(input_dir):
    date_keys = [x for x in os.listdir(input_dir)]
    all_raster_dict = {k: {} for k in date_keys}
    for root, dirs, files in os.walk(os.path.normpath(input_dir), topdown=False):
        for file_name in files:
            if 'AnalyticMS_clip' in file_name and file_name.endswith(".tif"):
                date_id = os.path.basename(os.path.dirname(root))
                inner_dict = all_raster_dict.get(date_id)
                inner_dict[os.path.basename(root)] = os.path.join(root, file_name)
    return all_raster_dict


def multi_raster_open_helper(r_dict):
    raster_fp_dict = {}
    for k in r_dict:
        temp_fp = open_raster(r_dict.get(k))
        raster_fp_dict[k] = temp_fp
    return raster_fp_dict


def open_raster(raster_path):
    raster_f = rasterio.open(raster_path)
    logging.info("Raster Image {} has {} indices"
                 .format(raster_path, raster_f.indexes))
    return raster_f


def mosaic_rasters(raster_fp_dict, aoi_polygon, out_fp):
    
    m_fp = [raster_fp_dict.get(k) for k in raster_fp_dict.keys()]
    try:
        m_raster, out_trans = merge(m_fp)
        out_meta = m_fp[0].meta.copy()
        out_meta.update(dtype=rasterio.uint16, count=4,
                        height=m_raster.shape[1], width=m_raster.shape[2],
                        transform=out_trans, driver="GTiff")
        with rasterio.open(out_fp, "w", **out_meta) as dest:
            dest.write(m_raster)
    except RasterioIOError as e:
        logging.error(e)
        return 1
    return 0


def planet_filter_coverage_mosaic(input_dir, mosaic_filtered_path, aoi_poly_file):
    all_raster_dict = find_all_rasters_by_date(input_dir)
    aoi_poly = read_aoi_polygon(aoi_poly_file)
    aoi_poly_area = aoi_poly.area
    logging.info("AOI Polygon Area: {}".format(aoi_poly_area))
    filtered_dates = []
    more_than_1_gc = {}
    sorted_raster_date_keys = sorted(list(all_raster_dict.keys()))
    logging.info("Found {} planet image day date keys in all_raster_dict".format(len(sorted_raster_date_keys)))
    for date_key in sorted_raster_date_keys:
        raster_fp_dict = multi_raster_open_helper(all_raster_dict.get(date_key))
        logging.info("Raster File Opener Keys: {}".format(raster_fp_dict.keys()))
        out_fp = os.path.join(input_dir, date_key, "{}_fmosaic.tiff".format(date_key))
        int_e = mosaic_rasters(raster_fp_dict, aoi_poly, out_fp)
        if int_e == 0:
            mosaic_test_fp = open_raster(out_fp)
            fmosaic_data = dataset_features(mosaic_test_fp, bidx=1, band=False, as_mask=True)
            gc_fmosaic = GeometryCollection([shape(feature["geometry"]).buffer(0) for feature in fmosaic_data])
        # There should only be one geometry feature in the Geometry Collection
        # except in bad / incomplete mosaics
            if len(gc_fmosaic) > 1:
                more_than_1_gc[date_key] = len(gc_fmosaic)
            fmosaic_inter = gc_fmosaic[0].intersection(aoi_poly)
            fmosaic_inter_area = fmosaic_inter.area 
            if not fmosaic_inter_area == aoi_poly_area:
                logging.info("Area is less than full raster coverage of polygon aoi: {} != {} for {}"
                             .format(fmosaic_inter_area, aoi_poly_area, date_key))
                filtered_dates.append(date_key)
        elif int_e == 1:
            filtered_dates.append(date_key)

    logging.info(filtered_dates)
    logging.info(len(filtered_dates))
    logging.info(len(all_raster_dict.keys()))
    logging.info(more_than_1_gc)
    if not os.path.exists(mosaic_filtered_path):
        os.mkdir(mosaic_filtered_path)
    for date_key in sorted_raster_date_keys:
        if not date_key in filtered_dates:
            poh_cloud_directory = os.path.join(input_dir, date_key)
            poh_mosaic_directory = os.path.join(mosaic_filtered_path, date_key)
            shutil.copytree(poh_cloud_directory, poh_mosaic_directory)
    return 0

def parse_xml_band_coeffs(xml_file, bands=['1', '2', '3', '4']):
        coeffs = {}
        xmldoc = minidom.parse(xml_file)
        nodes = xmldoc.getElementsByTagName("ps:bandSpecificMetadata")
        for node in nodes:
                band_node = node.getElementsByTagName("ps:bandNumber")[0].firstChild.data
                if band_node in bands:
                        i_band = int(band_node)
                        value = node.getElementsByTagName("ps:reflectanceCoefficient")[0].firstChild.data
                        coeffs[i_band] = float(value)
        return coeffs

def radiance_2_reflectance_conversion(radiance_raster_file, coeffs, out_fp):
        logging.info("Conversion coefficients: {}".format(coeffs))
        standard_scale = 10000
        logging.info("Standard scale is {}".format(standard_scale))
        with rasterio.open(radiance_raster_file) as src:
                blue_radiance = src.read(1)
                green_radiance = src.read(2)
                red_radiance = src.read(3)
                nir_radiance = src.read(4)
        radiance_lst = [blue_radiance, green_radiance, red_radiance, nir_radiance]
        scaled_reflectance_lst = []
        for indx, rad in enumerate(radiance_lst):
                band_num = int(indx + 1)
                reflectance_temp = rad * coeffs.get(band_num)
                scaled_temp = reflectance_temp * standard_scale
                scaled_reflectance_lst.append(scaled_temp)
        for r, sf, b in zip(radiance_lst, scaled_reflectance_lst, ['Blue', 'Green', 'Red', 'NIR']):
                logging.info("{} band radiance is from {} to {}".format(b, np.amin(r), np.amax(r)))
                logging.info("{} band scaled reflectance is from {} to {}".format(b, np.amin(sf), np.amax(sf)))
        kwargs = src.meta
        kwargs.update(dtype=rasterio.uint16, count=4)        
        with rasterio.open(out_fp, 'w', **kwargs) as dst:
                for indx, sr in enumerate(scaled_reflectance_lst):
                        band_num = int(indx + 1)
                        dst.write_band(band_num, sr.astype(rasterio.uint16))
        logging.info("Wrote {}".format(out_fp))
        return 0


def planet_radiance_2_reflectance(input_dir, output_dir):
    all_raster_dict = find_all_rasters_by_date(input_dir)
    errors = []
    for date_key in sorted(list(all_raster_dict.keys())):
        r_dict = all_raster_dict.get(date_key)
        for k in r_dict.keys():
            radiance_raster_file = r_dict.get(k)
            temp_dir = os.path.dirname(radiance_raster_file)
            temp_xml_name = '_'.join(os.path.basename(radiance_raster_file).split("_")[0:-1])
            out_fp_date_directory = os.path.join(output_dir, date_key, "{}".format('_'.join(os.path.basename(radiance_raster_file).split("_")[0:3])))
            if not os.path.isdir(out_fp_date_directory):
                os.makedirs(out_fp_date_directory)
            out_fp = os.path.join(out_fp_date_directory, "{}_refl.tif".format(os.path.basename(radiance_raster_file).split(".")[0]))
            k_xml_file = os.path.join(temp_dir, "{}_metadata_clip.xml".format(temp_xml_name))
            try:
                coeffs_dict = parse_xml_band_coeffs(k_xml_file)
                radiance_2_reflectance_conversion(radiance_raster_file, coeffs_dict, out_fp)
                copy_xml_path = os.path.join(out_fp_date_directory, "{}_metadata_clip.xml".format(temp_xml_name))
                shutil.copy2(k_xml_file, copy_xml_path)
            except FileNotFoundError:
                errors.append([date_key, radiance_raster_file])
                pass
    logging.error(errors)
    return 0


def parse_xml_offnadir_angle(xml_file):
        xmldoc = minidom.parse(xml_file)
        angle_deg = xmldoc.getElementsByTagName("ps:spaceCraftViewAngle")[0].firstChild.data
        return float(angle_deg)


def determine_target_img_coreg(input_dir):
    all_raster_dict = find_all_rasters_by_date(input_dir)
    one_img_dict = {}
    lowest_offnadir_angle = {}
    for date_key in sorted(list(all_raster_dict.keys())):
        temp_dict = all_raster_dict.get(date_key)
        if len(list(temp_dict.keys())) == 1:
            one_img_dict[date_key] = temp_dict
            xml_basename = os.path.basename(list(temp_dict.values())[0])
            xml_basename_split = '_'.join(xml_basename.split("_")[0:-2])
            xml_basename_parsed = xml_basename_split + '_metadata_clip.xml'
            full_xml_path = os.path.join(os.path.dirname(list(temp_dict.values())[0]), xml_basename_parsed)
            cloud_value = planet_filter_cloud_from_xml(full_xml_path)
            if cloud_value <= 0.05:
                offnadir_angle = parse_xml_offnadir_angle(full_xml_path)
                lowest_offnadir_angle[abs(offnadir_angle)] = date_key
    lna = sorted(list(lowest_offnadir_angle.keys()))
    target_img_angle = lna[0]
    target_img_date = lowest_offnadir_angle.get(lna[0])
    logging.info("Lowest off-nadir angle: {} {}".format(target_img_angle, target_img_date))
    return target_img_date

def arosics_coreg_single_planet_image(ref_path, img_path, project_dir):
    kwargs = {'grid_res': 200, 'window_size': (64, 64), 'path_out': 'auto', 'projectDir': project_dir}
    CRL = COREG_LOCAL(ref_path, img_path, **kwargs)
    CRL.correct_shifts()
    return 0

def arosics_coreg_multi_planet_helper(input_dir, out_dir, target_img_date):
    all_raster_dict = find_all_rasters_by_date(input_dir)
    ref_img_for_coreg = list(all_raster_dict.get(target_img_date).values())[0]
    print(ref_img_for_coreg)
    input("?")
    for date_key in sorted(list(all_raster_dict.keys())):
        if not date_key == target_img_date:
            temp_dict = all_raster_dict.get(date_key)
            for k in temp_dict.keys():
                project_dir = os.path.join(out_dir, date_key)
                if not os.path.isdir(project_dir):
                    os.mkdir(project_dir)
                try:
                    arosics_coreg_single_planet_image(ref_img_for_coreg, temp_dict.get(k), project_dir)
                except Exception as e:
                    logging.error(e)
    return 0


def final_mosaic(input_dir, output_dir, aoi_polygon_filepath):
    aoi_polygon = read_aoi_polygon(aoi_polygon_filepath)
    date_keys = [x for x in os.listdir(input_dir)]
    all_raster_dict = {k: {} for k in date_keys}
    for root, dirs, files in os.walk(os.path.normpath(input_dir), topdown=False):
        for file_name in files:
            if 'AnalyticMS_clip' in file_name and file_name.endswith(".bsq"):
                date_id = os.path.basename(root)
                print(date_id)
                inner_dict = all_raster_dict.get(date_id)
                print(type(inner_dict))
                print(file_name)
                file_name_parsed = '_'.join(file_name.split("_")[0:3])
                print(file_name_parsed)
                inner_dict[file_name_parsed] = os.path.join(root, file_name)
    for date_key in sorted(list(all_raster_dict.keys())):
        date_inner_dict = all_raster_dict.get(date_key)
        if len(list(date_inner_dict.keys())) > 1:
            logging.info("Final Mosaic of {} Processing".format(date_key))
            raster_file_open_dict = multi_raster_open_helper(date_inner_dict)
            output_file_dir = os.path.join(output_dir, date_key)
            if not os.path.isdir(output_file_dir):
                os.mkdir(output_file_dir)
            output_filepath = os.path.join(output_file_dir, "{}_final_mosaic.tif".format(date_key))
            mosaic_rasters(raster_file_open_dict, aoi_polygon, output_filepath)
        else:
            logging.info("No Mosaic Needed for {}, Copying".format(date_key))
            try:
                existing_filepath = list(date_inner_dict.values())[0]
                existing_dir = os.path.dirname(existing_filepath)
                output_file_dir = os.path.join(output_dir, date_key)
                shutil.copytree(existing_dir, output_file_dir)
            except IndexError:
                logging.error("Directory {} empty, skipping".format(date_key))
                pass

    return 0


def main_exe(ini_dict):
    data_dir = ini_dict.get("planet_data_folder")
    aoi_polygon_filepath = ini_dict.get("planet_aoi_polygon_geojson")
    # Create Directories
    merged_dir = os.path.join(os.path.dirname(data_dir), 'merged_planet_data')
    poh_dir = os.path.join(os.path.dirname(data_dir), 'poh_merged')
    poh_cloud = os.path.join(os.path.dirname(data_dir), "poh_cloudfiltered")    
    poh_mosaic = os.path.join(os.path.dirname(data_dir), "poh_mosaicfiltered")
    poh_refl = os.path.join(os.path.dirname(data_dir), 'poh_reflectance')
    coreg_dir = os.path.join(os.path.dirname(data_dir), 'poh_coreg')
    fcm_dir = os.path.join(os.path.dirname(data_dir), 'poh_final')
    for d in [merged_dir, poh_dir, poh_cloud, poh_mosaic, poh_refl, coreg_dir, fcm_dir]:
        if not os.path.isdir(d):
            os.mkdir(d)
    merge_multi_folder_2_single_folder(data_dir, merged_dir)    
    single_folder_2_planet_order_hierarchy(merged_dir, poh_dir)    
    planet_filter_cloud(poh_dir, poh_cloud, file_type='json', cloud_threshold=0.90)    
    planet_filter_coverage_mosaic(poh_cloud, poh_mosaic, aoi_polygon_filepath)    
    planet_radiance_2_reflectance(poh_mosaic, poh_refl)
    target_img_date = determine_target_img_coreg(poh_refl)    
    arosics_coreg_multi_planet_helper(poh_refl, coreg_dir, target_img_date)
    final_mosaic(coreg_dir, fcm_dir, aoi_polygon_filepath)
    return 0

if __name__ == "__main__":
    # begin runtime clock
    start = datetime.datetime.now()
    # determine the absolute file pathname of this *.py file
    abspath = os.path.abspath(__file__)
    # from the absolute file pathname determined above,
    # extract the directory path
    dir_name = os.path.dirname(abspath)
    # initiate logger
    log_file = os.path.join(dir_name, 'planet_pre_proc_{}.log'
                            .format(start.date()))
    create_logger(log_file)
    # create the command line parser object from argparse
    parser = argparse.ArgumentParser()
    # set the command line arguments available to user's
    parser.add_argument("--planet_data_folder", "-pdf", type=str,
                        help="Provide the absolute folder path containing PS image data")
    parser.add_argument("--planet_aoi_polygon_geojson", "-aoi", type=str,
                        help="Provide the absolute file path for the AOI polygon in geojson")
    # create an object of the command line inputs
    args = parser.parse_args()
    # read the command line inputs into a Python dictionary
    ini_dict = vars(args)
    main_exe(ini_dict)
    elapsed_time = datetime.datetime.now() - start
    logging.info("Runtime: {}".format(elapsed_time))  