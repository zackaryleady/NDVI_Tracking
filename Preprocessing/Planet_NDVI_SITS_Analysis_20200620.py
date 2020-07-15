# Required imported python libraries
# Python standard libraries, no need to install
import os
import sys
import logging
import time
import datetime
import argparse
# Data Manipulation libraries
import rasterio
import numpy as np
from sklearn.preprocessing import RobustScaler
# Data Visualization libraries



def CreateLogger(log_file):
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

def CreateImgPathList(img_folder):
    img_path_lst = [os.path.join(img_folder, f) 
                    for f in os.listdir(img_folder)
                    if f.endswith(".tif")]
    return img_path_lst



def gen_NDVI_with_write(image_path, output_path):
    with rasterio.open(image_path) as ds:
        red_band3 = ds.read(3)
        nir_band4 = ds.read(4)
        np.seterr(divide='ignore', invalid='ignore')
        ndvi_array = (((nir_band4.astype(float) - red_band3.astype(float))
                     / (nir_band4.astype(float) + red_band3.astype(float))))
    kwargs = ds.meta
    kwargs.update(dtype=rasterio.float32, count=1)       
    with rasterio.open(output_path, 'w', **kwargs) as dst:
        dst.write_band(1, ndvi_array.astype(rasterio.float32))
    return ndvi_array


def MainPrepare(ini_dict):

    return

def MainAnalyze(ini_dict):
    input_dir = ini_dict.get("planet_image_input")
    date_keys = [x for x in os.listdir(input_dir)]
    all_raster_dict = {k: '' for k in date_keys}
    for root, dirs, files in os.walk(os.path.normpath(input_dir), topdown=False):
        for file_name in files:
            if ('AnalyticMS_clip' in file_name or 'mosaic' in file_name) and (file_name.endswith('.bsq') or file_name.endswith('.tif')):
                date_id = os.path.basename(root)
                all_raster_dict[date_id] = os.path.join(root, file_name)

    output_ndvi_dir = os.path.join(os.path.dirname(input_dir), 'poh_final_NDVI')
    if not os.path.exists(output_ndvi_dir):
        os.mkdir(output_ndvi_dir)

    ndvi_dict = {}
    for k_date in sorted(list(all_raster_dict.keys())):
        ndvi_output_filepath = os.path.join(output_ndvi_dir, "{}_ndvi.tif".format(k_date))
        ndvi_arr = gen_NDVI_with_write(all_raster_dict.get(k_date), ndvi_output_filepath)
        ndvi_dict[k_date] = ndvi_arr
    
    

    return


if __name__ == "__main__":
    # begin runtime clock
    start = datetime.datetime.now()
    # initialize the command line parser object from argparse
    parser = argparse.ArgumentParser()
    # set the command line arguments available to user's
    parser.add_argument("--planet_image_input", "-pii", type=str,
                        help="Provide the absolute folder name containing PS\
                        images")
    parser.add_argument("--gen_ndwi", "-ndwi", type=bool,
                        help="True or False of generating NDWI raster")
    parser.add_argument("--gen_ndvi", "-ndvi", type=bool,
                        help="True or False for generating NDVI raster")
    args = parser.parse_args()
    ini_dict = vars(args)
    # determine the absolute file pathname of this *.py file
    abspath = os.path.abspath(__file__)
    # from the absolute file pathname determined above,
    # extract the directory path
    dir_name = os.path.dirname(abspath)
    # creates the log file pathname which is an input to CreateLogger
    log_name = os.path.join(dir_name, 'planet_ndvi_sits_{}.log'
                            .format(datetime.datetime.now().date()))
    # generic CreateLogger function which creates two loggers
    # one for the logfile write out and one for the on-screen stream write out
    logger = CreateLogger(log_name)
    logging.info("Absolute python file location: \n {}".format(abspath))
    logging.info("Absolute dir_name location: \n {}".format(dir_name))
    MainPrepare(ini_dict)
    MainAnalyze(ini_dict)
    # end the code's clock and reports runtime
    elapsed_time = datetime.datetime.now() - start
    logging.info("Runtime: {} seconds".format(elapsed_time))
