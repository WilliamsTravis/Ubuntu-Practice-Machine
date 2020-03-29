# -*- coding: utf-8 -*-
"""
    Updating WWDT Data on a monthly basis.

    Run this using crontab once a month to pull netcdf files from the
    WestWide Drought Tracker site, transform them to fit in the app, and either
    append them to an existing file, or build the data set from scratch. This
    also rebuilds each percentile netcdf entirely because those are rank based.

    Production notes:
        - The geographic coordinate systems work for the most part. I had to
          use 'degrees_south' for the latitude unit attribute to avoid flipping
          the image. Also, the netcdfs built using my functions appear to have
          coordinates at the grid center, which is different than previous
          geotiffs I created, however, the maps are rendered properly (point in
          lower left corner), and I believe that's because of the binning step.
        - There appears to be a directory bug when starting over, but only for
          wwdt on the virtual machines...this may have been fixed, need to
          check.
        - There are a few ways to make this a little more efficient but I left
          a few redundancies in just to make sure everything is kept in the
          right order, chronologically.
        - Also, if a download fails I'm leaving it be. It should correct itself
          in the next month. This could be improved by redownloading in an
          exception, but I would want advice on how best to handle the case
          where the link/server is down.
        - Also, at the moment, this script only checks the original index files
          so if there's a problem with only the percentiles it won't be
          noticed.

    crontab:
        - to setup a scheduler do this (1:30am on the 2nd of each month):
            1) enter <crontab -e>

            2) insert (but with spaces at these line breaks):
              <30 01 02 * *
               /root/Sync/Ubuntu-Practice-Machine/env/bin/python3
               /root/Sync/Ubuntu-Practice-Machine/scripts/Get_WWDT.py >>
               cronlog.log>

            3) ctrl + x
            4) This script, because it gets the bulk of the data, is also used
               to set the index ranges for all indices, so schedule it last.

Created on Fri Feb  10 14:33:38 2019

@author: user
"""

import datetime as dt
import logging
import multiprocessing as mp
import os
import sys
import urllib

from socket import timeout
from urllib.error import HTTPError, URLError
import numpy as np
import pandas as pd
import requests
import xarray as xr

from bs4 import BeautifulSoup
from netCDF4 import Dataset
from osgeo import gdal
from tqdm import tqdm

from drip.constants import PROJECT_DIR
from drip.data import Data_Path, toNetCDFPercentile
from drip.data import percentile_arrays, wgs_netcdf, albers_netcdf
from drip.gdalmethods import warp


# Get resolution from file call
if len(sys.argv) > 1:
    RES = float(sys.argv[1])
else:
    RES = 0.125


# Set the data path and also the index path to specify resolutions
DP = Data_Path(PROJECT_DIR, "data")
IP = Data_Path(DP.data_path, "droughtindices", str(RES).replace(".", "_"))


# Other constants
TODAYS_DATE = dt.datetime.today()
TODAY_STRING = np.datetime64(TODAYS_DATE)
WWDT_URL = 'https://wrcc.dri.edu/wwdt/data/PRISM'
LOCAL_PATH = IP.join("netcdfs/wwdt")
INDICES = ['spi1', 'spi2', 'spi3', 'spi4', 'spi5', 'spi6', 'spi7', 'spi8',
           'spi9', 'spi10', 'spi11', 'spi12', 'spei1', 'spei2', 'spei3',
           'spei4', 'spei5', 'spei6', 'spei7', 'spei8', 'spei9', 'spei10',
           'spei11', 'spei12', 'pdsi', 'scpdsi', 'pzi', 'mdn1']
PROJ = ('+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 +lon_0=-96 +x_0=0 +y_0=0 ' +
        '+ellps=GRS80 +datum=NAD83 +units=m no_defs')


def create_data(index):
    """Download monthly WWDT data and combine it in a single local file."""

    # Get the data from wwdt - no open dap :/
    path_dict = remote_local_paths(index)
    download(path_dict)

    # We need to reoder the dates, they have a funny system
    print("Merging and ordering netcdf datasets...")
    local_paths = path_dict.values()
    temp_file = IP.join("netcdfs", "wwdt", "temp.nc")
    if os.path.exists(temp_file):
        os.remove(temp_file)
    ds = xr.open_mfdataset(local_paths)
    ds = ds.sortby("day")
    ds.to_netcdf(temp_file)
    ds.close()

    # Warp WGS 84
    print("Resampling arrays...")
    src = temp_file
    dst_wgs = IP.join("netcdfs", "wwdt", "temp.tif")
    warp(src, dst_wgs, dtype="Float32", progress=False, overwrite=True,
         dstSRS='EPSG:4326', xRes=RES, yRes=RES, format="GTiff",
         dstNodata=-9999., outputBounds=[-130, 20, -55, 50])

    # Read it back in and set the attributes to save as a proper netcdf
    dst = IP.join("netcdfs", index + ".nc")
    ds = gdal.Open(dst_wgs)
    arrays = ds.ReadAsArray()
    proj84 = ds.GetProjection()
    wgs_netcdf(arrays, dst, proj84, template=temp_file)

    # Now let's create the percentile set
    arrays[arrays == -9999.] = np.nan
    parrays = percentile_arrays(arrays)
    parrays[np.isnan(parrays)] = -9999.
    parrays = parrays.astype("float32")
    dst = IP.join("netcdfs", "percentiles", index + ".nc")
    wgs_netcdf(parrays, dst, proj84, template=temp_file)
    del arrays

    # Warp Albers
    src = dst_wgs
    dst_albers = IP.join("netcdfs", "wwdt", "temp_albers.tif")
    warp(src, dst_albers, dtype="Float32", progress=False, overwrite=True,
         dstSRS=PROJ, format="GTiff")

    dst = IP.join("netcdfs", "albers", index + ".nc")
    ds = gdal.Open(dst_albers)
    arrays = ds.ReadAsArray()
    projal = ds.GetProjection()
    albers_netcdf(arrays, dst, projal, template=temp_file)
    del arrays


def dl(arg):
    """Download a remote file and store it locally."""

    # unpack args and check paths
    remote, local = arg
    os.makedirs(os.path.dirname(local), exist_ok=True)
    filename = os.path.basename(local)

    # Download
    try:
        urllib.request.urlretrieve(remote, local)
    except (HTTPError, URLError) as error:
        logging.error('%s not retrieved. %s\nURL: %s',
                      filename, error, remote)
    except timeout:
        logging.error('Socket timed out: %s', remote)
    else:
        logging.info('Access successful.')


def download(path_dict):
    """Download all files from a url:local_path dictionary."""

    args = list(path_dict.items())
    with mp.Pool(mp.cpu_count()) as pool:
        for _ in tqdm(pool.imap(dl, args), position=0,
                      total=len(args)):
            pass


def index_ranges():
    """Find value ranges for each file to use for color ramps later."""

    index_paths = [IP.join("netcdfs", i  + ".nc") for i in INDICES]
    maxes = {}
    mins = {}
    for i in tqdm(range(len(index_paths)), position=0):
        with xr.open_dataset(index_paths[i]) as data:
            indexlist = data
            data.close()
        mx = float(indexlist.max().value)
        mn = float(indexlist.min().value)
        maxes[INDICES[i]] = mx
        mins[INDICES[i]] = mn

    df = pd.DataFrame([maxes, mins]).T
    df.columns = ['max', 'min']
    df['index'] = df.index
    df.to_csv(DP.join("tables", "index_ranges.csv"), index=False)


def check_dates(nc_path, wwdt_index_url):
    """Check a netcdf file for missing dates."""

    # If we only need to add a few dates
    with xr.open_dataset(nc_path) as data:
        dates = pd.DatetimeIndex(data.time.data)
        data.close()

    # Extract Dates
    t1 = dates[0]
    t2 = dates[-1]
    this_year = TODAYS_DATE.year
    last_month = (TODAYS_DATE - dt.timedelta(weeks=4)).month
    if t2.year == this_year and t2.month == last_month:
        print('No missing files, moving on...\n')
        return
    else:
        # Get a list of the dates already in the netcdf file
        existing_dates = pd.date_range(t1, t2, freq='MS')

        # Get available dates from wwdt
        html = requests.get(wwdt_index_url)
        soup = BeautifulSoup(html.content, "html.parser")
        alllinks = soup.find_all('a')
        linktext = [str(l) for l in alllinks]
        netcdfs = [l.split('"')[1] for l in linktext if '.nc' in l]

        # Find the latest available monthly file
        month_ncs = [l for l in netcdfs if len(l.split('_')) == 4]
        years = [int(l.split('_')[1]) for l in month_ncs]
        final_year = max(years)
        final_months = [int(l.split('_')[2]) for l in month_ncs if
                        str(final_year) in l]
        final_month = max(final_months)

        # Now, get the range of dates
        t2 = pd.datetime(final_year, final_month, 1)
        available_dates = pd.date_range(t1, t2, freq='MS')
        needed_dates = [a for a in available_dates if
                        a not in existing_dates]

        # Just in case a date got in too early
        for d in needed_dates:
            if d > pd.Timestamp(TODAYS_DATE):
                idx = needed_dates.index(d)
                needed_dates.pop(idx)

        return needed_dates


def remote_local_paths(index):
    """Build a list of urls for a complete WWDT index record."""
    paths = {}
    for i in range(1, 13):
        file_name = '{}_{}_PRISM.nc'.format(index, i)
        target_url = WWDT_URL + '/' + index + '/' + file_name
        local_path = os.path.join(LOCAL_PATH, 'temp_{}.nc'.format(i))
        paths[target_url] = local_path

    return paths


def update_data(index):
    """If the file exists, check for missing dates and update if needed."""

    # Build this index's url
    wwdt_index_url = os.path.join(WWDT_URL, index)

    # We need the key 'value' to point to local data
    nc_path = IP.join("netcdfs", index + '.nc')
    nc_proj_path = IP.join("netcdfs", "albers", index + '.nc')

    # Check for missing dates
    needed_dates = check_dates(nc_path, wwdt_index_url)

    # Download new files
    if len(needed_dates) > 0:
        statement = '{} missing file(s) since {}...\n'
        print(statement.format(len(needed_dates), needed_dates[0]))

        for d in tqdm(needed_dates, position=0):

            # build paths
            file = '{}_{}_{}_PRISM.nc'.format(index, d.year, d.month)
            url = os.path.join(wwdt_index_url, file)
            source_path = os.path.join(LOCAL_PATH, 'temp.nc')

            # They save their dates on day 15
            date = pd.datetime(d.year, d.month, 15)

            dl((url, source_path))

            # now transform that file
            wgs_path = os.path.join(LOCAL_PATH, 'temp.tif')
            warp(source_path, wgs_path, progress=False, dtype="Float32",
                 overwrite=True,  dstSRS='EPSG:4326', xRes=RES, yRes=RES,
                 format="GTiff", outputBounds=[-130, 20, -55, 50])

            # Also create an alber's equal area projection
            source_path = wgs_path
            albers_path = os.path.join(LOCAL_PATH, 'proj_temp.tif')
            warp(source_path, albers_path, progress=False, dtype="Float32",
                 overwrite=True, dstSRS=PROJ, format="GTiff")

            # Open old data sets
            old = Dataset(nc_path, 'r+')
            old_p = Dataset(nc_proj_path, 'r+')
            times = old.variables['time']
            times_p = old_p.variables['time']
            values = old.variables['value']
            values_p = old_p.variables['value']
            n = times.shape[0]

            # Convert new date to days
            date = dt.datetime(d.year, d.month, day=15)
            days = date - dt.datetime(1900, 1, 1)
            days = np.float64(days.days)

            # Convert new data to array
            array = gdal.Open(wgs_path).ReadAsArray()
            array_p = gdal.Open(albers_path).ReadAsArray()

            # Write changes to file and close
            times[n] = days
            times_p[n] = days
            values[n] = array
            values_p[n] = array_p
            old.close()
            old_p.close()

        # Now recreate the entire percentile data set
        print('Reranking percentiles...')
        pc_path = IP.join("netcdfs", "percentiles", index + '.nc')
        os.remove(pc_path)
        toNetCDFPercentile(nc_path, pc_path)


def wwdt():
    """Run everything."""

    for index in INDICES:
        nc_path = IP.join("netcdfs", index + '.nc')
        if os.path.exists(nc_path):
            print(nc_path + " exists, checking for missing data...")
            update_data(index)
        else:
            print(nc_path + " not detected, building new data set...")
            print("Downloading the 12 netcdf files for " + index + "...")
            create_data(index)


if __name__ == "__main__":

    print("Running Get_WWDT.py using a " + str(RES) + " degree resolution:\n")
    print(str(TODAY_STRING) + '\n')

    wwdt()

    print("Downloads and transformations complete, finding value ranges...")
    index_ranges()
    print("Update Complete.")
