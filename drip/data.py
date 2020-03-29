# -*- coding: utf-8 -*-
"""
Data aquisition and transformation functions for DrIP.

Created on Sat Mar 28 13:13:20 2020

@author: travis
"""

import datetime as dt
import numpy as np
import os
import pandas as pd
import xarray as xr

from collections import OrderedDict
from glob import glob
from netCDF4 import Dataset
from osgeo import gdal, osr, ogr
from scipy.stats import rankdata
from tqdm import tqdm

from drip.constants import TITLE_MAP

class Data_Path:
    """Data_Path joins a root directory path to data file paths."""

    def __init__(self, *data_paths):
        """Initialize Data_Path."""

        self.data_path = os.path.join(*data_paths)
        self._expand_check()

    def __repr__(self):

        items = ["=".join([str(k), str(v)]) for k, v in self.__dict__.items()]
        arguments = " ".join(items)
        msg = "".join(["<Data_Path " + arguments + ">"])
        return msg

    def join(self, *args):
        """Join a file path to the root directory path"""

        path = os.path.join(self.data_path, *args)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def contents(self, *args):
        """List all content in the data_path or in sub directories."""

        items = glob(self.join(*args, "*"))

        return items

    def folders(self, *args):
        """List folders in the data_path or in sub directories."""

        items = self.contents(*args)
        folders = [i for i in items if os.path.isdir(i)]

        return folders

    def files(self, *args):
        """List files in the data_path or in sub directories."""

        items = self.contents(*args)
        folders = [i for i in items if os.path.isfile(i)]
        print(self.join(*args))

        return folders

    def _expand_check(self):

        # Expand the user path if a tilda is present in the root folder path.
        if "~" in self.data_path:
            self.data_path = os.path.expanduser(self.data_path)
        
        # Make sure the data path exists.
        os.makedirs(self.data_path, exist_ok=True)


def isInt(string):
    try:
        int(string)
        return True
    except:
        return False


def percentile_arrays(arrays):
    """Convert a 3D time series of numpy arrays into percentiles.

    Parameters
    ----------
    arrays : numpy.ndarray
        A 3 Dimensional numpy array representing values of a time series of
        spatial layers.

    Returns
    -------
    numpy.ndarray
        A 3 Dimensional numpy array representing percentiles of the input time
        series of spatial layers.
    """

    def percentiles(lst):
        """Convert a single time series of values into percentiles."""

        pct = (rankdata(lst) / len(lst)) * 100
        return pct

    where_nan = np.where(np.isnan(arrays))
    pcts = np.apply_along_axis(percentiles, axis=0, arr=arrays)
    pcts[where_nan] = np.nan    

    return pcts


def readRaster(rasterpath, band, navalue=-9999):
    """
    rasterpath = path to folder containing a series of rasters
    navalue = a number (float) for nan values if we forgot
                to translate the file with one originally

    This converts a raster into a numpy array along with spatial features
    needed to write any results to a raster file. The return order is:

      array (numpy), spatial geometry (gdal object),
                                      coordinate reference system (gdal object)
    """
    raster = gdal.Open(rasterpath)
    geometry = raster.GetGeoTransform()
    arrayref = raster.GetProjection()
    array = np.array(raster.GetRasterBand(band).ReadAsArray())
    del raster
    array = array.astype(float)
    if np.nanmin(array) < navalue:
        navalue = np.nanmin(array)
    array[array == navalue] = np.nan
    return(array, geometry, arrayref)


def readRasters(files, navalue=-9999):
    """
    files = list of files to read in
    navalue = a number (float) for nan values if we forgot
                to translate the file with one originally

    This converts monthly rasters into numpy arrays and them as a list in another
            list. The other parts are the spatial features needed to write
            any results to a raster file. The list order is:

      [[name_date (string),arraylist (numpy)], spatial geometry (gdal object),
       coordinate reference system (gdal object)]

    The file naming convention required is: "INDEXNAME_YYYYMM.tif"
    """
    print("Converting raster to numpy array...")
    files = [f for f in files if os.path.isfile(f)]
    names = [os.path.basename(files[i]) for i in range(len(files))]
    sample = gdal.Open(files[1])
    geometry = sample.GetGeoTransform()
    arrayref = sample.GetProjection()
    alist = []
    for i in tqdm(range(0, len(files))):
        rast = gdal.Open(files[i])
        array = np.array(rast.GetRasterBand(1).ReadAsArray())
        array = array.astype(float)
        array[array == navalue] = np.nan
        name = str.upper(names[i][:-4])
        alist.append([name, array])
    return(alist, geometry, arrayref)


def shapeReproject(src, dst, src_epsg, dst_epsg):
    '''
    There doesn't appear to be an ogr2ogr analog in Python's OGR module.
    This simply reprojects a shapefile from the file.

    src = source file path
    dst = destination file path
    src_epsg = the epsg coordinate reference code for the source file
    dst_epsg = the epsg coordinate reference code for the destination file

    src = 'data/shapefiles/temp/temp.shp'
    dst = 'data/shapefiles/temp/temp.shp'
    src_epsg = 102008
    dst_epsg = 4326

    '''
    # Get the shapefile driver
    driver = ogr.GetDriverByName('ESRI Shapefile')

    # If dst and src match, overwrite is true, set a temporary dst name
    if dst == src:
        overwrite = True
        base, filename = os.path.split(dst)
        name, ext = os.path.splitext(filename)
        dst = os.path.join(base, name + '2' + ext)
    else:
        overwrite = False

    # Establish Coordinate Reference Systems
    src_crs = osr.SpatialReference()
    dst_crs = osr.SpatialReference()
    src_crs.ImportFromEPSG(src_epsg)
    dst_crs.ImportFromEPSG(dst_epsg)

    # Create the tranformation method
    transformation = osr.CoordinateTransformation(src_crs, dst_crs)

    # Get/Generate layers
    src_dataset = driver.Open(src)
    src_layer = src_dataset.GetLayer()
    if os.path.exists(dst):
        driver.DeleteDataSource(dst)
    dst_dataset = driver.CreateDataSource(dst)
    dst_file_name = os.path.basename(dst)
    dst_layer_name = os.path.splitext(dst_file_name)[0]
    dst_layer = dst_dataset.CreateLayer(dst_layer_name,
                                        geom_type=ogr.wkbMultiPolygon)

    # add Fields
    src_layer_defn = src_layer.GetLayerDefn()
    for i in range(0, src_layer_defn.GetFieldCount()):
        field_defn = src_layer_defn.GetFieldDefn(i)
        dst_layer.CreateField(field_defn)

    # Get Destination Layer Feature Definition
    dst_layer_defn = dst_layer.GetLayerDefn()

    # Project Features
    src_feature = src_layer.GetNextFeature()
    while src_feature:
        geom = src_feature.GetGeometryRef()
        geom.Transform(transformation)
        dst_feature = ogr.Feature(dst_layer_defn)
        dst_feature.SetGeometry(geom)
        for i in range(0, dst_layer_defn.GetFieldCount()):
            dst_feature.SetField(dst_layer_defn.GetFieldDefn(i).GetNameRef(),
                                 src_feature.GetField(i))
        dst_layer.CreateFeature(dst_feature)
        dst_feature = None
        src_feature = src_layer.GetNextFeature()

    # Set coordinate extents?
    dst_layer.GetExtent()

    # Save and close
    src_dataset = None
    dst_dataset = None

    # Overwrite if needed
    if overwrite is True:
        src_files = glob('data/shapefiles/temp/temp.*')
        dst_files = glob('data/shapefiles/temp/temp2.*')
        for sf in src_files:
            os.remove(sf)
        for df in dst_files:
            os.rename(df, df.replace('2', ''))


def standardize(indexlist):
    '''
    Min/max standardization
    '''
    def single(array, mins, maxes):
        newarray = (array - mins)/(maxes - mins)
        return newarray

    if type(indexlist[0][0]) == str:
        arrays = [a[1] for a in indexlist]
        mins = np.nanmin(arrays)
        maxes = np.nanmax(arrays)
        standardizedlist = [[indexlist[i][0],
                             single(indexlist[i][1],
                                    mins,
                                    maxes)] for i in range(len(indexlist))]

    else:
        mins = np.nanmin(indexlist)
        maxes = np.nanmax(indexlist)
        standardizedlist = [single(indexlist[i],
                                   mins, maxes) for i in range(len(indexlist))]
    return standardizedlist



def wgs_netcdf(arrays, dst, proj, template, epsg=4326, wmode='w'):
    """Take a ndarray and write to netcdf. Use a template to get
    attributes"""

    # For attributes
    todays_date = dt.datetime.today()
    today = np.datetime64(todays_date)
    index = os.path.splitext(os.path.basename(dst))[0]

    # Create data set
    nco = Dataset(dst, mode=wmode, format='NETCDF4')

    # We need some things from the template nc file
    temp = xr.open_dataset(template)
    base = dt.datetime(1900, 1, 1)
    days = temp.variables['day'][:].data
    days = [pd.Timestamp(d).to_pydatetime() for d in days]
    days = [(d - base).days for d in days]

    # Read raster for the structure
    data = gdal.Open(template)
    geom = data.GetGeoTransform()
    res = geom[1]
    navalue = data.GetRasterBand(1).GetNoDataValue()
    ntime, nlat, nlon = np.shape(arrays)
    lons = np.arange(nlon) * geom[1] + geom[0]
    lats = np.arange(nlat) * geom[5] + geom[3]
    del data

    # Dimensions
    nco.createDimension('lat', nlat)
    nco.createDimension('lon', nlon)
    nco.createDimension('time', None)

    # Variables
    latitudes = nco.createVariable('lat', 'f4', ('lat', ))
    longitudes = nco.createVariable('lon', 'f4', ('lon', ))
    times = nco.createVariable('time', 'f8', ('time', ))
    variable = nco.createVariable('value', 'f4', ('time', 'lat', 'lon'),
                                  fill_value=navalue)#, zlib=True
    variable.standard_name = 'index'
    variable.units = 'unitless'
    variable.long_name = 'Index Value'

    # Appending the CRS information
    refs = osr.SpatialReference()
    refs.ImportFromEPSG(epsg)
    crs = nco.createVariable('crs', 'c')
    variable.setncattr('grid_mapping', 'crs')
    crs.geographic_crs_name = 'WGS 84'
    crs.spatial_ref = proj
    crs.epsg_code = "EPSG:4326"  # How about this?
    crs.GeoTransform = geom
    crs.long_name = 'Lon/Lat WGS 84'
    crs.grid_mapping_name = 'latitude_longitude'
    crs.longitude_of_prime_meridian = 0.0
    crs.semi_major_axis = refs.GetSemiMajor()
    crs.inverse_flattening = refs.GetInvFlattening()

    # Attributes
    nco.title = TITLE_MAP[index]
    nco.subtitle = "Monthly Index values since 1895-01-15"
    nco.description = ("Monthly gridded data at " + str(res) +
                       " decimal degree (15 arc-minute resolution, calibrated "
                       "to 1895-2010 for the continental United States."),
    nco.original_author = 'John Abatzoglou - University of Idaho'
    nco.date = pd.to_datetime(str(today)).strftime('%Y-%m-%d')
    nco.projection = 'WGS 1984 EPSG: 4326'
    nco.citation = ('Westwide Drought Tracker, ' +
                    'http://www.wrcc.dri.edu/monitor/WWDT')
    nco.Conventions = 'CF-1.6'

    # Variable Attrs
    times.units = 'days since 1900-01-01'
    times.standard_name = 'time'
    times.calendar = 'gregorian'
    latitudes.units = 'degrees_north'
    latitudes.standard_name = 'latitude'
    longitudes.units = 'degrees_east'
    longitudes.standard_name = 'longitude'

    # Write - set this to write one or multiple
    latitudes[:] = lats
    longitudes[:] = lons
    times[:] = days
    variable[:] = arrays

    # Done
    nco.close()


def albers_netcdf(arrays, dst,  proj,template, epsg=102008, wmode='w'):
    """Take a multiband geotiff file and write to netcdf. Use a template to get
    attributes"""

    # For attributes
    todays_date = dt.datetime.today()
    today = np.datetime64(todays_date)
    index = os.path.splitext(os.path.basename(dst))[0]

    # Create data set
    nco = Dataset(dst, mode=wmode, format='NETCDF4')

    # We need some things from the template nc file
    temp = xr.open_dataset(template)
    base = dt.datetime(1900, 1, 1)
    days = temp.variables['day'][:].data
    days = [pd.Timestamp(d).to_pydatetime() for d in days]
    days = [(d - base).days for d in days]

    # Read raster for the structure
    data = gdal.Open(template)
    geom = data.GetGeoTransform()
    res = geom[1]
    navalue = data.GetRasterBand(1).GetNoDataValue()
    ntime, nlat, nlon = np.shape(arrays)
    lons = np.arange(nlon) * geom[1] + geom[0]
    lats = np.arange(nlat) * geom[5] + geom[3]
    del data

    # Dimensions
    nco.createDimension('lat', nlat)
    nco.createDimension('lon', nlon)
    nco.createDimension('time', None)

    # Variables
    latitudes = nco.createVariable('lat', 'f4', ('lat', ))
    longitudes = nco.createVariable('lon', 'f4', ('lon', ))
    times = nco.createVariable('time', 'f8', ('time', ))
    variable = nco.createVariable('value', 'f4', ('time', 'lat', 'lon'),
                                  fill_value=navalue)
    variable.standard_name = 'index'
    variable.units = 'unitless'
    variable.long_name = 'Index Value'

    # Appending the CRS information
    crs = nco.createVariable('crs', 'c')
    variable.setncattr('grid_mapping', 'crs')
    crs.spatial_ref = proj
    crs.GeoTransform = geom
    crs.grid_mapping_name = 'albers_conical_equal_area'
    crs.standard_parallel = [20.0, 60.0]
    crs.longitude_of_central_meridian = -32.0
    crs.latitude_of_projection_origin = 40.0
    crs.false_easting = 0.0
    crs.false_northing = 0.0

    # Attributes
    nco.title = TITLE_MAP[index]
    nco.subtitle = "Monthly Index values since 1895-01-15"
    nco.description = ("Monthly gridded data at " + str(res) +
                       " decimal degree (15 arc-minute resolution, calibrated "
                       "to 1895-2010 for the continental United States."),
    nco.original_author = 'John Abatzoglou - University of Idaho'
    nco.date = pd.to_datetime(str(today)).strftime('%Y-%m-%d')
    nco.projection = 'Albers Conical Equal Area, North America:  EPSG: 102008'
    nco.citation = ('Westwide Drought Tracker, ' +
                    'http://www.wrcc.dri.edu/monitor/WWDT')
    nco.Conventions = 'CF-1.6'

    # Variable Attrs
    times.units = 'days since 1900-01-01'
    times.standard_name = 'time'
    times.calendar = 'gregorian'
    latitudes.units = 'meters'
    latitudes.standard_name = 'projection_y_coordinate'
    longitudes.units = 'meters'
    longitudes.standard_name = 'projection_x_coordinate'

    # Write - set this to write one or multiple
    latitudes[:] = lats
    longitudes[:] = lons
    times[:] = days
    variable[:] = arrays

    # Done
    nco.close()


def toNetCDFSingle(file, ncfile, savepath, index, epsg=4326, wmode='w'):
    '''Take an individual tif and either write or append to netcdf.'''
    # For attributes
    todays_date = dt.datetime.today()
    today = np.datetime64(todays_date)

    # Create data set
    nco = Dataset(savepath, mode=wmode, format='NETCDF4')

    # We need some things from the old nc file
    data = Dataset(ncfile)
    days = data.variables['day'][0]  # This is in days since 1900

    # Read raster for the structure
    data = gdal.Open(file)
    geom = data.GetGeoTransform()
    proj = data.GetProjection()
    array = data.ReadAsArray()
    array[array == -9999.] = np.nan
    nlat, nlon = np.shape(array)
    lons = np.arange(nlon) * geom[1] + geom[0]
    lats = np.arange(nlat) * geom[5] + geom[3]
    del data

    # Dimensions
    nco.createDimension('lat', nlat)
    nco.createDimension('lon', nlon)
    nco.createDimension('time', None)

    # Variables
    latitudes = nco.createVariable('lat', 'f4', ('lat', ))
    longitudes = nco.createVariable('lon', 'f4', ('lon', ))
    times = nco.createVariable('time', 'f8', ('time', ))
    variable = nco.createVariable('value', 'f4', ('time', 'lat', 'lon'),
                                  fill_value=-9999)
    variable.standard_name = 'index'
    variable.units = 'unitless'
    variable.long_name = 'Index Value'

    # Appending the CRS information
    # EPSG information
    refs = osr.SpatialReference()
    refs.ImportFromEPSG(epsg)
    crs = nco.createVariable('crs', 'c')
    variable.setncattr('grid_mapping', 'crs')
    crs.geographic_crs_name = 'WGS 84'  # is this buried in refs anywhere?
    crs.spatial_ref = proj
    crs.epsg_code = "EPSG:4326"  # How about this?
    crs.GeoTransform = geom
    crs.long_name = 'Lon/Lat WGS 84'
    crs.grid_mapping_name = 'latitude_longitude'
    crs.longitude_of_prime_meridian = 0.0
    crs.semi_major_axis = refs.GetSemiMajor()
    crs.inverse_flattening = refs.GetInvFlattening()

    # Attributes
    # Global Attrs
    nco.title = TITLE_MAP[index]
    nco.subtitle = "Monthly Index values since 1948-01-01"
    nco.description = ('Monthly gridded data at 0.25 decimal degree' +
                       ' (15 arc-minute resolution, calibrated to 1895-2010 ' +
                       ' for the continental United States.'),
    nco.original_author = 'John Abatzoglou - University of Idaho'
    nco.date = pd.to_datetime(str(today)).strftime('%Y-%m-%d')
    nco.projection = "WGS 1984 EPSG: 4326"
    nco.citation = ("Westwide Drought Tracker, "
                    "http://www.wrcc.dri.edu/monitor/WWDT")
    nco.Conventions = "CF-1.6"

    # Variable Attrs
    times.units = "days since 1900-01-01"
    times.standard_name = "time"
    times.calendar = "gregorian"
    latitudes.units = "degrees_north"
    latitudes.standard_name = "latitude"
    longitudes.units = "degrees_east"
    longitudes.standard_name = "longitude"

    # Write - set this to write one or multiple
    latitudes[:] = lats
    longitudes[:] = lons
    times[:] = int(days)
    variable[0, :,] = array

    # Done
    nco.close()


def toNetCDF(tfiles, ncfiles, savepath, index, year1, month1, year2, month2,
             proj=4326, percentiles=False, wmode='w'):
    '''
    Take multiple multiband netcdfs with unordered dates and multiple tiffs
    with desired geometries and write to a single netcdf as a single time
    series. This has a lot of options and is only meant for the app.

    As an expediency, if there isn't an nc file it defaults to reading dates
    from the file names.

    I need to go back and parameterize the subtitle and other attributes to
    reflect the actual dates used in each dataset.  # <------------------------ Not critical since we aren't sharing these files yet, but do this before we do

    Test parameters for toNetCDF2
        tfiles = glob('f:/data/droughtindices/netcdfs/wwdt/tifs/temp*')
        ncfiles = glob('f:/data/droughtindices/netcdfs/wwdt/*nc')
        savepath = 'testing.nc'
        index = 'spi1'
        year1=1948
        month1=1
        year2=2019
        month2=12
        epsg=4326
        percentiles=False
        wmode='w'
    '''
    # For attributes
    todays_date = dt.datetime.today()
    today = np.datetime64(todays_date)

    # Use one tif (one array) for spatial attributes
    data = gdal.Open(tfiles[0])
    geom = data.GetGeoTransform()
    res = abs(geom[1])
    proj = data.GetProjection()
    array = data.ReadAsArray()
    if len(array.shape) == 3:
        ntime, nlat, nlon = np.shape(array)
    else:
        nlat, nlon = np.shape(array)
    lons = np.arange(nlon) * geom[1] + geom[0]
    lats = np.arange(nlat) * geom[5] + geom[3]
    del data

    # use osr for more spatial attributes
    refs = osr.SpatialReference()
    if type(proj) is int:
        refs.ImportFromEPSG(proj)
    elif '+' in proj:
        refs.ImportFromProj4(proj)

    # Create Dataset
    nco = Dataset(savepath, mode=wmode, format='NETCDF4')

    # Dimensions
    nco.createDimension('lat', nlat)
    nco.createDimension('lon', nlon)
    nco.createDimension('time', None)

    # Variables
    latitudes = nco.createVariable('lat', 'f4', ('lat', ))
    longitudes = nco.createVariable('lon', 'f4', ('lon', ))
    times = nco.createVariable('time', 'f8', ('time', ))
    variable = nco.createVariable('value', 'f4', ('time', 'lat', 'lon'),
                                  fill_value=-9999)
    variable.standard_name = 'index'
    variable.units = 'unitless'
    variable.long_name = 'Index Value'

    # Appending the CRS information
    crs = nco.createVariable('crs', 'c')
    variable.setncattr('grid_mapping', 'crs')
    crs.spatial_ref = proj
    if type(crs) is int:
        crs.epsg_code = "EPSG:" + str(proj)
    elif '+' in proj:
        crs.proj4 = proj
    crs.GeoTransform = geom
    crs.grid_mapping_name = 'latitude_longitude'
    crs.longitude_of_prime_meridian = 0.0
    crs.semi_major_axis = refs.GetSemiMajor()
    crs.inverse_flattening = refs.GetInvFlattening()

    # Attributes
    # Global Attrs
    nco.title = TITLE_MAP[index]
    nco.subtitle = "Monthly Index values since 1895-01-01"
    nco.description = ('Monthly gridded data at '+ str(res) +
                       ' decimal degree (15 arc-minute resolution, ' +
                       'calibrated to 1895-2010 for the continental ' +
                       'United States.'),
    nco.original_author = 'John Abatzoglou - University of Idaho'
    nco.date = pd.to_datetime(str(today)).strftime('%Y-%m-%d')
    nco.projection = 'WGS 1984 EPSG: 4326'
    nco.citation = ('Westwide Drought Tracker, ' +
                    'http://www.wrcc.dri.edu/monitor/WWDT')
    nco.Conventions = 'CF-1.6'  # Should I include this if I am not sure?

    # Variable Attrs
    times.units = 'days since 1900-01-01'
    times.standard_name = 'time'
    times.calendar = 'gregorian'
    latitudes.units = 'degrees_south'
    latitudes.standard_name = 'latitude'
    longitudes.units = 'degrees_east'
    longitudes.standard_name = 'longitude'

    # Now getting the data, which is not in order because of how wwdt does it
    # We need to associate each day with its array
    try:
        tfiles.sort()
        ncfiles.sort()
        test = Dataset(ncfiles[0])
        test.close()
        date_tifs = {}
        for i in range(len(ncfiles)):
            nc = Dataset(ncfiles[i])
            days = nc.variables['day'][:]
            rasters = gdal.Open(tfiles[i])
            arrays = rasters.ReadAsArray()
            for y in range(len(arrays)):
                date_tifs[days[y]] = arrays[y]

        # okay, that was just in case the dates wanted to bounce around
        date_tifs = OrderedDict(sorted(date_tifs.items()))

        # Now that everything is in the right order, split them back up
        days = np.array(list(date_tifs.keys()))
        arrays = np.array(list(date_tifs.values()))

    except Exception as e:
        tfiles.sort()

        # print('Combining data using filename dates...')
        datestrings = [f[-10:-4] for f in tfiles if isInt(f[-10:-4])]
        dates = [dt.datetime(year=int(d[:4]), month=int(d[4:]), day=15) for
                 d in datestrings]
        deltas = [d - dt.datetime(1900, 1, 1) for d in dates]
        days = np.array([d.days for d in deltas])
        arrays = []
        for t in tfiles:
            data = gdal.Open(t)
            array = data.ReadAsArray()
            arrays.append(array)
        arrays = np.array(arrays)

    # Filter out dates
    base = dt.datetime(1900, 1, 1)
    start = dt.datetime(year1, month1, 1)
    day1 = start - base
    day1 = day1.days
    end = dt.datetime(year2, month2, 1)
    day2 = end - base
    day2 = day2.days
    idx = len(days) - len(days[np.where(days >= day1)])
    idx2 = len(days[np.where(days < day2)])
    days = days[idx:idx2]
    arrays = arrays[idx:idx2]

    # This allows the option to store the data as percentiles
    if percentiles:
        arrays[arrays == -9999] = np.nan
        arrays = percentile_arrays(arrays)

    # Write - set this to write one or multiple
    latitudes[:] = lats
    longitudes[:] = lons
    times[:] = days.astype(int)
    variable[:, :, :] = arrays

    # Done
    nco.close()


def toNetCDFAlbers(tfiles, ncfiles, savepath, index, year1, month1,
                   year2, month2, proj=4326, percentiles=False, wmode='w'):
    '''
    This does the same as above but is specific to the north american
    albers equal area conic projection

    Test parameters for toNetCDF2
        tfiles = glob('f:/data/droughtindices/netcdfs/wwdt/tifs/proj*')
        ncfiles = glob('f:/data/droughtindices/netcdfs/wwdt/temp_*[0-9]*.nc')
        savepath = 'f:/data/droughtindices/netcdfs/wwdt/testing.nc'
        index = 'spi1'
        year1=1895
        month1=1
        year2=2019
        month2=12
        epsg=102008
        percentiles=False
        wmode='w'
    '''
    # For attributes
    todays_date = dt.datetime.today()
    today = np.datetime64(todays_date)

    # Use one tif (one array) for spatial attributes
    data = gdal.Open(tfiles[0])
    geom = data.GetGeoTransform()
    res = abs(geom[1])
    proj = data.GetProjection()
    array = data.ReadAsArray()
    if len(array.shape) == 3:
        ntime, nlat, nlon = np.shape(array)
    else:
        nlat, nlon = np.shape(array)
    lons = np.arange(nlon) * geom[1] + geom[0]
    lats = np.arange(nlat) * geom[5] + geom[3]
    del data

    # use osr for more spatial attributes
    refs = osr.SpatialReference()
    refs.ImportFromProj4(proj)

    # Create Dataset
    nco = Dataset(savepath, mode=wmode, format='NETCDF4')

    # Dimensions
    nco.createDimension('lat', nlat)
    nco.createDimension('lon', nlon)
    nco.createDimension('time', None)

    # Variables
    latitudes = nco.createVariable('lat', 'f4', ('lat', ))
    longitudes = nco.createVariable('lon', 'f4', ('lon', ))
    times = nco.createVariable('time', 'f8', ('time', ))
    variable = nco.createVariable('value', 'f4', ('time', 'lat', 'lon'),
                                  fill_value=-9999)
    variable.standard_name = 'index'
    variable.units = 'unitless'
    variable.long_name = 'Index Value'

    # Appending the CRS information
    crs = nco.createVariable('crs', 'c')
    variable.setncattr('grid_mapping', 'crs')
    crs.spatial_ref = proj
    # crs.epsg_code = "EPSG:" + str(proj)
    crs.GeoTransform = geom
    crs.grid_mapping_name = 'albers_conical_equal_area'
    crs.standard_parallel = [20.0, 60.0]
    crs.longitude_of_central_meridian = -32.0
    crs.latitude_of_projection_origin = 40.0
    crs.false_easting = 0.0
    crs.false_northing = 0.0

    # Attributes
    # Global Attrs
    nco.title = TITLE_MAP[index]
    nco.subtitle = "Monthly Index values since 1895-01-01"
    nco.description = ('Monthly gridded data at '+ str(res) +
                       ' decimal degree (15 arc-minute resolution, ' +
                       'calibrated to 1895-2010 for the continental ' +
                       'United States.'),
    nco.original_author = 'John Abatzoglou - University of Idaho'
    nco.date = pd.to_datetime(str(today)).strftime('%Y-%m-%d')
    nco.projection = 'Albers Conical Equal Area, North America:  EPSG: 102008'
    nco.citation = ('Westwide Drought Tracker, ' +
                    'http://www.wrcc.dri.edu/monitor/WWDT')
    nco.Conventions = 'CF-1.6'  # Should I include this if I am not sure?

    # Variable Attrs
    times.units = 'days since 1900-01-01'
    times.standard_name = 'time'
    times.calendar = 'gregorian'
    latitudes.units = 'meters'
    latitudes.standard_name = 'projection_y_coordinate'
    longitudes.units = 'meters'
    longitudes.standard_name = 'projection_x_coordinate'

    # Now getting the data, which is not in order because of how wwdt does it.
    # We need to associate each day with its array, let's sort files to start
    # Dates may be gotten from either the original nc files or tif filenames
    try:
        ncfiles.sort()
        tfiles.sort()
        test = Dataset(ncfiles[0])
        test.close()
        date_tifs = {}
        for i in range(len(ncfiles)):
            nc = Dataset(ncfiles[i])
            days = nc.variables['day'][:]
            rasters = gdal.Open(tfiles[i])
            arrays = rasters.ReadAsArray()
            for y in range(len(arrays)):
                date_tifs[days[y]] = arrays[y]

        # okay, that was just in case the dates wanted to bounce around
        date_tifs = OrderedDict(sorted(date_tifs.items()))

        # Now that everything is in the right order, split them back up
        days = np.array(list(date_tifs.keys()))
        arrays = np.array(list(date_tifs.values()))

    except Exception as e:
        tfiles.sort()
        datestrings = [f[-10:-4] for f in tfiles if isInt(f[-10:-4])]
        dates = [dt.datetime(year=int(d[:4]), month=int(d[4:]), day=15) for
                 d in datestrings]
        deltas = [d - dt.datetime(1900, 1, 1) for d in dates]
        days = np.array([d.days for d in deltas])
        arrays = []
        for t in tfiles:
            data = gdal.Open(t)
            array = data.ReadAsArray()
            arrays.append(array)
        arrays = np.array(arrays)

    # Filter out dates
    base = dt.datetime(1900, 1, 1)
    start = dt.datetime(year1, month1, 1)
    day1 = start - base
    day1 = day1.days
    end = dt.datetime(year2, month2, 1)
    day2 = end - base
    day2 = day2.days
    idx = len(days) - len(days[np.where(days >= day1)])
    idx2 = len(days[np.where(days < day2)])
    days = days[idx:idx2]
    arrays = arrays[idx:idx2]

    # This allows the option to store the data as percentiles
    if percentiles:
        arrays[arrays == -9999] = np.nan
        arrays = percentile_arrays(arrays)

    # Write - set this to write one or multiple
    latitudes[:] = lats
    longitudes[:] = lons
    times[:] = days.astype(int)
    variable[:, :, :] = arrays

    # Done
    nco.close()


def toNetCDF3(tfile, ncfile, savepath, index, epsg=102008, percentiles=False,
              wmode='w'):
    '''
    Unlike toNetCDF2, this takes a multiband netcdf with correct dates and a
    single tiff with desired geometry to write to a single netcdf as
    a single time series projected to the North American Albers Equal Area
    Conic Projection.

    Still need to parameterize grid mapping and coordinate names.
    '''
    # For attributes
    todays_date = dt.datetime.today()
    today = np.datetime64(todays_date)

    # Use one tif (one array) for spatial attributes
    data = gdal.Open(tfile)
    geom = data.GetGeoTransform()
    proj = data.GetProjection()
    arrays = data.ReadAsArray()
    ntime, nlat, nlon = np.shape(arrays)
    lons = np.arange(nlon) * geom[1] + geom[0]
    lats = np.arange(nlat) * geom[5] + geom[3]
    del data

    # use osr for more spatial attributes
    refs = osr.SpatialReference()
    refs.ImportFromEPSG(epsg)

    # Create Dataset
    nco = Dataset(savepath, mode=wmode, format='NETCDF4')

    # Dimensions
    nco.createDimension('lat', nlat)
    nco.createDimension('lon', nlon)
    nco.createDimension('time', None)

    # Variables
    latitudes = nco.createVariable('lat', 'f4', ('lat',))
    longitudes = nco.createVariable('lon', 'f4', ('lon',))
    times = nco.createVariable('time', 'f8', ('time',))
    variable = nco.createVariable('value', 'f4', ('time', 'lat', 'lon'),
                                  fill_value=-9999)
    variable.standard_name = 'index'
    variable.units = 'unitless'
    variable.long_name = 'Index Value'

    # Appending the CRS information
    crs = nco.createVariable('crs', 'c')
    variable.setncattr('grid_mapping', 'crs')
    crs.spatial_ref = proj
    crs.epsg_code = "EPSG:" + str(epsg)
    crs.GeoTransform = geom
    crs.grid_mapping_name = 'albers_conical_equal_area'
    crs.standard_parallel = [20.0, 60.0]
    crs.longitude_of_central_meridian = -32.0
    crs.latitude_of_projection_origin = 40.0
    crs.false_easting = 0.0
    crs.false_northing = 0.0

    # Attributes
    # Global Attrs
    nco.title = TITLE_MAP[index]
    nco.subtitle = "Monthly Index values since 1895-01-01"
    nco.description = ('Monthly gridded data at 0.25 decimal degree' +
                       ' (15 arc-minute resolution, calibrated to 1895-2010 ' +
                       ' for the continental United States.'),
    nco.original_author = 'John Abatzoglou - University of Idaho'
    nco.date = pd.to_datetime(str(today)).strftime('%Y-%m-%d')
    nco.projection = 'WGS 1984 EPSG: 4326'
    nco.citation = ('Westwide Drought Tracker, ' +
                    'http://www.wrcc.dri.edu/monitor/WWDT')
    nco.Conventions = 'CF-1.6'

    # Variable Attrs
    times.units = 'days since 1900-01-01'
    times.standard_name = 'time'
    times.calendar = 'gregorian'
    latitudes.units = 'meters'
    latitudes.standard_name = 'projection_y_coordinate'
    longitudes.units = 'meters'
    longitudes.standard_name = 'projection_x_coordinate'

    # Now getting the data, which is not in order because of how wwdt does it
    # We need to associate each day with its array
    nc = Dataset(ncfile)

    # Make sure there are the same number of time steps
    if ntime != len(nc.variables['time']):
        print("Time lengths don't match.")
        sys.exit(1)

    days = nc.variables['time'][:]

    # This allows the option to store the data as percentiles
    if percentiles:
        arrays = percentile_arrays(arrays)

    # Write - set this to write one or multiple
    latitudes[:] = lats
    longitudes[:] = lons
    times[:] = days.astype(int)
    variable[:, :, :] = arrays

    # Done
    nco.close()


def toNetCDFPercentile(src_path, dst_path):
    '''
    Take an existing netcdf file and simply transform the data into percentile
    space

    Sample arguments:
    src_path = 'f:/data/droughtindices/netcdfs/spi2.nc'
    dst_path = 'f:/data/droughtindices/netcdfs/percentiles/spi2.nc'

    src = Dataset(src_path)
    dst = Dataset(dst_path, 'w')
    '''
    with Dataset(src_path) as src, Dataset(dst_path, 'w') as dst:

        # copy attributes
        for name in src.ncattrs():
            dst.setncattr(name, src.getncattr(name))

        # Some attributes need to change
        dst.setncattr('subtitle',
                      'Monthly percentile values since 1895')
        dst.setncattr('standard_name', 'percentile')

        # set dimensions
        nlat = src.dimensions['lat'].size
        nlon = src.dimensions['lon'].size
        dst.createDimension('lat', nlat)
        dst.createDimension('lon', nlon)
        dst.createDimension('time', None)

        # set variables
        latitudes = dst.createVariable('lat', 'f4', ('lat', ))
        longitudes = dst.createVariable('lon', 'f4', ('lon', ))
        times = dst.createVariable('time', 'f8', ('time', ))
        variable = dst.createVariable('value', 'f4',
                                      ('time', 'lat', 'lon'),
                                      fill_value=-9999)
        crs = dst.createVariable('crs', 'c')
        variable.setncattr('grid_mapping', 'crs')

        # Set coordinate system attributes
        src_crs = src.variables['crs']
        for name in src_crs.ncattrs():
            crs.setncattr(name, src_crs.getncattr(name))

        # Variable Attrs
        times.units = 'days since 1900-01-01'
        times.standard_name = 'time'
        times.calendar = 'gregorian'
        latitudes.units = 'degrees_north'
        latitudes.standard_name = 'latitude'
        longitudes.units = 'degrees_east'
        longitudes.standard_name = 'longitude'

        # Set most values
        latitudes[:] = src.variables['lat'][:]
        longitudes[:] = src.variables['lon'][:]
        times[:] = src.variables['time'][:]

        # finally rank and transform values into percentiles
        values = src.variables['value'][:]
        percentiles = percentile_arrays(values)
        variable[:] = percentiles


def toRaster(array, path, geometry, srs, navalue=-9999):
    """
    Writes a single array to a raster with coordinate system and geometric
    information.

    path = target path
    srs = spatial reference system
    """
    xpixels = array.shape[1]
    ypixels = array.shape[0]
    path = path.encode('utf-8')
    image = gdal.GetDriverByName("GTiff").Create(path, xpixels, ypixels, 1,
                                                 gdal.GDT_Float32)
    image.SetGeoTransform(geometry)
    image.SetProjection(srs)
    image.GetRasterBand(1).WriteArray(array)
    image.GetRasterBand(1).SetNoDataValue(navalue)


def toRasters(arraylist, path, geometry, srs):
    """
    Writes a list of 2d arrays, or a 3d array, to a series of tif files.

    Arraylist format = [[name,array],[name,array],....]
    path = target path
    geometry = gdal geometry object
    srs = spatial reference system object
    """
    if path[-2:] == "\\":
        path = path
    else:
        path = path + "\\"
    sample = arraylist[0][1]
    ypixels = sample.shape[0]
    xpixels = sample.shape[1]
    for ray in  tqdm(arraylist):
        path = os.path.join(path, ray[0] + ".tif")
        image = gdal.GetDriverByName("GTiff").Create(path, xpixels, ypixels,
                                                     1, gdal.GDT_Float32)
        image.SetGeoTransform(geometry)
        image.SetProjection(srs)
        image.GetRasterBand(1).WriteArray(ray[1])


def wgsToAlbers(arrays, crdict, proj_sample):
    '''
    Takes an xarray dataset in WGS 84 (epsg: 4326) with a specified mask and
    returns that mask projected to Alber's North American Equal Area Conic
    (epsg: 102008).
    '''
    # dates = range(len(arrays.time))
    wgs_proj = Proj(init='epsg:4326')
    geom = crdict.source.transform
    wgrid = salem.Grid(nxny=(crdict.x_length, crdict.y_length),
                       dxdy=(crdict.res, -crdict.res),
                       x0y0=(geom[0], geom[3]), proj=wgs_proj)
    lats = np.unique(wgrid.xy_coordinates[1])
    lats = lats[::-1]
    lons = np.unique(wgrid.xy_coordinates[0])
    data_array = xr.DataArray(data=arrays.value[0],
                              coords=[lats, lons],
                              dims=['lat', 'lon'])
    wgs_data = xr.Dataset(data_vars={'value': data_array})

    # Albers Equal Area Conic North America (epsg not working)
    albers_proj = Proj('+proj=aea +lat_1=20 +lat_2=60 +lat_0=40 \
                        +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 \
                        +datum=NAD83 +units=m +no_defs')

    # Create an albers grid
    geom = proj_sample.crs.GeoTransform
    array = proj_sample.value[0].values
    res = geom[1]
    x_length = array.shape[1]
    y_length = array.shape[0]
    agrid = salem.Grid(nxny=(x_length, y_length), dxdy=(res, -res),
                       x0y0=(geom[0], geom[3]), proj=albers_proj)
    lats = np.unique(agrid.xy_coordinates[1])
    lats = lats[::-1]
    lons = np.unique(agrid.xy_coordinates[0])
    data_array = xr.DataArray(data=array,
                              coords=[lats, lons],
                              dims=['lat', 'lon'])
    data_array = data_array
    albers_data = xr.Dataset(data_vars={'value': data_array})
    albers_data.salem.grid._proj = albers_proj
    projection = albers_data.salem.transform(wgs_data, 'linear')
    proj_mask = projection.value.data
    proj_mask = proj_mask * 0 + 1

    # Set up grid info from coordinate dictionary
    nlat, nlon = proj_mask.shape
    xs = np.arange(nlon) * geom[1] + geom[0]
    ys = np.arange(nlat) * geom[5] + geom[3]


    # Create mask xarray
    proj_mask = xr.DataArray(proj_mask,
                             coords={'lat': ys.astype(np.float32),
                                     'lon': xs.astype(np.float32)},
                             dims={'lat': len(ys),
                                   'lon': len(xs)})
    return proj_mask

