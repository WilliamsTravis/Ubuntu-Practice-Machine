# -*- coding: utf-8 -*-
"""
Support functions for Drought-Index-Portal

Created on Tue Jan 22 18:02:17 2019

@author: User
"""

import gc
import json
import os
import warnings

from collections import OrderedDict

import datetime as dt
import dask
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from dateutil.relativedelta import relativedelta
from dash.exceptions import PreventUpdate
from numba import jit
from osgeo import gdal, ogr, osr

from drip.data import toRaster, readRaster, wgsToAlbers
from drip.constants import NONINDICES

warnings.filterwarnings("ignore")


@jit(nopython=True)
def correlationField(ts, arrays):
    '''
    Create a 2d array of pearson correlation coefficient between the time
    series at the location of the grid id and every other grid in a 3d array
    '''
    # Apply that to each cell
    one_field = np.zeros((arrays.shape[1], arrays.shape[2]))
    for i in range(arrays.shape[1]):
        for j in range(arrays.shape[2]):
            lst = arrays[:, i, j]
            cor = np.corrcoef(ts, lst)[0, 1]
            one_field[i, j] = cor
    return one_field


def datePrint(y1, y2, m1, m2, month_filter, monthmarks):
    if y1 != y2:
        if len(month_filter) == 12:
            if m1 == 1 and m2 == 12:
                date_print = '{} - {}'.format(y1, y2)
            elif m1 != 1 or m2 != 12:
                date_print = (monthmarks[m1] + ' ' + str(y1) + ' - ' +
                              monthmarks[m2] + ' ' + str(y2))
        else:
            letters = "".join([monthmarks[m][0] for m in month_filter])
            date_print = '{} - {}'.format(y1, y2) + ' ' + letters
    elif y1 == y2:
        if len(month_filter) == 12:
            if m1 == 1 and m2 == 12:
                date_print = '{}'.format(y1)
            elif m1 != 1 or m2 != 12:
                date_print = (monthmarks[m1] + ' - ' +
                              monthmarks[m2] + ' ' + str(y2))
        else:
            letters = "".join([monthmarks[m][0] for m in month_filter])
            date_print = '{}'.format(y1) + ' ' + letters
    return date_print


def im(array):
    '''
    This just plots an array as an image
    '''
    fig = plt.imshow(array)
    fig.figure.canvas.raise_()


# For making outlines...move to css, maybe
def outLine(color, width):
    string = ('-{1}px -{1}px 0 {0}, {1}px -{1}px 0 {0}, ' +
              '-{1}px {1}px 0 {0}, {1}px {1}px 0 {0}').format(color, width)
    return string



def print_assign(name, variable):
    """This will print a variable assignment from the application scope to
    help with troubleshooting."""

    if isinstance(variable, str):
        variable = "'" + variable + "'"
    print("{} = {}".format(name, variable))
    

class Admin_Elements:

    def __init__(self, resolution, data_path):
        self.resolution = resolution
        self.res_ext = "_" + str(round(resolution, 3)).replace(".", "_")
        self.data_path = data_path

    def buildAdmin(self):

        # Build the resolution extension to identify datasets
        res_ext = self.res_ext
        county_path = self._paths("rasters", "us_counties" + res_ext + ".tif")
        state_path = self._paths("rasters", "us_states" + res_ext + ".tif")

        # Use the shapefile for just the county, it has state and county fips
        src_path = self._paths("shapefiles", "contiguous_counties.shp")

        # And rasterize
        self.rasterize(src_path, county_path, attribute='COUNTYFP',
                       extent=[-130, 50, -55, 20])
        self.rasterize(src_path, state_path, attribute='STATEFP',
                       extent=[-130, 50, -55, 20])

    def buildAdminDF(self):

        res_ext = self.res_ext
        grid_path = self._paths("rasters/grid" + res_ext + ".tif")
        gradient_path = self._paths("rasters/gradient" + res_ext + ".tif")
        county_path = self._paths("rasters/us_counties" + res_ext + ".tif")
        state_path = self._paths("rasters/us_states" + res_ext + ".tif")
        admin_path = self._paths("tables/admin_df" + res_ext + ".csv")

        # There are several administrative elements used in the app
        fips = pd.read_csv(self._paths("tables/US_FIPS_Codes.csv"),
                           skiprows=1, index_col=0)
        states = pd.read_table(self._paths("tables/state_fips.txt"), sep="|")
        states = states[["STATE_NAME", "STUSAB", "STATE"]]

        # Read, mask and flatten the arrays
        def flttn(array_path):
            '''
            Mask and flatten the grid array
            '''
            grid = gdal.Open(array_path).ReadAsArray()
            grid = grid.astype(np.float64)
            na = grid[0, 0]
            grid[grid == na] = np.nan
            return grid.flatten()

        grid = flttn(grid_path)
        gradient = flttn(gradient_path)
        carray = flttn(county_path)
        sarray = flttn(state_path)

        # Associate county and state fips with grid ids
        cdf = pd.DataFrame(OrderedDict({'grid': grid, 'county_fips': carray,
                                        'state_fips': sarray,
                                        'gradient': gradient}))
        cdf = cdf.dropna()
        cdf = cdf.astype(int)

        # Create the full county fips (state + county)
        def frmt(number):
            return '{:03d}'.format(number)
        fips['fips'] = (fips['FIPS State'].map(frmt) +
                        fips['FIPS County'].map(frmt))
        cdf['fips'] = (cdf['state_fips'].map(frmt) +
                       cdf['county_fips'].map(frmt))
        df = cdf.merge(fips, left_on='fips', right_on='fips', how='inner')
        df = df.merge(states, left_on='state_fips', right_on='STATE',
                      how='inner')
        df['place'] = df['County Name'] + ' County, ' + df['STUSAB']
        df = df[['County Name', 'STATE_NAME', 'place', 'grid', 'gradient',
                 'county_fips', 'state_fips', 'fips', 'STUSAB']]
        df.columns = ['county', 'state', 'place', 'grid', 'gradient',
                      'county_fips', 'state_fips', 'fips', 'state_abbr']

        df.to_csv(admin_path, index=False)

    def buildGrid(self):
        '''
        Use the county raster to build this.
        '''

        res_ext = self.res_ext
        array_path = self._paths("rasters/us_counties" + res_ext + ".tif")
        if not os.path.exists(array_path):
            self.buildAdmin()
        source = gdal.Open(array_path)
        geom = source.GetGeoTransform()
        proj = source.GetProjection()
        array = source.ReadAsArray()
        array = array.astype(np.float64)
        na = array[0, 0]
        mask = array.copy()
        mask[mask == na] = np.nan
        mask = mask * 0 + 1
        gradient = mask.copy()
        for i in range(gradient.shape[0]):
            for j in range(gradient.shape[1]):
                gradient[i, j] = i * j
        gradient = gradient * mask
        grid = mask.copy()
        num = grid.shape[0] * grid.shape[1]
        for i in range(gradient.shape[0]):
            for j in range(gradient.shape[1]):
                num -= 1
                grid[i, j] = num
        grid = grid * mask
        toRaster(grid, self._paths("rasters/grid" + res_ext + ".tif"),
                 geom, proj, -9999.)
        toRaster(grid, self._paths("rasters/gradient" + res_ext + ".tif"),
                 geom, proj, -9999.)

        return grid, gradient

    def buildNA(self):
        """For when there isn't any data I am printing NA across the screen. So
        all this will do is reproject an existing 'NA' raster to the specified
        resolution.
        """
        
        res = self.resolution
        res_ext = self.res_ext
        src_path = self._paths("rasters/na_banner.tif")
        out_path = self._paths("rasters/na_banner" + res_ext + ".tif")
        ds = gdal.Warp(out_path, src_path, dstSRS='EPSG:4326',
                       xRes=res, yRes=res, outputBounds=[-130, 20, -55, 50])
        del ds

    def buildSource(self):
        '''
        take a single band raster and convert it to a data array for use as a
        source. Make one of these for each resolution you might need.
        '''

        res_ext = self.rest_ext
        array_path = self._paths("data/rasters/us_counties" + res_ext + ".tif")
        if not os.path.exists(array_path):
            self.buildAdmin(self.resolution)
        data = gdal.Open(array_path)
        geom = data.GetGeoTransform()
        array = data.ReadAsArray()
        array = np.array([array])
        if len(array.shape) == 3:
            ntime, nlat, nlon = np.shape(array)
        else:
            nlat, nlon = np.shape(array)
        lons = np.arange(nlon) * geom[1] + geom[0]
        lats = np.arange(nlat) * geom[5] + geom[3]
        del data

        attributes = OrderedDict({'transform': geom,
                                  'res': (geom[1], geom[1])})

        data = xr.DataArray(data=array,
                            name=("A " + str(self.resolution) + " resolution "
                                  "grid used as a source array"),
                            coords=(('band', np.array([1])),
                                    ('y', lats),
                                    ('x', lons)),
                            attrs=attributes)
        wgs_path = self._paths("rasters/source_array" + res_ext + ".nc")
        data.to_netcdf(wgs_path)

        # We also need a source data set for Alber's projection geometry
        grid_path = self._paths("rasters/grid" + res_ext + ".tif")
        albers_path = self._paths("rasters/source_albers" + res_ext + ".tif")
        ds = gdal.Warp(albers_path, grid_path, dstSRS='EPSG:102008')
        del ds

    def getElements(self):
        '''
        I want to turn this into a class that handles all resolution dependent
        objects, but for now I'm just tossing this together for a meeting.
        '''
        # Get paths
        [grid_path, gradient_path, county_path, state_path,
         source_path, albers_path, admin_path] = self.pathRequest()

        # Read in/create objects
        states = gdal.Open(state_path).ReadAsArray()
        states[states == -9999] = np.nan
        cnty = gdal.Open(county_path).ReadAsArray()
        cnty[cnty == -9999] = np.nan
        grid = gdal.Open(grid_path).ReadAsArray()
        grid[grid == -9999] = np.nan
        mask = grid * 0 + 1
        cd = Coordinate_Dictionaries(source_path, grid)
        admin_df = pd.read_csv(admin_path)
        albers_source = gdal.Open(albers_path)
        with xr.open_dataarray(source_path) as data:
            source = data.load()

        # We actually want full state-county fips for counties
        state_counties = np.stack([states, cnty])

        # Format fips as 3-digit strings with leading zeros
        def formatFIPS(lst):
            try:
                fips1 = '{:03d}'.format(int(lst[0]))
                fips2 = '{:03d}'.format(int(lst[1]))
                fips = float(fips1 + fips2)
            except:
                fips = np.nan
            return fips

        # Get a field of full fips to use as location ids
        cnty = np.apply_along_axis(formatFIPS, 0, state_counties)

        return states, cnty, grid, mask, source, albers_source, cd, admin_df

    def pathRequest(self):

        # Set paths to each element then make sure they exist
        resolution = self.resolution
        res_str = str(round(resolution, 3))
        res_ext = '_' + res_str.replace('.', '_')
        grid_path = self._paths("rasters/grid" + res_ext + ".tif")
        gradient_path = self._paths("rasters/gradient" + res_ext + ".tif")
        county_path = self._paths("rasters/us_counties" + res_ext + ".tif")
        state_path = self._paths("rasters/us_states" + res_ext + ".tif")
        source_path = self._paths("rasters/source_array" + res_ext + ".nc")
        albers_path = self._paths("rasters/source_albers" + res_ext + ".tif")
        admin_path = self._paths("tables/admin_df" + res_ext + ".csv")
        na_path = self._paths("rasters/na_banner" + res_ext + ".tif")

        if not os.path.exists(county_path) or not os.path.exists(state_path):
            self.buildAdmin()
        if not os.path.exists(grid_path) or not os.path.exists(gradient_path):
            self.buildGrid()
        if not os.path.exists(source_path) or not os.path.exists(albers_path):
            self.buildSource()
        if not os.path.exists(admin_path):
            self.buildAdminDF()
        if not os.path.exists(na_path):
            self.buildNA()

        # Return everything at once
        path_package = [grid_path, gradient_path, county_path, state_path,
                        source_path, albers_path, admin_path]

        return path_package

    def rasterize(self, src, dst, attribute, extent, all_touch=False,
                  epsg=4326, na=-9999):
        """It seems to be unreasonably involved to do this in Python compared
        to the command line.
        """

        resolution = self.resolution

        # Open shapefile, retrieve the layer
        src_data = ogr.Open(src)
        layer = src_data.GetLayer()
        extent = layer.GetExtent()  # This is wrong

        # Create the target raster layer
        xmin, xmax, ymin, ymax = extent
        cols = int((xmax - xmin)/resolution)
        rows = int((ymax - ymin)/resolution)
        trgt = gdal.GetDriverByName('GTiff').Create(dst, cols, rows, 1,
                                                    gdal.GDT_Float32)
        trgt.SetGeoTransform((xmin, resolution, 0, ymax, 0, -resolution))

        # Add crs
        refs = osr.SpatialReference()
        refs.ImportFromEPSG(epsg)
        trgt.SetProjection(refs.ExportToWkt())

        # Set no value
        band = trgt.GetRasterBand(1)
        band.SetNoDataValue(na)

        # Set options
        if all_touch is True:
            ops = ['-at', 'ATTRIBUTE=' + attribute]
        else:
            ops = ['ATTRIBUTE=' + attribute]

        # Finally rasterize
        gdal.RasterizeLayer(trgt, [1], layer, options=ops)

        # Close target an source rasters
        del trgt
        del src_data

    def _paths(self, *paths):

        # Join paths to the root data path
        return os.path.join(self.data_path, *paths)


class Cacher:
    '''
    A simple stand in cache for storing objects in memory.
    '''
    def __init__(self, key):
        self.cache = {}
        self.key = key
    def memoize(self, function):
        def cacher(*args):
            arg = [a for a in args]
            key = json.dumps(arg)
            if key not in self.cache.keys():
                print("Generating/replacing dataset...")
                if self.cache:
                    del self.cache[list(self.cache.keys())[0]]
                self.cache.clear()
                gc.collect()
                self.cache[key] = function(*args)
            else:
                print("Returning existing dataset...")
            return self.cache[key]
        return cacher


class Coordinate_Dictionaries:
    '''
    This translates cartesian coordinates to geographic coordinates and back.
    It also provides information about the coordinate system used in the
    source data set, and methods to translate grid ids to plotly point objects
    and back.
    '''
    def __init__(self, source_path, grid):
        # Source Data Array
        self.source = xr.open_dataarray(source_path)

        # Geometry
        self.x_length = self.source.shape[2]
        self.y_length = self.source.shape[1]
        self.res = self.source.res[0]
        self.lon_min = self.source.transform[0]
        self.lat_max = self.source.transform[3]
        self.xs = range(self.x_length)
        self.ys = range(self.y_length)
        self.lons = [self.lon_min + self.res*x for x in self.xs]
        self.lats = [self.lat_max - self.res*y for y in self.ys]

        # Dictionaires with coordinates and array index positions
        self.grid = grid
        self.londict = dict(zip(self.lons, self.xs))
        self.latdict = dict(zip(self.lats, self.ys))
        self.londict_rev = {y: x for x, y in self.londict.items()}
        self.latdict_rev = {y: x for x, y in self.latdict.items()}

    def pointToGrid(self, point):
        '''
        Takes in a plotly point dictionary and outputs a grid ID
        '''
        lon = point['points'][0]['lon']
        lat = point['points'][0]['lat']
        x = self.londict[lon]
        y = self.latdict[lat]
        gridid = self.grid[y, x]
        return gridid

    # Let's say we also a list of gridids
    def gridToPoint(self, gridid):
        '''
        Takes in a grid ID and outputs a plotly point dictionary
        '''
        y, x = np.where(self.grid == gridid)
        lon = self.londict_rev[int(x[0])]
        lat = self.latdict_rev[int(y[0])]
        point = {'points': [{'lon': lon, 'lat': lat}]}
        return point



class Index_Maps:
    """This class creates a singular map as a function of some timeseries of
    rasters for use in the Ubuntu-Practice-Machine index comparison app.
    It also returns information needed for rendering.

    I think I could also incorporate the location data into this, include the
    correlation and area series functions, and simplify the logic built into
    the call backs.

    Initializing arguments:
            time_data (list)    = [[Year1, Year2], Month1, Month2,
                                   Month_Filter]
            function (string)   = 'mean_perc': 'Average Percentiles',
                                  'max': 'Maxmium Percentile',
                                  'min': 'Minimum Percentile',
                                  'mean_original': 'Mean Original Values',
                                  'omax': 'Maximum Original Value',
                                  'omin': 'Minimum Original Value',
                                  'ocv': 'Coefficient of Variation - Original'
            choice (string)     = 'noaa', 'pdsi', 'scpdsi', 'pzi', 'spi1',
                                  'spi2', 'spi3', 'spi6', 'spei1', 'spei2',
                                  'spei3', 'spei6', 'eddi1', 'eddi2', 'eddi3',
                                  'eddi6'
    """

    # Create Initial Values
    def __init__(self, data_path, choice='pdsi', choice_type='original',
                 time_data=[[2000, 2018], [1, 12], list(range(1, 13))],
                 color_class='Default', resolution=0.125, chunk=True):
        self.data_path = data_path
        self.chunk = chunk
        self.choice = choice  # This does not yet update information
        self.choice_type = choice_type
        self.color_class = color_class
        self.res_ext = "_" + str(round(resolution, 3)).replace(".", "_")
        self.reverse = False
        self.setData()
        self.setReverse()
        self.index_ranges = pd.read_csv(self._paths("tables/index_ranges.csv"))
        self.time_data = time_data  # This updates info but cannot be accessed  # <-- Help

    @property
    def time_data(self):
        return self.time_data

    @property
    def color_class(self):
        return self.color_class

    @time_data.setter
    def time_data(self, time_data):
        """To avoid reading in a new dataset when only changing dates, this
        method is separate. It sets the beginning and end dates and months used
        in all of the calculations and updates an xarray reference called
        "dataset_interval".
        """

        # Get the full data set
        dataset = self.dataset

        # Split up time information
        year1 = time_data[0][0]
        year2 = time_data[0][1]
        month1 = time_data[1][0]
        month2 = time_data[1][1]
        month_filter = time_data[2]

        # Filter the dataset by date and location
        d1 = dt.datetime(year1, month1, 1)
        d2 = dt.datetime(year2, month2, 1)
        d2 = d2 + relativedelta(months=+1) - relativedelta(days=+1)
        data = dataset.sel(time=slice(d1, d2))
        data = data.sel(time=np.in1d(data['time.month'], month_filter))

        # If this filters all of the data out, return a special "NA" data set
        if len(data.time) == 0:

            # Get the right resolution file
            res_ext = self.res_ext
            na_path = self._paths("rasters/na_banner" + res_ext + ".tif")

            # The whole data set just says "NA" using the value -9999
            today = dt.datetime.now()
            na = readRaster(na_path, 1, -9999)[0]
            na = na * 0 - 9999
            arrays = np.repeat(na[np.newaxis, :, :], 2, axis=0)
            arrays = da.from_array(arrays, chunks=100)
            days = [today - dt.timedelta(30), today]
            lats = data.coords['lat'].data
            lons = data.coords['lon'].data
            array = xr.DataArray(arrays,
                                 coords={'time': days,
                                         'lat': lats,
                                         'lon': lons},
                                 dims={'time': 2,
                                       'lat': len(lats),
                                       'lon': len(lons)})
            data = xr.Dataset({'value': array})

        self.dataset_interval = data

        # I am also setting index ranges from the full data sets
        if self.choice_type == 'percentile':
            self.data_min = 0
            self.data_max = 100
        else:
            ranges = self.index_ranges
            minimum = ranges['min'][ranges['index'] == self.choice].values[0]
            maximum = ranges['max'][ranges['index'] == self.choice].values[0]

            # For index values we want them to be centered on zero
            if self.choice not in NONINDICES:
                limits = [abs(minimum), abs(maximum)]
                self.data_max = max(limits)
                self.data_min = self.data_max * -1
            else:
                self.data_max = maximum
                self.data_min = minimum

    @color_class.setter
    def color_class(self, value):
        """This is tricky because the color can be a string pointing to
        a predefined plotly color scale, or an actual color scale, which is
        a list.
        """

        options = {'Blackbody': 'Blackbody', 'Bluered': 'Bluered',
                   'Blues': 'Blues', 'Default': 'Default', 'Earth': 'Earth',
                   'Electric': 'Electric', 'Greens': 'Greens',
                   'Greys': 'Greys', 'Hot': 'Hot', 'Jet': 'Jet',
                   'Picnic': 'Picnic', 'Portland': 'Portland',
                   'Rainbow': 'Rainbow', 'RdBu': 'RdBu', 'Viridis': 'Viridis',
                   'Reds': 'Reds',
                   'RdWhBu': [
                       [0.00, 'rgb(115,0,0)'],
                       [0.10, 'rgb(230,0,0)'],
                       [0.20, 'rgb(255,170,0)'],
                       [0.30, 'rgb(252,211,127)'],
                       [0.40, 'rgb(255, 255, 0)'],
                       [0.45, 'rgb(255, 255, 255)'],
                       [0.55, 'rgb(255, 255, 255)'],
                       [0.60, 'rgb(143, 238, 252)'],
                       [0.70, 'rgb(12,164,235)'],
                       [0.80, 'rgb(0,125,255)'],
                       [0.90, 'rgb(10,55,166)'],
                       [1.00, 'rgb(5,16,110)']],
                   'RdWhBu (Extreme Scale)':  [
                       [0.00, 'rgb(115,0,0)'],
                       [0.02, 'rgb(230,0,0)'],
                       [0.05, 'rgb(255,170,0)'],
                       [0.10, 'rgb(252,211,127)'],
                       [0.20, 'rgb(255, 255, 0)'],
                       [0.30, 'rgb(255, 255, 255)'],
                       [0.70, 'rgb(255, 255, 255)'],
                       [0.80, 'rgb(143, 238, 252)'],
                       [0.90, 'rgb(12,164,235)'],
                       [0.95, 'rgb(0,125,255)'],
                       [0.98, 'rgb(10,55,166)'],
                       [1.00, 'rgb(5,16,110)']],
                   'RdYlGnBu':  [
                       [0.00, 'rgb(124, 36, 36)'],
                       [0.25, 'rgb(255, 255, 48)'],
                       [0.5, 'rgb(76, 145, 33)'],
                       [0.85, 'rgb(0, 92, 221)'],
                       [1.00, 'rgb(0, 46, 110)']],
                   'BrGn':  [
                       [0.00, 'rgb(91, 74, 35)'],
                       [0.10, 'rgb(122, 99, 47)'],
                       [0.15, 'rgb(155, 129, 69)'],
                       [0.25, 'rgb(178, 150, 87)'],
                       [0.30, 'rgb(223,193,124)'],
                       [0.40, 'rgb(237, 208, 142)'],
                       [0.45, 'rgb(245,245,245)'],
                       [0.55, 'rgb(245,245,245)'],
                       [0.60, 'rgb(198,234,229)'],
                       [0.70, 'rgb(127,204,192)'],
                       [0.75, 'rgb(62, 165, 157)'],
                       [0.85, 'rgb(52,150,142)'],
                       [0.90, 'rgb(1,102,94)'],
                       [1.00, 'rgb(0, 73, 68)']]
                   }

        # Default color schemes
        defaults = {'percentile': options['RdWhBu'],
                    'original':  options['BrGn'],
                    'area': options['RdWhBu'],
                    'correlation_o': 'Viridis',
                    'correlation_p': 'Viridis'}

        if value == 'Default':
            scale = defaults[self.choice_type]
        else:
            scale = options[value]

        self.color_scale = scale

    def setData(self):
        """The challenge is to read as little as possible into memory without
        slowing the app down. So xarray and dask are lazy loaders, which means
        we can access the full dataset hear without worrying about that.
        """
        # There are three types of datsets
        res_ext = self.res_ext[1:]
        nc_path = self._paths("droughtindices", res_ext, "netcdfs")
        type_paths = {'original': nc_path,
                      'area': nc_path,
                      'correlation_o': nc_path,
                      'correlation_p': os.path.join(nc_path, "percentiles"),
                      'percentile': os.path.join(nc_path, "percentiles"),
                      'projected': os.path.join(nc_path, "albers")}

        # Build path and retrieve the data set
        netcdf_path = type_paths[self.choice_type]
        file_path = self._paths(netcdf_path, self.choice + ".nc")
        if self.chunk:
            dataset = xr.open_dataset(file_path, chunks=100)  # <------------------ Best chunk size/shape?
        else:
            dataset = xr.open_dataset(file_path)

        dataset["value"] = dataset["value"]

        # Set this as an attribute for easy retrieval
        self.dataset = dataset

    def setMask(self, location, crdict):
        """Take a location object and the coordinate dictionary to create an
        xarray for masking the dask datasets without pulling into memory.

        location = location from Location_Builder or 1d array
        crdict = coordinate dictionary
        """

        # Get x, y coordinates from location
        flag, y, x, label, idx = location
        mask = crdict.grid.copy()

        # Create mask array
        if flag != 'all':
            y = json.loads(y)
            x = json.loads(x)
            gridids = crdict.grid[y, x]
            mask[~np.isin(crdict.grid, gridids)] = np.nan
            mask = mask * 0 + 1
        else:
            mask = mask * 0 + 1

        # Set up grid info from coordinate dictionary
        geom = crdict.source.transform
        nlat, nlon = mask.shape
        lons = np.arange(nlon) * geom[1] + geom[0]
        lats = np.arange(nlat) * geom[5] + geom[3]

        # Create mask xarray
        xmask = xr.DataArray(mask, coords={"lat": lats, "lon": lons},
                             dims={"lat": len(lats), "lon": len(lons)})
        self.mask = xmask

    def setReverse(self):
        """Set an attribute to reverse the colorscale if needed for the
        indicator."""

        choice = self.choice
        reversals = ['eddi', 'tmin', 'tmax', 'tmean', 'tdmean', 'vpdmax',
                     'vpdmin', 'vpdmean']
        if any(r in choice for r in reversals):
            self.reverse = True
        else:
            self.reverse = False

    def getTime(self):

        # Now read in the corrollary albers data set
        dates = pd.DatetimeIndex(self.dataset_interval.time[:].values)
        year1 = min(dates.year)
        year2 = max(dates.year)
        month1 = dates.month[0]
        month2 = dates.month[-1]
        month_filter = list(pd.unique(dates.month))
        time_data = [[year1, year2], [month1, month2], month_filter]

        return time_data

    def getMean(self):
        return self.dataset_interval.mean('time', skipna=True).value.data

    def getMin(self):
        return self.dataset_interval.min('time', skipna=True).value.data

    def getMax(self):
        return self.dataset_interval.max('time', skipna=True).value.data

    def getSeries(self, location, crdict):
        """This uses the mask to get a monthly time series of values at a
        specified location.
        """

        # Get filtered dataset
        dataset = self.dataset_interval

        # Get the location coordinates
        flag, y, x, label, idx = location

        # Filter if needed and generate timeseries
        if flag == 'all':
            timeseries = dataset.mean(dim=('lat', 'lon'), skipna=True)
            timeseries = timeseries.value.values
        else:
            if flag == 'grid':
                y = json.loads(y)
                x = json.loads(x)
                timeseries = dataset.value[:, y, x].values
            else:
                mdata = dataset.where(self.mask == 1)
                timeseries = mdata.mean(dim=('lat', 'lon'), skipna=True)
                timeseries = timeseries.value.values

        return timeseries

    def getCorr(self, location, crdict):
        """Create a field of pearson's correlation coefficients with any one
        selection.
        """

        ts = self.getSeries(location, crdict)
        arrays = self.dataset_interval.value.values
        one_field = correlationField(ts, arrays)

        return one_field

    def getArea(self, data_path, crdict, resolution):
        """This will take in a time series of arrays and a drought severity
        category and mask out all cells with values above or below the category
        thresholds. If inclusive is 'True' it will only mask out all cells that
        fall above the chosen category.

        This should be cached with functools, and will be calculated only once
        in the app if the Index_Maps object is properly cached with flask
        caching.

        For now this requires original values, percentiles even out too
        quickly.
        """

        # Specify choice in case it needs to be inverse for eddi
        choice = self.choice
        data = self.dataset_interval

        # Now read in the corrollary albers data set
        time_data = self.getTime()        
        choice_type = 'projected'
        proj_data = Index_Maps(data_path, choice, choice_type, time_data, 'RdWhBu',
                               resolution=resolution, chunk=True)
        proj_sample = proj_data.dataset

        # Filter data by the mask (should be set already)
        arrays = data.value.values
        mask = self.mask.values
        masked_arrays = arrays * mask
        albers_mask = wgsToAlbers(masked_arrays, crdict, proj_sample)
        arrays = proj_data.dataset_interval.where(albers_mask == 1).value

        # Flip if this is EDDI
        if 'eddi' in choice:
            arrays = arrays*-1

        # Drought Categories
        drought_cats = {'sp': {0: [-0.5, -0.8],
                               1: [-0.8, -1.3],
                               2: [-1.3, -1.5],
                               3: [-1.5, -2.0],
                               4: [-2.0, -999]},
                        'eddi': {0: [-0.5, -0.8],
                                 1: [-0.8, -1.3],
                                 2: [-1.3, -1.5],
                                 3: [-1.5, -2.0],
                                 4: [-2.0, -999]},
                        'pdsi': {0: [-1.0, -2.0],
                                 1: [-2.0, -3.0],
                                 2: [-3.0, -4.0],
                                 3: [-4.0, -5.0],
                                 4: [-5.0, -999]},
                        'leri': {0: [-0.5, -0.8],
                                 1: [-0.8, -1.3],
                                 2: [-1.3, -1.5],
                                 3: [-1.5, -2.0],
                                 4: [-2.0, -999]}}

        # Choose a set of categories
        cat_key = [key for key in drought_cats.keys() if key in choice][0]
        cats = drought_cats[cat_key]

        def catFilter(arrays, d, inclusive=False):
            """There is some question about the Drought Severity Coverage Index.
            The NDMC does not use inclusive drought categories though NIDIS
            appeared to in the "Historical Character of US Northern Great
            Plains Drought" study. In an effort to match NIDIS' sample chart,
            we are using the inclusive method for now. It would be fine either
            way as long as the index is compared to other values with the same
            calculation, but we should really defer to NDMC. We could also add
            an option to display inclusive vs non-inclusive drought severity
            coverages.
            """

            totals = arrays.where(~np.isnan(arrays)).count(dim=('lat', 'lon'))
            if inclusive:
                counts = arrays.where(arrays < d[0]).count(dim=('lat', 'lon'))
            else:
                counts = arrays.where((arrays < d[0]) &
                                      (arrays >= d[1])).count(
                                          dim=('lat', 'lon'))
            ratios = counts / totals
            pcts = ratios.compute().data * 100
            return pcts

        # Calculate non-inclusive percentages # <------------------------------ parallelizing with delayed speeds it up but takes just a bit too much memory for the virtual machine to handle the full time series
        pnincs = [dask.delayed(catFilter)(arrays, cats[i]) for i in range(5)]
        pnincs = np.array(dask.compute(*pnincs))
        # pnincs =  np.array([catFilter(arrays, cats[i]) for i in range(5)])

        # Use the noninclusive percentages to create the inclusive percentages
        pincs = [np.sum(pnincs[i:], axis=0) for i in range(len(pnincs))]

        # Also use the noninclusive arrays to get the DSCI
        pnacc = np.array([pnincs[i]*(i+1) for i in range(5)])
        DSCI = list(np.nansum(pnacc, axis=0))

        # To store these in a div they need to work with json
        pincs = [list(a) for a in pincs]
        pnincs = [list(p) for p in pnincs]

        # Return the list of five layers
        return pincs, pnincs, DSCI

    def getFunction(self, function):
        '''
        To choose which function to return using a string from a dropdown app.
        '''
        functions = {"omean": self.getMean,
                     "omin": self.getMin,
                     "omax": self.getMax,
                     "pmean": self.getMean,
                     "pmin": self.getMin,
                     "pmax": self.getMax,
                     "oarea": self.getMean,  # <------------------------------- Note that this is returning the mean for now (skipped,  performed in app for now)
                     "ocorr": self.getMean,
                     "pcorr": self.getMean}
        function = functions[function]

        return function()

    def _paths(self, *paths):

        # Join paths to the root data path
        return os.path.join(self.data_path, *paths)


class Location_Builder:
    '''
    This takes a location selection determined to be the triggering choice,
    decides what type of location it is, and builds the appropriate location
    list object needed further down the line. To do so, it holds county,
    state, grid, and other administrative information.
    '''

    def __init__(self, trig_id, trig_val, coordinate_dictionary, admin_df,
                 state_array, county_array):
        self.trig_id = trig_id
        self.trig_val = trig_val
        self.cd = coordinate_dictionary
        self.admin_df = admin_df
        self.states_df = admin_df[['state', 'state_abbr',
                                   'state_fips']].drop_duplicates().dropna()
        self.state_array = state_array
        self.county_array = county_array

    def chooseRecent(self):
        '''
        Check the location for various features to determine what type of
        selection it came from. Return a list with some useful elements.
        Possible triggering elements:
            'map_1.clickData',
            'map_2.clickData',
            'map_1.selectedData',
            'map_2.selectedData',
            'county_1.value',
            'county_2.value',
            'state_1.value',
            'state_2.value'
        '''
        trig_id = self.trig_id
        trig_val = self.trig_val
        admin_df = self.admin_df
        states_df = self.states_df
        cd = self.cd
        county_array = self.county_array
        state_array = self.state_array

        # print("Location Picker Trigger: " + str(trig_id))
        # print("Location Picker Value: " + str(trig_val))

        # 1: Selection is a county selection
        if 'county' in trig_id:
            county = admin_df['place'][admin_df.fips == trig_val].unique()[0]
            y, x = np.where(county_array == trig_val)
            # crds =
            location = ['county', str(list(y)), str(list(x)), county]

        # 2: Selection is a single grid IDs
        elif 'clickData' in trig_id:
            lon = trig_val['points'][0]['lon']
            lat = trig_val['points'][0]['lat']
            crds = [lat, lon]
            x = cd.londict[lon]
            y = cd.latdict[lat]
            gridid = cd.grid[y, x]
            counties = admin_df['place'][admin_df.grid == gridid]
            county = counties.unique()
            label = county[0] + ' (Grid ' + str(int(gridid)) + ')'
            location = ['grid', str(y), str(x), label]

        # 3: Selection is a set of grid IDs
        elif 'selectedData' in trig_id:
            if trig_val is not None:
                selections = trig_val['points']
                lats = [d['lat'] for d in selections]
                lons = [d['lon'] for d in selections]
                crds = [max(lats), min(lons), min(lats), max(lons)]
                y = list([cd.latdict[d['lat']] for d in selections])
                x = list([cd.londict[d['lon']] for d in selections])
                counties = np.array([d['text'][:d['text'].index('<')] for
                                     d in selections])
                cnty_lst = list(np.unique(counties))
                local_df = admin_df[admin_df['place'].isin(cnty_lst)]

                # Use gradient to print NW and SE most counties as a range
                NW = local_df['place'][
                    local_df['gradient'] == min(local_df['gradient'])].item()
                SE = local_df['place'][
                    local_df['gradient'] == max(local_df['gradient'])].item()
                label = NW + " to " + SE
                location = ['grids', str(y), str(x), label]
            else:
                raise PreventUpdate

        # 2: location is a list of states
        elif 'update' in trig_id:
            # Selection is the default 'all'
            if type(trig_val) is str:
                location = ['all', 'y', 'x', 'Contiguous United States']

            # Empty list, default to CONUS
            elif len(trig_val) == 0:
                location = ['all', 'y', 'x', 'Contiguous United States']

            # A selection of 'all' within a list
            elif len(trig_val) == 1 and trig_val[0] == 'all':
                location = ['all', 'y', 'x', 'Contiguous United States']

            # Single or multiple, not 'all' or empty, state or list of states
            elif len(trig_val) >= 1:
                # Return the mask, a flag, and the state names
                state = list(states_df['state_abbr'][
                    states_df['state_fips'].isin(trig_val)])

                if len(state) < 4:  # Spell out full state name in title
                    state = [states_df['state'][
                        states_df['state_abbr'] == s].item() for s in state]
                states = ", ".join(state)
                y, x = np.where(np.isin(state_array, trig_val))

                # And return the location information
                location = ['state', str(list(y)), str(list(x)), states]

        # 3: location is the basename of a shapefile saved as temp.shp
        elif 'shape' in trig_id:
            # We don't have the x,y values just yet
            try:
                shp = gdal.Open('data/shapefiles/temp/temp.tif').ReadAsArray()
                shp[shp == -9999] = np.nan
                y, x = np.where(~np.isnan(shp))
                # crds =
                location = ['shape', str(list(y)), str(list(x)), trig_val]
            except:
                location = ['all', 'y', 'x', 'Contiguous United States']

        # 4: A reset button was clicked
        elif 'reset_map' in trig_id:
            location = ['all', 'y', 'x', 'Contiguous United States']

        # I am not done creating coord objects yet
        try:
            crds
        except:
            crds = "Coordinates not available yet"

        return location, crds
