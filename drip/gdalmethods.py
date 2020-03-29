# -*- coding: utf-8 -*-
"""
GDAL Methods.

Created on Sat Mar 28 15:28:32 2020

@author: travis
"""

import geopandas as gpd
import numpy as np
import os
import rasterio
import requests
import shutil
import subprocess as sp
import sys
import zipfile

from multiprocessing import Pool
from osgeo import gdal, ogr, osr
from shapely.geometry import Point
from tqdm import tqdm

gdal.UseExceptions()
gdal.PushErrorHandler('CPLQuietErrorHandler')
os.environ['GDAL_PAM_ENABLED'] = 'NO'

# CONSTANTS
GDAL_TYPES = {"GDT_Byte": "Eight bit unsigned integer",
              "GDT_CFloat32": "Complex Float32",
              "GDT_CFloat64": "Complex Float64",
              "GDT_CInt16": "Complex Int16",
              "GDT_CInt32": "Complex Int32",
              "GDT_Float32": "Thirty two bit floating point",
              "GDT_Float64": "Sixty four bit floating point",
              "GDT_Int16": "Sixteen bit signed integer",
              "GDT_Int32": "Thirty two bit signed integer",
              "GDT_UInt16": "Sixteen bit unsigned integer",
              "GDT_UInt32": "Thirty two bit unsigned integer",
              "GDT_Unknown": "Unknown or unspecified type"}

GDAL_TYPEMAP = {"byte": gdal.GDT_Byte,
                "cfloat32": gdal.GDT_CFloat32,
                "cfloat64": gdal.GDT_CFloat64,
                "cint16": gdal.GDT_CInt16,
                "cint32": gdal.GDT_CInt32,
                "float32": gdal.GDT_Float32,
                "float64": gdal.GDT_Float64,
                "int16": gdal.GDT_Int16,
                "int32": gdal.GDT_Int32,
                "uint16": gdal.GDT_UInt16,
                "uint32": gdal.GDT_UInt32,
                "unknown": gdal.GDT_Unknown}

# FUNCTIONS
def gdal_options(module="translate", **kwargs):
    """Capture any availabe option for gdal functions. Print available options
    if one was mispelled. Alternately, run with no **kwargs for a list of
    available options and descriptions.

    Examples:
        gdal_options("warp")
        ops = gdal_options("warp", dstSRS="epsg:4326", xRes=.25, yRes=.25)
    """

    # Standardize case
    module = module.lower().replace("gdal", "").replace("_", "")

    # All available options
    options = [m for m in gdal.__dict__ if "Options" in m and "_" not in m and "GDAL" not in m]

    # Get the module option method associated with module
    modules = []
    for o in options:
        o = o.replace("GDAL", "").replace("Options", "").replace("_", "")
        o = o.lower()
        modules.append(o)

    # Create options dictionary
    option_dict = dict(zip(modules, options))

    # Get the requested option method
    try:
        option = option_dict[module]
        method = getattr(gdal, option)
    except KeyError:
        print("GDAL options for " + module + " are not available.")
        docs = "\n   ".join(modules)
        print("Available methods with options:\n   " + docs)
        return

    # Get the docs for that method
    docs = "\n".join(method.__doc__.split("\n")[1:])

    # Return the appropriate options object
    try:
        assert kwargs
        ops = method(**kwargs)
        return ops
    except AssertionError:
        print(docs)
        return
    except TypeError as terror:
        te = terror.args[0]
        missing = te.split()[-1]
        print("The " + missing + " option is not available or formatted "
              "incorrectly.")
        print(docs)
        return



def dlzip(url, path):
    """Download, unzip, and remove zip file from url."""
    if not os.path.exists(path):
        r = requests.get(url)
        with open(path, 'wb') as file:
            file.write(r.content)
        with zipfile.ZipFile(path, 'r') as zip_ref:
            save_dir = os.path.splitext(path.replace(".zip", ""))[0]
            zip_ref.extractall(save_dir)
        os.remove(path)


def gdal_progress(complete, message, unknown):
    """A progress callback that recreates the gdal printouts."""

    # We don't need the message or unknown objects
    del message, unknown

    # Complete is a number between 0 and 1
    percent = int(complete * 100)

    # Between numeric printouts we need three dots
    dots = [[str(i) + d for d in ["2", "5", "8"]] for i in range(10)]
    dots = [int(l) for sl in dots for l in sl]

    # If divisible by ten, print the number
    if percent % 10 == 0 and percent != 0:
        print("{}".format(percent), end="")

    # If one of three numbers between multiples of 10, print a dot
    elif percent in dots:
        print(".", end="")

    return 1


def rasterize(src, dst, attribute, t_srs=None, transform=None, height=None,
              width=None, template_path=None, navalue=-9999, all_touch=False,
              dtype=gdal.GDT_Float32, overwrite=False):
    """
    Use GDAL RasterizeLayer to rasterize a shapefile stored on disk and write
    outputs to a file.

    Parameters
    ----------
    src : str
        File path for the source file to rasterize.
    dst : str
        Destination path for the output raster.
    attribute : str
        Attribute name being rasterized.
    t_srs : int
        EPSG Code associated with target coordinate reference system.
    transform : list | tuple | array
        Geometric affine transformation:
            (x-min, x-resolution, x-rotation, y-max, y-rotation, y-resoltution)
    height : int
        Number of y-axis grid cells.
    width : int
        Number of x-axis grid cells.
    template_path : str
        The path to a raster with target geometries.
    na : int | float
        The value to assign to non-value grid cells. (defaults to -99999)
    all_touch : boolean
        Wether or not to associate vector values with all intersecting grid
        cells. (defaults to False)
    dtype : str | gdal object
        GDAL data type. Can be a string or a gdal type object (e.g.
        gdal.GDT_Float32, "GDT_Float32", "float32"). Available GDAL data types
        and descriptions can be found in the GDAL_TYPES dictionary.
    overwrite : boolean

    Returns
    -------
    None

    # Things to do:
        1) Catch exceptions
        2) Progress callback
        3) Use more than just EPSG (doesn't always work, also accept proj4)
    """

    # Overwrite existing file
    if os.path.exists(dst):
        if overwrite:
            if os.path.isfile(dst):
                os.remove(dst)
            else:
                shutil.rmtree(dst)
        else:
            print(dst + " exists, use overwrite=True to replace this file.")
            return

    # Open shapefile, retrieve the layer
    src_data = ogr.Open(src)
    layer = src_data.GetLayer()

    # Create a spatial reference object
    refs = osr.SpatialReference()

    # If a template is provided
    if template_path:
        temp = gdal.Open(template_path)
        transform = temp.GetGeoTransform()
        width = temp.RasterXSize
        height = temp.RasterYSize
        t_srs = temp.GetProjection()
        refs.ImportFromWkt(t_srs)
    else:
        try:
            refs.ImportFromEPSG(t_srs)
        except TypeError:
            refs.ImportFromProj4(t_srs)

    # Use transform to derive coordinates and dimensions
    xmin, xres, xrot, ymax, yrot, yres = transform
    xs = [xmin + xres * i for i in range(width)]
    ys = [ymax + yres * i for i in range(height)]
    nx = len(xs)
    ny = len(ys)

    # Specifying data types shouldn't be so difficult
    if isinstance(dtype, str):
        dtype = dtype.lower().replace("gdt_", "")
        try:
            dtype = GDAL_TYPEMAP[dtype]
        except KeyError:
            print("\n'" + dtype + "' is not an available data type. "
                  "Choose a value from this list:")
            print(str(list(GDAL_TYPEMAP.keys())))

    # Create the target raster layer
    driver = gdal.GetDriverByName("GTiff")
    trgt = driver.Create(dst, nx, ny, 1, dtype)
    trgt.SetGeoTransform((xmin, xres, xrot, ymax, yrot, yres))
    trgt.SetProjection(refs.ExportToWkt())

    # Set no value
    band = trgt.GetRasterBand(1)
    band.SetNoDataValue(navalue)

    # Set options
    if all_touch is True:
        ops = ["-at", "ATTRIBUTE=" + attribute]
    else:
        ops = ["ATTRIBUTE=" + attribute]

    # Finally rasterize
    gdal.RasterizeLayer(trgt, [1], layer, options=ops, callback=gdal_progress)

    # Close target an source rasters
    del trgt
    del src_data


def read_raster(rasterpath, band=1, navalue=-9999):
    """Converts a raster file on disk into a numpy array along with
    spatial features needed to write results to a raster file.

    Parameters
    ----------
    rasterpath : str
        Path to a raster file.
    band : int
        The band number desired.
    navalue : int | float
        The number used for non-values in the raster data set

    Returns
    -------
        tuple:
             raster values : numpy.ndarray
             affine transformation : tuple
                 (top left x coordinate, x resolution, row rotation,
                  top left y coordinate, column rotation, y resolution)),
            coordinate reference system : str
                 Well-Known Text format
    """

    # Open raster file and read in parts necessary for rewriting
    raster = gdal.Open(rasterpath)
    geometry = raster.GetGeoTransform()
    arrayref = raster.GetProjection()
    array = np.array(raster.GetRasterBand(band).ReadAsArray())
    raster = None

    # This helped for some old use-case, but might not be necessary
    array = array.astype(float)
    if np.nanmin(array) < navalue:
        navalue = np.nanmin(array)
    array[array == navalue] = np.nan

    return(array, geometry, arrayref)


def reproject_polygon(src, dst, t_srs):
    """Reproject a shapefile of polygons and write results to disk. Recreates
    this GDAL command:
        ogr2ogr -s_srs <source_projection> -t_srs <target_projection> dst src

    Parameters
    ----------
    src : str
        Path to source shapefile.
    dst : str
        Path to target file.
    tproj (int | str):
        Target coordinate projection system as an epsg code or proj4 string.
        Sometimes EPSG codes aren't available to GDAL installations, but
        they're easier to use when they are so this will try both.

    Note
    ----
    This only handles ESRI Shapefiles at the moment, but can be written to
    handle any available driver.
    """

    # Create target directory
    save_path = os.path.dirname(dst)
    os.makedirs(save_path, exist_ok=True)

    # Create Shapefile driver
    driver = ogr.GetDriverByName("ESRI Shapefile")

    # Source reference information
    src_file = driver.Open(src)
    src_layer = src_file.GetLayer()
    src_srs = src_layer.GetSpatialRef()
    src_defn = src_layer.GetLayerDefn()

    # Target reference information
    trgt_srs = osr.SpatialReference()
    try:
        trgt_srs.ImportFromEPSG(t_srs)
    except Exception:
        trgt_srs.ImportFromProj4(t_srs)

    # The transformation equation
    transform = osr.CoordinateTransformation(src_srs, trgt_srs)

    # Target file and layer
    if os.path.exists(dst):
        driver.DeleteDataSource(dst)
    trgt_file = driver.CreateDataSource(dst)
    trgt_layer = trgt_file.CreateLayer('', trgt_srs, ogr.wkbMultiPolygon)

    # Add Fields
    for i in range(0, src_defn.GetFieldCount()):
        defn = src_defn.GetFieldDefn(i)
        trgt_layer.CreateField(defn)

    # Get the target layer definition
    trgt_defn = trgt_layer.GetLayerDefn()

    # You have to reproject each feature
    src_feature = src_layer.GetNextFeature()
    while src_feature:
        # Get geometry
        geom = src_feature.GetGeometryRef()

        # Reproject geometry
        geom.Transform(transform)

        # Create target feature
        trgt_feature = ogr.Feature(trgt_defn)
        trgt_feature.SetGeometry(geom)
        for i in range(0, trgt_defn.GetFieldCount()):
            trgt_feature.SetField(trgt_defn.GetFieldDefn(i).GetNameRef(),
                                  src_feature.GetField(i))

        # Add feature to target file
        trgt_layer.CreateFeature(trgt_feature)

        # Close current feature
        trgt_feature = None

        # Get the next feature
        src_feature = src_layer.GetNextFeature()

    # Close both shapefiles
    src_file = None
    trgt_file = None


def reproject_point(src, dst, tproj):
    """Reproject a shapefile of points and write results to disk. Recreates
    this GDAL command:

        ogr2ogr -s_srs <source_projection> -t_srs <target_projection> dst src

    Parameters
    ----------
    src : str
        Path to source shapefile.
    dst : str
        Path to target file.
    tproj (int | str):
        Target coordinate projection system as an epsg code or proj4 string.
        Sometimes EPSG codes aren't available to GDAL installations, but
        they're easier to use when they are so this will try both.

    Note
    ----
    This only handles ESRI Shapefiles at the moment, but can be written to
    handle any available driver.
    """

    # Create Shapefile driver
    driver = ogr.GetDriverByName("ESRI Shapefile")

    # Source reference information
    src_file = driver.Open(src)
    src_layer = src_file.GetLayer()
    src_srs = src_layer.GetSpatialRef()
    src_defn = src_layer.GetLayerDefn()

    # Target reference information
    trgt_srs = osr.SpatialReference()
    try:
        trgt_srs.ImportFromEPSG(tproj)
    except Exception:
        trgt_srs.ImportFromProj4(tproj)

    # The transformation equation
    transform = osr.CoordinateTransformation(src_srs, trgt_srs)

    # Target file and layer
    if os.path.exists(dst):
        driver.DeleteDataSource(dst)
    trgt_file = driver.CreateDataSource(dst)
    trgt_layer = trgt_file.CreateLayer('', trgt_srs, ogr.wkbPoint)

    # Add Fields
    for i in range(0, src_defn.GetFieldCount()):
        defn = src_defn.GetFieldDefn(i)
        trgt_layer.CreateField(defn)

    # Get the target layer definition
    trgt_defn = trgt_layer.GetLayerDefn()

    # You have to reproject each feature
    src_feature = src_layer.GetNextFeature()
    while src_feature:
        # Get geometry
        geom = src_feature.GetGeometryRef()

        # Reproject geometry
        geom.Transform(transform)

        # Create target feature
        trgt_feature = ogr.Feature(trgt_defn)
        trgt_feature.SetGeometry(geom)
        for i in range(0, trgt_defn.GetFieldCount()):
            trgt_feature.SetField(trgt_defn.GetFieldDefn(i).GetNameRef(),
                                  src_feature.GetField(i))

        # Add feature to target file
        trgt_layer.CreateFeature(trgt_feature)

        # Close current feature
        trgt_feature = None

        # Get the next feature
        src_feature = src_layer.GetNextFeature()

    # Close both shapefiles
    src_file = None
    trgt_file = None


def split_extent(raster_file, n=100):
    """Split a raster files extent into n extent pieces."""

    # Get raster geometry
    rstr = rasterio.open(raster_file)
    geom = rstr.get_transform()
    ny = rstr.height
    nx = rstr.width
    xs = [geom[0] + geom[1] * i for i in range(nx)]
    ys = [geom[3] + geom[-1] * i for i in range(ny)]

    # Get number of chunks along each axis
    nc = np.sqrt(n)

    # Split coordinates into 10 pieces along both axes...
    xchunks = np.array(np.array_split(xs, nc))
    ychunks = np.array(np.array_split(ys, nc))

    # Get min/max of each coordinate chunk...
    sides = lambda x: [min(x), max(x)]
    xmap = map(sides, xchunks)
    ymap = map(sides, ychunks)
    xext = np.array([v for v in xmap])
    yext = np.array([v for v in ymap])

    # Combine these in this order [xmin, ymin, xmax, ymax]....
    extents = []
    for xex in xext:
        for yex in yext:
            extents.append([xex[0], yex[0], xex[1], yex[1]])

    return extents


def tile_raster(raster_file, out_folder, ntiles, ncpu):
    """ Take a raster and write n tiles from it.

    Parameters
    ----------
    raster_file : str
        Path to a GeoTiff
    out_folder : str
        Path to a folder in which to store tiles. Will create if not present.
    ntiles : int
        Number of tiles to write.
    ncpu : int
        Number of cpus to use for processing.

    Returns
    -------
    None.
    """

    # Create the output folder
    if not out_folder:
        base_name = os.path.splitext(raster_file)[0]
        out_folder = "_".join([base_name, "tiles"])
    os.makedirs(out_folder, exist_ok=True)

    # Get all of the extents needed to make n tiles
    extents = split_extent(raster_file, n=ntiles)

    # Wrap arguments into one object
    raster_files = np.repeat(raster_file, len(extents))
    chunknumbers = [i for i in range(len(extents))]
    out_folders = np.repeat(out_folder, len(extents))
    args = list(zip(extents, raster_files, chunknumbers, out_folders))

    # Run each
    with Pool(ncpu) as pool:
        tfiles = []
        for tfile in tqdm(pool.imap(tile_single, args), total=len(extents),
                          position=0, file=sys.stdout):
            tfiles.append(tfile)

    return tfiles


def tile_single(arg):
    """Use gdal to cut a raster into a smaller pieces.

    Note:
        This is made for tile_raster and is not intuitive as a standalone.
        Add in a check to make sure each output file is good. Moving to
        a class method soon.
    """

    # Separate arguments
    extent = arg[0]
    rfile = arg[1]
    chunk = arg[2]
    outfolder = arg[3]

    # Get everything in order
    extent = [str(e) for e in extent]
    chunk = "{:02d}".format(chunk)
    outbase = os.path.basename(rfile).split(".")[0]
    outfile = os.path.join(outfolder, outbase + "_" + chunk + ".tif")

    # Let's not overwrite - use the warp function from below here instead
    if not os.path.exists(outfile):
        sp.call(["gdalwarp",
                 "-q",
                 "-te", extent[0], extent[1], extent[2], extent[3],
                 rfile,
                 outfile],
                stdout=sp.PIPE, stderr=sp.PIPE)

    return outfile


def to_geo(data_frame, loncol="lon", latcol="lat", epsg=4326):
    """Convert a Pandas DataFrame object to a GeoPandas GeoDataFrame object.

    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        A pandas data frame with latitude and longitude coordinates.
    loncol : str
        The name of the longitude column.
    latcol : str
        The name of the latitude column.
    epsg : int
        EPSG code associated with the Coordinate Reference System.

    Returns
    -------
    geopandas.geodataframe.GeoDataFrame
        A GeoPandas GeoDataFrame object.
    """

    crs = {"init": "epsg:{}".format(epsg)}
    to_point = lambda x: Point((x[loncol], x[latcol]))
    data_frame["geometry"] = data_frame.apply(to_point, axis=1)
    gdf = gpd.GeoDataFrame(data_frame, geometry="geometry", crs=crs)

    return gdf


def to_raster(array, savepath, crs=None, geometry=None, template=None,
              dtype=gdal.GDT_Float32, compress=None, navalue=-9999):
    """Takes in a numpy array and writes data to a GeoTiff.

    Parameters
    ----------
    array : numpy.ndarray
        Numpy array to write to raster file.
    savepath : str
        Path to the target raster file.
    crs : str
        Coordinate reference system in Well-Known Text format.
    geometry : tuple
        Affine transformation information in this order:
            (top left x coordinate, x resolution, row rotation,
            top left y coordinate, column rotation, y resolution)
    template : str
        Path to a raster file with desired target raster geometry, crs, and na
        value. This will overwrite other arguments provided for these
        parameters.
    dtype : str | gdal object
        GDAL data type. Can be a string or a gdal type object (e.g.
        gdal.GDT_Float32, "GDT_Float32", "float32"). Available GDAL data types
        and descriptions can be found in the GDAL_TYPES dictionary.
    compress : str
        A compression technique. Available options are "DEFLATE", "JPEG",
        "LZW"
    navalue : int | float
        The number used for non-values in the raster data set. Defaults to
        -9999.
    """

    # Retrieve needed raster elements
    xpixels = array.shape[1]
    ypixels = array.shape[0]

    # This helps sometimes
    savepath = savepath.encode('utf-8')

    # Specifying data types shouldn't be so difficult
    if isinstance(dtype, str):
        dtype = dtype.lower().replace("gdt_", "")
        try:
            dtype = GDAL_TYPEMAP[dtype]
        except KeyError:
            print("\n'" + dtype + "' is not an available data type. "
                  "Choose a value from this list:")
            print(str(list(GDAL_TYPEMAP.keys())))

    # Create file
    driver = gdal.GetDriverByName("GTiff")

    # Get options here - not built out yet
    if compress:
        creation_ops = ["compress=LZW"]
        image = driver.Create(savepath, xpixels, ypixels, 1, dtype,
                              options=creation_ops)
    else:
        image = driver.Create(savepath, xpixels, ypixels, 1, dtype)

    # Use a template file to extract affine transformation, crs, and na value
    if template:
        template_file = gdal.Open(template)
        geometry = template_file.GetGeoTransform()
        crs = template_file.GetProjection()

    # Write raster data and attributes to file
    image.SetGeoTransform(geometry)
    image.SetProjection(crs)
    image.GetRasterBand(1).WriteArray(array)
    image.GetRasterBand(1).SetNoDataValue(navalue)


def translate(src, dst, overwrite=False, compress=None, progress=True,
              **kwargs):
    """
    Translate a raster dataset from one format to another.

    Parameters
    ----------
    src : str
        Path to source raster file or containing folder for ESRI Grids.
    dst : str
        Path to target raster file.
    overwrite : boolean
    compress : str
        A compression technique. Available options are "DEFLATE", "JPEG",
        "LZW"
    **kwargs
        Any available key word arguments for gdal_translate. Available options
        and descriptions can be found using gdal_options("translate").

    Returns
    -------
    None.

    Notes
    -----

    The progress bar needs work.

    """

    # Create progress callback - these behave differently by module
    def translate_progress(percent, message, unknown):
        """A progress callback that recreates the gdal printouts."""

        # We don't need the message or unknown objects
        del message, unknown

        # Between numeric printouts we need three dots
        dots = [[str(i) + d for d in ["2", "5", "8"]] for i in range(10)]
        dots = [int(l) for sl in dots for l in sl]

        # If divisible by ten, print the number
        if percent % 10 == 0 and percent != 0:
            print("{}".format(percent), end="")

        # If one of three numbers between multiples of 10, print a dot
        elif percent in dots:
            print(".", end="")

        return 1

    # Expand user paths
    src = os.path.expanduser(src)
    dst = os.path.expanduser(dst)

    # Overwrite existing file
    if os.path.exists(dst):
        if overwrite:
            if os.path.isfile(dst):
                os.remove(dst)
            else:
                shutil.rmtree(dst)
        else:
            print(dst + " exists, use overwrite=True to replace this file.")
            return

    # What is the best way to deal with ESRI grids?
    if not os.path.isfile(src):
        files = os.listdir(src)
        if not "hdr.adf" in files:
            raise FileNotFoundError("Cannot find a translatable file.")

    # Add in key word arguments
    if progress:
        print("Processing " + dst + " :")
        kwargs["callback"] = gdal_progress

    # Compress
    if compress:
        kwargs["creationOptions"] = ["COMPRESS=" + compress]

    # Create an options object
    ops = gdal_options("translate", **kwargs)

    # We need to open the src data set
    src = gdal.Open(src)

    # Call
    ds = gdal.Translate(destName=dst, srcDS=src, options=ops)
    del ds


def warp(src, dst, dtype="Float32", template=None, overwrite=False,
         compress=None, progress=True, **kwargs):
    """
    Warp a raster to a new geometry.

    Parameters
    ----------
    src : str
        Path to source raster file.
    dst : str
        Path to target raster file.
    dtype : str | gdal object
        GDAL data type. Can be a string or a gdal type object (e.g.
        gdal.GDT_Float32, "GDT_Float32", "float32"). Available GDAL data types
        and descriptions can be found in the GDAL_TYPES dictionary.
    template : str
        Path to a raster file with desired target raster geometry, crs,
        resolution, and extent values. This will overwrite other arguments
        provided for these parameters. Template-derived arguments will
        overwrite **kwargs.
    overwrite : boolean
    compress : str
        A compression technique. Available options are "DEFLATE", "JPEG",
        "LZW"
    **kwargs
        Any available key word arguments for gdalwarp. Available options
        and descriptions can be found using gdal_options("warp").

    Returns
    -------
    None.

    Example:
        warp(src="/Users/twillia2/Box/WETO 1.2/data/rasters/agcounty_product.tif",
             dst="/Users/twillia2/Box/WETO 1.2/data/rasters/test.tif",
             template="/Users/twillia2/Box/WETO 1.2/data/rasters/albers/acre/cost_codes_ac.tif",
             dstSRS="epsg:102008")
    """

    # Create progress callback - these behave differently by module
    def warp_progress(percent, message, unknown):
        """A progress callback that recreates the gdal printouts."""

        # We don't need the message or unknown objects
        del message, unknown

        # Between numeric printouts we need three dots
        dots = [[str(i) + d for d in ["2", "5", "8"]] for i in range(10)]
        dots = [int(l) for sl in dots for l in sl]

        # If divisible by ten, print the number
        if percent % 10 == 0 and percent != 0:
            print("{}".format(percent), end="")

        # If one of three numbers between multiples of 10, print a dot
        elif percent in dots:
            print(".", end="")

        return 1

    # Overwrite existing file
    if os.path.exists(dst):
        if overwrite:
            if os.path.isfile(dst):
                os.remove(dst)
            else:
                shutil.rmtree(dst)
        else:
            print(dst + " exists, use overwrite=True to replace this file.")
            return

    # Specifying data types shouldn't be so difficult
    if isinstance(dtype, str):
        dtype = dtype.lower().replace("gdt_", "")
        try:
            dtype = GDAL_TYPEMAP[dtype]
        except KeyError:
            print("\n'" + dtype + "' is not an available data type. "
                  "Choose a value from this list:")
            print(str(list(GDAL_TYPEMAP.keys())))

    # Create a spatial reference object
    spatial_ref = osr.SpatialReference()

    # If a template is provided, use its geometry for target figures
    if template:
        temp = gdal.Open(template)
        spatial_ref.ImportFromWkt(temp.GetProjection())
        srs = spatial_ref.ExportToProj4()
        width = temp.RasterXSize  # consider using these warp options
        height = temp.RasterYSize
        transform = temp.GetGeoTransform()
        xmin, xres, xrot, ymax, yrot, yres = transform
        xs = [xmin + xres * i for i in range(width)]
        ys = [ymax + yres * i for i in range(height)]
        xmax = max(xs) + 0.5*xres
        ymax = ymax + 0.5*xres
        ymin = min(ys)
        extent = [xmin, ymin, xmax, ymax]
        kwargs["dstSRS"] = srs
        kwargs["outputBounds"] = extent
        kwargs["xRes"] = transform[1]
        kwargs["yRes"] = transform[-1]  # careful here
        kwargs["outputType"] = dtype
    elif not kwargs:
        print("No warp options provided.")
        gdal_options("warp")
        return

    # Get source srs
    source = gdal.Open(src)
    spatial_ref.ImportFromWkt(source.GetProjection())
    srs = spatial_ref.ExportToProj4()
    if srs:
        kwargs["srcSRS"] = srs

    # Use the progress callback
    if progress:
        print("Processing " + dst + " :")
        kwargs["callback"] = gdal_progress

    # Compress
    if compress:
        kwargs["creationOptions"] = ["COMPRESS=" + compress]

    # Check Options: https://gdal.org/python/osgeo.gdal-module.html#WarpOptions
    try:
        ops = gdal_options("warp", **kwargs)
    except Exception as error:
        print(error)

    # Call
    ds = gdal.Warp(dst, src, options=ops)
    del ds
