# -*- coding: utf-8 -*-
"""
Just an app to visualize raster time series.

Created on Fri Jan  4 12:39:23 2019

@author: User
"""

# In[] Functions and Libraries
import copy
import dash
from dash.dependencies import Input, Output, State, Event
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
import gc
import gdal
import glob
import json
from flask import Flask
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import numpy as np
import numpy.ma as ma
from collections import OrderedDict
import os
import pandas as pd
import plotly
import re
from textwrap import dedent
import time
from tqdm import tqdm
import xarray as xr
from sys import platform
import warnings
# warnings.filterwarnings("ignore")

# Work for Windows and Linux
if platform == 'win32':
    home_path = 'c:/users/user/github'
    data_path = 'd:/'
    os.chdir(os.path.join(home_path, 'Ubuntu-Practice-Machine'))
    from flask_cache import Cache  # This one works on Windows but not Linux
    startyear = 1948
else:
    home_path = '/home/ubuntu'  # Not sure yet
    os.chdir(os.path.join(home_path, 'Ubuntu-Practie-Machine'))
    data = '/home/ubunutu'
    from flask_caching import Cache  # This works on Linux but not Windows :)
    startyear = 1980

from functions import indexHist
from functions import npzIn
from functions import calculateCV
# In[] Create the DASH App object
app = dash.Dash(__name__)

# Go to stylesheet, styled after a DASH example (how to serve locally?)
app.css.append_css({'external_url': 'https://codepen.io/williamstravis/pen/' +
                                    'maxwvK.css'})

# Create Server Object
server = app.server

# Create and initialize a cache for data storage
cache = Cache(config={'CACHE_TYPE': 'simple'})
cache.init_app(server)

# Mapbox Access
mapbox_access_token = ('pk.eyJ1IjoidHJhdmlzc2l1cyIsImEiOiJjamZiaHh4b28waXNk' +
                       'MnptaWlwcHZvdzdoIn0.9pxpgXxyyhM6qEF_dcyjIQ')

# In[] Drought and Climate Indices (looking to include any raster time series)
# Index Paths (for npz files)
indices = [{'label': 'Rainfall Index', 'value': 'noaa'},
           {'label': 'PDSI', 'value': 'pdsi'},
           {'label': 'PDSI-Self Calibrated', 'value': 'pdsisc'},
           {'label': 'Palmer Z Index', 'value': 'pdsiz'},
           {'label': 'SPI-1', 'value': 'spi1'},
           {'label': 'SPI-2', 'value': 'spi2'},
           {'label': 'SPI-3', 'value': 'spi3'},
           {'label': 'SPI-6', 'value': 'spi6'},
           {'label': 'SPEI-1', 'value': 'spei1'},
           {'label': 'SPEI-2', 'value': 'spei2'},
           {'label': 'SPEI-3', 'value': 'spei3'},
           {'label': 'SPEI-6', 'value': 'spei6'},
           {'label': 'EDDI-1', 'value': 'eddi1'},
           {'label': 'EDDI-2', 'value': 'eddi2'},
           {'label': 'EDDI-3', 'value': 'eddi3'},
           {'label': 'EDDI-6', 'value': 'eddi6'}]

# Index dropdown labels
indexnames = {'noaa': 'NOAA CPC-Derived Rainfall Index',
              'pdsi': 'Palmer Drought Severity Index',
              'pdsisc': 'Self-Calibrated Palmer Drought Severity Index',
              'pdsiz': 'Palmer Z Index',
              'spi1': 'Standardized Precipitation Index - 1 month',
              'spi2': 'Standardized Precipitation Index - 2 month',
              'spi3': 'Standardized Precipitation Index - 3 month',
              'spi6': 'Standardized Precipitation Index - 6 month',
              'spei1': 'Standardized Precipitation-Evapotranspiration Index' +
                       ' - 1 month',
              'spei2': 'Standardized Precipitation-Evapotranspiration Index' +
                       ' - 2 month',
              'spei3': 'Standardized Precipitation-Evapotranspiration Index' +
                       ' - 3 month',
              'spei6': 'Standardized Precipitation-Evapotranspiration Index' +
                       ' - 6 month',
              'eddi1': 'Evaporative Demand Drought Index - 1 month',
              'eddi2': 'Evaporative Demand Drought Index - 2 month',
              'eddi3': 'Evaporative Demand Drought Index - 3 month',
              'eddi6': 'Evaporative Demand Drought Index - 6 month'}

# Function options
function_options = [{'label': 'Mean - Percentiles', 'value': 'mean_perc'},
                    {'label': 'Coefficient of Variation - Original Values',
                     'value': 'cv'}]
# In[] The map
# Map types
maptypes = [{'label': 'Light', 'value': 'light'},
            {'label': 'Dark', 'value': 'dark'},
            {'label': 'Basic', 'value': 'basic'},
            {'label': 'Outdoors', 'value': 'outdoors'},
            {'label': 'Satellite', 'value': 'satellite'},
            {'label': 'Satellite Streets', 'value': 'satellite-streets'}]

# Year Marks for Slider
years = [int(y) for y in range(startyear, 2018)]
yearmarks = dict(zip(years, years))
for y in yearmarks:
    if y % 5 != 0:
        yearmarks[y] = ""

# Set up initial signal and raster to scatterplot conversion
# A source grid for scatterplot maps - will need more for optional resolution
source = xr.open_dataarray(os.path.join(data_path,
                                        "data/droughtindices/source_array.nc"))

# Create Coordinate index positions from xarray
# Geometry
x_length = source.shape[2]
y_length = source.shape[1]
res = source.res[0]  
lon_min = source.transform[0]
lat_max = source.transform[3] - res

# Make dictionaires with coordinates and array index positions
xs = range(x_length)
ys = range(y_length)
lons = [lon_min + res*x for x in xs]
lats = [lat_max - res*y for y in ys]
londict = dict(zip(lons, xs))
latdict = dict(zip(lats, ys))
londict2 = {y: x for x, y in londict.items()}
latdict2 = {y: x for x, y in latdict.items()}

# Map Layout:
# Check this out! https://paulcbauer.shinyapps.io/plotlylayout/
layout = dict(
    autosize=True,
    height=500,
    font=dict(color='#CCCCCC',
              fontweight='bold'),
    titlefont=dict(color='#CCCCCC',
                   size='20',
                   family='Time New Roman',
                   fontweight='bold'),
    margin=dict(
        l=55,
        r=35,
        b=65,
        t=95,
        pad=4
    ),
    hovermode="closest",
    plot_bgcolor="#083C04",
    paper_bgcolor="#0D347C",
    legend=dict(font=dict(size=10, fontweight='bold'), orientation='h'),
    title='<b>Potential Payout Frequencies</b>',
    mapbox=dict(
        accesstoken=mapbox_access_token,
        style="satellite-streets",
        center=dict(
            lon=-95.7,
            lat=37.1
        ),
        zoom=2,
    )
)

# In[]: Create App Layout
app.layout = html.Div([
        html.Div([html.Img(id='banner',
                           src=('https://github.com/WilliamsTravis/' +
                                'Ubuntu-Practice-Machine/blob/master/images/' +
                                'banner1.png?raw=true'),
                  style={'width': '100%',
                         'box-shadow': '1px 1px 1px 1px black'})]),
        html.Hr(),
        html.Div([html.H1('Raster to Scatterplot Visualization')],
                 className='twelve columns',
                 style={'font-weight': 'bold',
                        'text-align': 'center',
                        'font-family': 'Times New Roman'}),

        # Year Slider
        html.Div([
                 html.Hr(),
                 html.P('Study Period Year Range'),
                 dcc.RangeSlider(
                     id='year_slider',
                     value=[2017, 2017],
                     min=startyear,
                     max=2017,
                     marks=yearmarks)],
                 className="twelve columns",
                 style={'margin-top': '0',
                        'margin-bottom': '40'}),

        # Maptype
        html.Div([
                html.P("Map Type"),
                dcc.Dropdown(
                        id="map_type",
                        value="light",
                        options=maptypes,
                        multi=False)],
                style={'width': '25%'}),

        # Four by Four Map Layout
        # Row 1
        html.Div([
                 html.Div([
                          html.Div([
                                   html.Div([dcc.Dropdown(id='choice_1',
                                                options=indices,
                                                value='pdsi')],
                                            className='three columns'),
                                   html.Div([dcc.Dropdown(id='function_1',
                                                options=function_options,
                                                value='mean_perc')],
                                             className='six columns')],
                                   # style={'width': '30%'},
                                   className='row'),
                          dcc.Graph(id='map_1')],
                          className='six columns',
                          style={'float': 'left',
                                 'margin-top': '40'}),
                 html.Div([
                          html.Div([
                                   dcc.Dropdown(id='choice_2',
                                                options=indices,
                                                value='noaa')],
                                   style={'width': '35%'}),
                          dcc.Graph(id='map_2')],
                          className='six columns',
                          style={'float': 'right',
                                 'margin-top': '40'})],
                 className='row'),

        # Row 2
        html.Div([
                 html.Div([
                          html.Div([
                                   dcc.Dropdown(id='choice_3',
                                                options=indices,
                                                value='noaa')],
                                   style={'width': '35%'}),
                          dcc.Graph(id='map_3')],
                          className='six columns',
                          style={'float': 'left',
                                 'margin-top': '40'}),
                 html.Div([
                          html.Div([
                                   dcc.Dropdown(id='choice_4',
                                                options=indices,
                                                value='noaa')],
                                   style={'width': '35%'}),
                          dcc.Graph(id='map_4')],
                          className='six columns',
                          style={'float': 'right',
                                 'margin-top': '40'})],
                 className='row'),
    # The end!
        ],
    className='ten columns offset-by-one')


# In[]: App callbacks

@app.callback(Output('map_1', 'figure'),
              [Input('choice_1', 'value'),
               Input('function_1', 'value'),
               Input('year_slider', 'value'),
               Input('map_type', 'value')])
def makeMap1(choice, function, year_range, map_type):
    # Clear memory space...what's the best way to do this?
    gc.collect()

    # Get numpy arrays
    if function == 'mean_perc':
        array_path = os.path.join(data_path,
                                  "data/droughtindices/npz/percentiles",
                                  choice + '_arrays.npz')
        date_path = os.path.join(data_path,
                                 "data/droughtindices/npz/percentiles",
                                 choice + '_dates.npz')
        indexlist = npzIn(array_path, date_path)

        # Get total Min and Max Values for colors
        dmin = np.nanmin([i[1] for i in indexlist])
        dmax = np.nanmax([i[1] for i in indexlist])

        # filter by year
        indexlist = [a for a in indexlist if
                     int(a[0][-6:-2]) >= year_range[0] and
                     int(a[0][-6:-2]) <= year_range[1]]
        arrays = [i[1] for i in indexlist]
        
        # Apply chosen funtion
        array = np.nanmean(arrays, axis=0)
        
        # Colors - RdYlGnBu
        colorscale = [[0.00, 'rgb(197, 90, 58)'],
                      [0.25, 'rgb(255, 255, 48)'],
                      [0.50, 'rgb(39, 147, 57)'],
                      # [0.75, 'rgb(6, 104, 70)'],
                      [1.00, 'rgb(1, 62, 110)']]

    else:
        array_path = os.path.join(data_path,
                                  "data/droughtindices/npz",
                                  choice + '_arrays.npz')
        date_path = os.path.join(data_path,
                                 "data/droughtindices/npz",
                                 choice + '_dates.npz')
        indexlist = npzIn(array_path, date_path)

        # Get total Min and Max Values for colors
        dmin = 0
        dmax = 1

        # filter by year
        indexlist = [a for a in indexlist if 
                     int(a[0][-6:-2]) >= year_range[0] and
                     int(a[0][-6:-2]) <= year_range[1]]
        arrays = [i[1] for i in indexlist]
        
        # Apply chosen funtion
        array = calculateCV(arrays)

        # Colors - RdYlGnBu
        colorscale = [[0.00, 'rgb(1, 62, 110)'],
                      [0.35, 'rgb(6, 104, 70)'],
                      [0.45, 'rgb(39, 147, 57)'],
                      [0.55, 'rgb(255, 255, 48)'],
                      [1.00, 'rgb(197, 90, 58)']]

    # get coordinate-array index dictionaries data!
    source.data[0] = array

    # Now all this
    dfs = xr.DataArray(source, name="data")
    pdf = dfs.to_dataframe()
    step = res
    to_bin = lambda x: np.floor(x / step) * step
    pdf["latbin"] = pdf.index.get_level_values('y').map(to_bin)
    pdf["lonbin"] = pdf.index.get_level_values('x').map(to_bin)
    pdf['gridx'] = pdf['lonbin'].map(londict)
    pdf['gridy'] = pdf['latbin'].map(latdict)
    # grid2 = np.copy(grid)
    # grid2[np.isnan(grid2)] = 0
    # pdf['grid'] = grid2[pdf['gridy'], pdf['gridx']]
    # pdf['grid'] = pdf['grid'].apply(int).apply(str)
    # pdf['data'] = pdf['data'].astype(float).round(3)
    # pdf['printdata'] = "GRID #: " + pdf['grid'] + "<br>Data: " + pdf['data'].apply(str)

    df_flat = pdf.drop_duplicates(subset=['latbin', 'lonbin'])
    df = df_flat[np.isfinite(df_flat['data'])]

    # Create the scattermapbox object
    data = [
        dict(
            type='scattermapbox',
            lon=df['lonbin'],
            lat=df['latbin'],
            text=df['data'],
            mode='markers',
            hoverinfo='text',
            marker=dict(
                colorscale=colorscale,
                cmin=dmin,
                color=df['data'],
                cmax=dmax,
                opacity=0.85,
                size=5,
                colorbar=dict(
                    textposition="auto",
                    orientation="h",
                    font=dict(size=15,
                              fontweight='bold')
                )
            )
        )]

    layout['mapbox'] = dict(
        accesstoken=mapbox_access_token,
        style=map_type,
        center=dict(lon=-95.7, lat=37.1),
        zoom=2)

    figure = dict(data=data, layout=layout)
    return figure

@app.callback(Output('banner', 'src'),
              [Input('choice_1', 'value')])
def whichBanner(value):
    # which banner?
    time_modulo = round(time.time()) % 5
    print(str(time_modulo))
    banners = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
    image_time = banners[time_modulo]
    image = ('https://github.com/WilliamsTravis/' +
             'Ubuntu-Practice-Machine/blob/master/images/' +
             'banner' + str(image_time) + '.png?raw=true')
    return image

# In[] Run Application through the server
if __name__ == '__main__':
    app.run_server()
