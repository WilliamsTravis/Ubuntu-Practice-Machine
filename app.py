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
import datetime as dt
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
from functions import readRaster

# In[] for development
# function =  'mean_perc'
# choice = 'pdsi'
# year_range = [2000, 2017]

# In[] Create the DASH App object
app = dash.Dash(__name__)

# Go to stylesheet, styled after a DASH example (how to serve locally?)
app.css.append_css({'external_url': 'https://codepen.io/williamstravis/pen/' +
                                    'maxwvK.css'})

# Create Server Object
server = app.server

# Disable exceptions (attempt to speed things up)
app.config['suppress_callback_exceptions'] = True

# Create and initialize a cache for data storage
cache = Cache(config={'CACHE_TYPE': 'simple'})
cache.init_app(server)

# Mapbox Access
mapbox_access_token = ('pk.eyJ1IjoidHJhdmlzc2l1cyIsImEiOiJjamZiaHh4b28waXNk' +
                       'MnptaWlwcHZvdzdoIn0.9pxpgXxyyhM6qEF_dcyjIQ')


# In[] Functions
def makeMap(function, choice, year_range):

    # Get numpy arrays
    if function != 'mean_original':
        array_path = os.path.join(
                data_path, "data/droughtindices/netcdfs/percentiles",
                choice + '.nc')

        indexlist = xr.open_dataset(array_path)

        # Get total Min and Max Values for colors
        dmin = 0
        dmax = 1

        # filter by year
        y1 = dt.datetime(year_range[0], 1, 1)
        y2 = dt.datetime(year_range[1], 12, 1)
        arrays = indexlist.sel(time=slice(y1, y2))

        # Apply chosen funtion
        if function == 'mean_perc':
            data = arrays.mean('time')
            array = data.value.data
            array[array == 0] = np.nan
        elif function == 'max':
            data = arrays.max('time')
            array = data.value.data
            array[array == 0] = np.nan
        else:
            data = arrays.min('time')
            array = data.value.data
            array[array == 0] = np.nan

        # Colors - USDM Scale
        colorscale = [[0.00, 'rgb(115,0,0)'],
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
                      [1.00, 'rgb(5,16,110)']]

    else:
        array_path = os.path.join(
                data_path, "data/droughtindices/netcdfs",
                choice + '.nc')

        indexlist = xr.open_dataset(array_path)

        # Get total Min and Max Values for colors
        values = indexlist.value.data
        values[values == 0] = np.nan
        dmin = np.nanmin(values)
        dmax = np.nanmax(values)

        # filter by year
        y1 = dt.datetime(year_range[0], 1, 1)
        y2 = dt.datetime(year_range[1], 12, 1)
        arrays = indexlist.sel(time=slice(y1, y2))

        # Apply chosen funtion
        data = arrays.mean('time')
        array = data.value.data

        # Colors - RdYlGnBu
        colorscale = [[0.00, 'rgb(124, 36, 36)'],
                      [0.25, 'rgb(255, 255, 48)'],
                      [0.5, 'rgb(76, 145, 33)'],
                       [0.85, 'rgb(0, 92, 221)'],
                      [1.00, 'rgb(0, 46, 110)']]

    return [[array, arrays], colorscale, dmax, dmin]


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
                    {'label': 'Mean - Original Values',
                     'value': 'mean_original'},
                    {'label': 'Maximum - Percentiles', 'value': 'max'},
                    {'label': 'Minimum - Percentiles', 'value': 'min'}]
function_names = {'mean_perc': 'Average Percentiles',
                  'mean_original': 'Original Index Values',
                  'min': 'Minimum Percentile',
                  'max': 'Maxmium Percentile'}

# For the city list
grid = readRaster(data_path + 'data/droughtindices/prfgrid.tif', 1, -9999)[0]
cities_df = pd.read_csv("cities.csv")

cities = [{'label': cities_df['NAME'][i] + ", " + cities_df['STATE'][i],
           'value': cities_df['grid'][i]} for i in range(len(cities_df))]
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
        t=90,
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


# In[]: Create a Div maker
def divMaker(id_num, index='noaa'):
    div = html.Div([

                 html.Div([dcc.Dropdown(id='choice_{}'.format(id_num),
                                        options=indices, value=index)],
                          style={'width': '30%'}),
                 dcc.Graph(id='map_{}'.format(id_num)),
                 html.Div([dcc.Graph(id='series_{}'.format(id_num))]),

              ], className='six columns')
    return div


# In[]: Create App Layout
app.layout = html.Div([
        # html.Div([
        #         html.Img(id='banner',
        #                  src=('https://github.com/WilliamsTravis/' +
        #                       'Ubuntu-Practice-Machine/blob/master/images/' +
        #                       'banner1.png?raw=true'),
        #                  style={'width': '100%',
        #                         'box-shadow': '1px 1px 1px 1px black'})]),
        # html.Hr(),
        html.Div([html.H1('Raster to Scatterplot Visualization')],
                 className='twelve columns',
                 style={'font-weight': 'bold',
                        'text-align': 'center',
                        'font-size': '50px',
                        'font-family': 'Times New Roman'}),

        # Year Slider
        html.Div([
                 html.Hr(),
                 html.H3('Study Period Year Range'),
                 html.Div([dcc.RangeSlider(
                             id='year_slider',
                             value=[2000, 2017],
                             min=startyear,
                             max=2017,
                             marks=yearmarks)],
                 className="twelve columns",
                 style={'margin-top': '0',
                        'margin-bottom': '40'})]),
        # Submit button
        html.Div([
                html.Hr(),
                html.Button(id='submit',
                            children='Submit Years',
                            type='button',
                            title='It updates automatically without this.')]),

        # Options
        html.Div([
            # Maptype
            html.Div([
                    html.H3("Map Type"),
                     dcc.RadioItems(
                            id="map_type",
                            value="light",
                            options=maptypes)],
                     className='two columns'),

            # Function
            html.Div([
                     html.H3("Function"),
                     dcc.RadioItems(id='function_choice',
                                    options=function_options,
                                    value='mean_original')],
                     className='three columns'),
            # Syncing locations on click
            html.Div([
                    html.H3("Sync Click Locations"),
                    dcc.RadioItems(id='click_sync',
                                   options=[{'label': 'Yes', 'value': 'yes'},
                                            {'label': 'No', 'value': 'no'}],
                                   value='yes')],
                     className='three columns')
            ],
           className='row',
           style={'margin-bottom': '50',
                  'margin-top': '0'}),
        
        # Break    
        html.Hr(),
    
        # Four by Four Map Layout
        # Row 1
        html.Div([divMaker(1, 'spei1'), divMaker(2, 'spei2')],
                 className='row'),

        # Row 2
        html.Div([divMaker(3, 'spei3'), divMaker(4, 'spei6')],
                 className='row'),

        # Signals
        html.Div(id='signal', style={'display': 'none'}),
        html.Div(id='click_store', style={'display': 'none'}),
        html.Div(id='cache_1', children='1', style={'display': 'none'}),
        html.Div(id='cache_2', children='2', style={'display': 'none'}),
        html.Div(id='cache_3', children='3', style={'display': 'none'}),
        html.Div(id='cache_4', children='4', style={'display': 'none'}),
        html.Div(id='time_1', style={'display': 'none'}),
        html.Div(id='time_2', style={'display': 'none'}),
        html.Div(id='time_3', style={'display': 'none'}),
        html.Div(id='time_4', style={'display': 'none'}),


        # The end!
        ], className='ten columns offset-by-one')


# In[]: App callbacks
@cache.memoize()
def global_store1(signal):
    gc.collect()
    data = makeMap(signal[1], signal[0], signal[2])
    return data
def retrieve_data1(signal):
    data = global_store1(signal)
    return data

@cache.memoize()
def global_store2(signal):
    data = makeMap(signal[1], signal[0], signal[2])
    return data
def retrieve_data2(signal):
    data = global_store2(signal)
    return data

@cache.memoize()
def global_store3(signal):
    data = makeMap(signal[1], signal[0], signal[2])
    return data
def retrieve_data3(signal):
    data = global_store3(signal)
    return data

@cache.memoize()
def global_store4(signal):
    data = makeMap(signal[1], signal[0], signal[2])
    return data
def retrieve_data4(signal):
    data = global_store4(signal)
    return data


# Store data in the cache and hide the signal to activate it in the hidden div
@app.callback(Output('signal', 'children'),
              [Input('submit', 'n_clicks')],
              [State('year_slider', 'value')])
def submitSignal(click, year_range):
    print(year_range)
    return json.dumps(year_range)


@app.callback(Output('click_store', 'children'),
              [Input('map_1', 'clickData'),
               Input('map_2', 'clickData'),
               Input('map_3', 'clickData'),
               Input('map_4', 'clickData')],
              [State('click_sync', 'value'),
               State('time_1', 'children'),
               State('time_2', 'children'),
               State('time_3', 'children'),
               State('time_4', 'children')])
def clickPicker(click1, click2, click3, click4, click_sync,
                time1, time2, time3, time4):
    print(click_sync)
    if click_sync == 'yes':
        clicks = [click1, click2, click3, click4]
        times = [time1, time2, time3, time4]
        if not any(c is not None for c in clicks):
            click = {'points': [{'curveNumber': 0, 'pointNumber': 5755,
                                 'pointIndex': 5755, 'lon': -107.75,
                                 'lat': 40.5, 'text': -0.08303022384643555,
                                 'marker.color': -0.08303022384643555}]}
        else:
            times = [0 if t is None else t for t in times]
            index = times.index(max(times))
            click = clicks[index]
        return json.dumps(click)
    else:
        pass


# In[] For the future
for i in range(1, 5):
    @app.callback(Output('time_{}'.format(i), 'children'),
                  [Input('map_{}'.format(i), 'clickData')])
    def clickTime(click):
        print(click)
        clicktime = time.time()
        return(clicktime)

    @app.callback(Output("map_{}".format(i), 'figure'),
                  [Input('signal', 'children'),
                   Input('cache_{}'.format(i), 'children'),
                   Input('choice_{}'.format(i), 'value'),
                   Input('function_choice', 'value'),
                   Input('map_type', 'value')])
    def makeGraph(year_range, cache, choice, function, map_type):
        # Clear memory space...what's the best way to do this?
        gc.collect()

        # Create signal for the global_store
        year_range = json.loads(year_range)
        signal = [choice, function, year_range]

        # Get data - check which cache first
        if cache == '1':
            [[array, arrays],
             colorscale, dmax, dmin] = retrieve_data1(signal)
        elif cache == '2':
            [[array, arrays],
             colorscale, dmax, dmin] = retrieve_data2(signal)
        elif cache == '3':
            [[array, arrays],
             colorscale, dmax, dmin] = retrieve_data3(signal)
        else:
            [[array, arrays],
             colorscale, dmax, dmin] = retrieve_data4(signal)

        # Individual array min/max
        amax = np.nanmax(array)
        amin = np.nanmin(array)

        # Set to this thing
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

        # For hover information
        grid2 = np.copy(grid)
        grid2[np.isnan(grid2)] = 0
        pdf['grid'] = grid2[pdf['gridy'], pdf['gridx']]
        pdf['grid'] = pdf['grid'].apply(int).apply(str)
        pdf['data'] = pdf['data'].astype(float).round(3)
        pdf['printdata'] = "GRID #: " + pdf['grid'] + "<br>Data: " + pdf['data'].apply(str)

        df_flat = pdf.drop_duplicates(subset=['latbin', 'lonbin'])
        df = df_flat[np.isfinite(df_flat['data'])]

        # Create the scattermapbox object
        data = [
            dict(
                type='scattermapbox',
                lon=df['lonbin'],
                lat=df['latbin'],
                text=df['printdata'],
                mode='markers',
                hoverinfo='text',
                hovermode='closest',
                marker=dict(
                    colorscale=colorscale,
                    color=df['data'],
                    cmax=amax,
                    cmin=amin,
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

        if year_range[0] != year_range[1]:
            year_print = '{} - {}'.format(year_range[0], year_range[1])
        else:
            year_print = str(year_range[0])

        layout_copy = copy.deepcopy(layout)
        layout_copy['mapbox'] = dict(
            accesstoken=mapbox_access_token,
            style=map_type,
            center=dict(lon=-95.7, lat=37.1),
            zoom=2)

        layout_copy['title'] = (indexnames[choice] + '<br>' +
                                function_names[function] + ', ' +
                                year_print)

        figure = dict(data=data, layout=layout_copy)
        return figure

    @app.callback(Output('series_{}'.format(i), 'figure'),
                  [Input('click_sync', 'value'),
                   Input("map_{}".format(i), 'clickData'),
                   Input('click_store', 'children'),
                   Input('choice_{}'.format(i), 'value'),
                   Input('signal', 'children'),
                   Input('function_choice', 'value')])
    def makeSeries(click_sync, click, synced_click, choice, year_range,
                   function):
        year_range = json.loads(year_range)
        signal = [choice, function, year_range]

        # Get data - check which cache first
        if cache == '1':
            [[array, arrays],
             colorscale, dmax, dmin] = retrieve_data1(signal)
        elif cache == '2':
            [[array, arrays],
             colorscale, dmax, dmin] = retrieve_data2(signal)
        elif cache == '3':
            [[array, arrays],
             colorscale, dmax, dmin] = retrieve_data3(signal)
        else:
            [[array, arrays],
             colorscale, dmax, dmin] = retrieve_data4(signal)

        # # Check if we are syncing clicks
        if click_sync == 'yes':
            if synced_click:
                click = json.loads(synced_click)
            else:
                click = {'points': [{'curveNumber': 0, 'pointNumber': 5755,
                                     'pointIndex': 5755, 'lon': -107.75,
                                     'lat': 40.5, 'text': -0.08303022384643555,
                                     'marker.color': -0.08303022384643555}]}

        # Get Coordinates
        if click is None:
            x = londict[-100]
            y = latdict[40]
            gridid = grid[y, x]
            county = cities_df['COUNTY'][cities_df.grid == gridid].unique()
        else:
            lon = click['points'][0]['lon']
            lat = click['points'][0]['lat']
            x = londict[lon]
            y = latdict[lat]
            gridid = grid[y, x]
            print(str(gridid))
            county = cities_df['COUNTY'][cities_df.grid == gridid].unique()

        # There are often more than one county, sometimes none in this df
        if len(county) == 0:
            county = ""
        else:
            county = county[0]

        # Get time series
        dates1 = arrays.time.data
        dates2 = [pd.to_datetime(str(d)) for d in dates1]
        dates = [d.strftime('%Y-%m') for d in dates2]
        just_arrays = [a.data for a in arrays.value]
        timeseries = np.array([round(a[y, x], 4) for a in just_arrays])

        # Build the data dictionaries that plotly reads
        data = [
            dict(
                type='bar',
                # yaxis=dict(range=[dmin, dmax]),
                x=dates,
                y=timeseries,
                marker=dict(color=timeseries,
                            colorscale=colorscale,
                            line=dict(width=0.2, color="#000000")),
            )]

        # Copy and customize Layout
        layout_copy = copy.deepcopy(layout)
        layout_copy['title'] = ("Time Series - Individual Observations" +
                                "<Br>" + county)
        layout_copy['plot_bgcolor'] = "white"
        layout_copy['paper_bgcolor'] = "white"
        layout_copy['height'] = 250
        layout_copy['titlefont']['color'] = '#636363'
        layout_copy['font']['color'] = '#636363'
        figure = dict(data=data, layout=layout_copy)

        return figure


# In[]
# @app.callback(Output('banner', 'src'),
#               [Input('choice_1', 'value')])
# def whichBanner(value):
#     # which banner?
#     time_modulo = round(time.time()) % 5
#     print(str(time_modulo))
#     banners = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}
#     image_time = banners[time_modulo]
#     image = ('https://github.com/WilliamsTravis/' +
#              'Ubuntu-Practice-Machine/blob/master/images/' +
#              'banner' + str(image_time) + '.png?raw=true')
#     return image


# In[] Run Application through the server
if __name__ == '__main__':
    app.run_server()
