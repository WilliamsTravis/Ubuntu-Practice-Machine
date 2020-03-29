# -*- coding: utf-8 -*-
"""
Constants such as menus items for DrIP.

Created on Sat Mar 28 13:16:28 2020

@author: travis
"""


import pathlib

REPO = "Drought-Index-Portal"
PWD = str(pathlib.Path(__file__).parent.absolute())
PROJECT_DIR = PWD[:PWD.index(REPO) + len(REPO)]

INDICES = [{'label': 'PDSI', 'value': 'pdsi'},
           {'label': 'PDSI-SC', 'value': 'scpdsi'},
           {'label': 'Palmer Z Index', 'value': 'pzi'},
           {'label': 'SPI-1', 'value': 'spi1'},
           {'label': 'SPI-2', 'value': 'spi2'},
           {'label': 'SPI-3', 'value': 'spi3'},
           {'label': 'SPI-4', 'value': 'spi4'},
           {'label': 'SPI-5', 'value': 'spi5'},
           {'label': 'SPI-6', 'value': 'spi6'},
           {'label': 'SPI-7', 'value': 'spi7'},
           {'label': 'SPI-8', 'value': 'spi8'},
           {'label': 'SPI-9', 'value': 'spi9'},
           {'label': 'SPI-10', 'value': 'spi10'},
           {'label': 'SPI-11', 'value': 'spi11'},
           {'label': 'SPI-12', 'value': 'spi12'},
           {'label': 'SPEI-1', 'value': 'spei1'},
           {'label': 'SPEI-2', 'value': 'spei2'},
           {'label': 'SPEI-3', 'value': 'spei3'},
           {'label': 'SPEI-4', 'value': 'spei4'},
           {'label': 'SPEI-5', 'value': 'spei5'},
           {'label': 'SPEI-6', 'value': 'spei6'},
           {'label': 'SPEI-7', 'value': 'spei7'},
           {'label': 'SPEI-8', 'value': 'spei8'},
           {'label': 'SPEI-9', 'value': 'spei9'},
           {'label': 'SPEI-10', 'value': 'spei10'},
           {'label': 'SPEI-11', 'value': 'spei11'},
           {'label': 'SPEI-12', 'value': 'spei12'},
           {'label': 'EDDI-1', 'value': 'eddi1'},
           {'label': 'EDDI-2', 'value': 'eddi2'},
           {'label': 'EDDI-3', 'value': 'eddi3'},
           {'label': 'EDDI-4', 'value': 'eddi4'},
           {'label': 'EDDI-5', 'value': 'eddi5'},
           {'label': 'EDDI-6', 'value': 'eddi6'},
           {'label': 'EDDI-7', 'value': 'eddi7'},
           {'label': 'EDDI-8', 'value': 'eddi8'},
           {'label': 'EDDI-9', 'value': 'eddi9'},
           {'label': 'EDDI-10', 'value': 'eddi10'},
           {'label': 'EDDI-11', 'value': 'eddi11'},
           {'label': 'EDDI-12', 'value': 'eddi12'},
           # {'label': 'LERI-1', 'value': 'leri1'},
           # {'label': 'LERI-3', 'value': 'leri3'},
           {'label': 'TMIN', 'value': 'tmin'},
           {'label': 'TMAX', 'value': 'tmax'},
           {'label': 'TMEAN', 'value': 'tmean'},
           {'label': 'TDMEAN', 'value': 'tdmean'},
           {'label': 'PPT', 'value': 'ppt'},
           {'label': 'VPDMAX', 'value': 'vpdmax'},
           {'label': 'VPDMIN', 'value': 'vpdmin'}
           # {'label': 'VPDMEAN', 'value': 'vpdmean'}
           ]

NONINDICES = ['tdmean', 'tmean', 'tmin', 'tmax', 'ppt',  'vpdmax',
              'vpdmin', 'vpdmean']

TITLE_MAP = {'noaa': 'NOAA CPC-Derived Rainfall Index',
             'mdn1': 'Mean Temperature Departure  (1981 - 2010) - 1 month',
             'pdsi': 'Palmer Drought Severity Index',
             'scpdsi': 'Self-Calibrated Palmer Drought Severity Index',
             'pzi': 'Palmer Z-Index',
             'spi1': 'Standardized Precipitation Index - 1 month',
             'spi2': 'Standardized Precipitation Index - 2 month',
             'spi3': 'Standardized Precipitation Index - 3 month',
             'spi4': 'Standardized Precipitation Index - 4 month',
             'spi5': 'Standardized Precipitation Index - 5 month',
             'spi6': 'Standardized Precipitation Index - 6 month',
             'spi7': 'Standardized Precipitation Index - 7 month',
             'spi8': 'Standardized Precipitation Index - 8 month',
             'spi9': 'Standardized Precipitation Index - 9 month',
             'spi10': 'Standardized Precipitation Index - 10 month',
             'spi11': 'Standardized Precipitation Index - 11 month',
             'spi12': 'Standardized Precipitation Index - 12 month',
             'spei1': 'Standardized Precipitation-Evapotranspiration Index' +
                      ' - 1 month',
             'spei2': 'Standardized Precipitation-Evapotranspiration Index' +
                      ' - 2 month',
             'spei3': 'Standardized Precipitation-Evapotranspiration Index' +
                      ' - 3 month',
             'spei4': 'Standardized Precipitation-Evapotranspiration Index' +
                      ' - 4 month',
             'spei5': 'Standardized Precipitation-Evapotranspiration Index' +
                      ' - 5 month',
             'spei6': 'Standardized Precipitation-Evapotranspiration Index' +
                      ' - 6 month',
             'spei7': 'Standardized Precipitation-Evapotranspiration Index' +
                      ' - 7 month',
             'spei8': 'Standardized Precipitation-Evapotranspiration Index' +
                      ' - 8 month',
             'spei9': 'Standardized Precipitation-Evapotranspiration Index' +
                      ' - 9 month',
             'spei10': 'Standardized Precipitation-Evapotranspiration Index' +
                       ' - 10 month',
             'spei11': 'Standardized Precipitation-Evapotranspiration Index' +
                       ' - 11 month',
             'spei12': 'Standardized Precipitation-Evapotranspiration Index' +
                       ' - 12 month',
             'eddi1': 'Evaporative Demand Drought Index - 1 month',
             'eddi2': 'Evaporative Demand Drought Index - 2 month',
             'eddi3': 'Evaporative Demand Drought Index - 3 month',
             'eddi4': 'Evaporative Demand Drought Index - 4 month',
             'eddi5': 'Evaporative Demand Drought Index - 5 month',
             'eddi6': 'Evaporative Demand Drought Index - 6 month',
             'eddi7': 'Evaporative Demand Drought Index - 7 month',
             'eddi8': 'Evaporative Demand Drought Index - 8 month',
             'eddi9': 'Evaporative Demand Drought Index - 9 month',
             'eddi10': 'Evaporative Demand Drought Index - 10 month',
             'eddi11': 'Evaporative Demand Drought Index - 11 month',
             'eddi12': 'Evaporative Demand Drought Index - 12 month',
             'leri1': 'Landscape Evaporative Response Index - 1 month',
             'leri3': 'Landscape Evaporative Response Index - 3 month',
             'tmin': 'Average Daily Minimum Temperature (PRISM)',
             'tmax': 'Average Daily Maximum Temperature (PRISM)',
             'tmean': 'Mean Temperature (PRISM)',
             'tdmean': 'Mean Dew Point Temperature (PRISM)',
             'ppt': 'Total Precipitation (PRISM)',
             'vpdmax': 'Maximum Vapor Pressure Deficit (PRISM)',
             'vpdmin': 'Minimum Vapor Pressure Deficit (PRISM)',
             'vpdmean': 'Mean Vapor Pressure Deficit (PRISM)'}

UNIT_MAP = {'noaa': '%',
            'mdn1': '°C',
            'pdsi': 'Index',
            'scpdsi': 'Index',
            'pzi': 'Index',
            'spi1': 'Index',
            'spi2': 'Index',
            'spi3': 'Index',
            'spi4': 'Index',
            'spi5': 'Index',
            'spi6': 'Index',
            'spi7': 'Index',
            'spi8': 'Index',
            'spi9': 'Index',
            'spi10': 'Index',
            'spi11': 'Index',
            'spi12': 'Index',
            'spei1': 'Index',
            'spei2': 'Index',
            'spei3': 'Index',
            'spei4': 'Index',
            'spei5': 'Index',
            'spei6': 'Index',
            'spei7': 'Index',
            'spei8': 'Index',
            'spei9': 'Index',
            'spei10': 'Index',
            'spei11': 'Index',
            'spei12': 'Index',
            'eddi1': 'Index',
            'eddi2': 'Index',
            'eddi3': 'Index',
            'eddi4': 'Index',
            'eddi5': 'Index',
            'eddi6': 'Index',
            'eddi7': 'Index',
            'eddi8': 'Index',
            'eddi9': 'Index',
            'eddi10': 'Index',
            'eddi11': 'Index',
            'eddi12': 'Index',
            'leri1': 'Index',
            'leri3': 'Index',
            'tmin': '°C',
            'tmax': '°C',
            'tmean': '°C',
            'tdmean': '°C',
            'ppt': 'mm',
            'vpdmax': 'hPa',
            'vpdmin': 'hPa',
            'vpdmean': 'hPa'}

ACRONYM_TEXT = ("""
    INDEX/INDICATOR ACRONYMS


    PDSI:            Palmer Drought Severity Index

    SC-PDSI:         Self-Calibrating PDSI

    Palmer Z Index:  Palmer Z Index

    SPI:             Standardized Precipitation Index

    SPEI:            Standardized Precip-ET Index

    EDDI:            Evaporative Demand Drought Index

    LERI:            Landscape Evaporation Response Index

    TMIN:            Average Daily Minimum Temp (°C)

    TMAX:            Average Daily Maximum Temp (°C)

    TMEAN:           Mean Temperature (°C)

    TDMEAN:          Mean Dew Point Temperature (°C)

    PPT:             Average Precipitation (mm)

    VPDMAX:          Max Vapor Pressure Deficit (hPa)

    VPDMIN:          Min Vapor Pressure Deficit (hPa)
    """)

# Acronym "options"
ams = [{'label': 'PDSI: The Palmer Drought Severity Index (WWDT)', 'value': 0},
       {'label': 'PDSI-Self Calibrated: The Self-Calibrating Palmer Drought ' +
                 'Severity Index (WWDT)', 'value': 1},
       {'label': 'Palmer Z Index: The Palmer Z Index (WWDT)', 'value': 2},
       {'label': 'SPI: The Standardized Precipitation Index - 1 to 12 ' +
                 'months (WWDT)', 'value': 3},
       {'label': 'SPEI: The Standardized Precipitation-Evapotranspiration ' +
                 'Index - 1 to 12 months (WWDT)', 'value': 4},
       {'label': 'EDDI: The Evaporative Demand Drought Index - 1 to 12 ' +
                 'months (PSD)', 'value': 5},
       {'label': 'LERI: The Landscape Evaporative Response Index - 1 or 3 ' +
                 'months (PSD)', 'value': 6},
       {'label': 'TMIN: Average Daily Minimum Temperature ' +
                 '(°C)(PRISM)', 'value': 7},
       {'label': 'TMAX: Average Daily Maximum Temperature ' +
                 '(°C)(PRISM)', 'value': 9},
       {'label': 'TMEAN: Mean Temperature (°C)(PRISM)', 'value': 11},
       {'label': 'TDMEAN: Mean Dew Point Temperature ' +
                 '(°C)(PRISM)', 'value': 14},
       {'label': 'PPT: Average Precipitation (mm)(PRISM)', 'value': 15},
       {'label': 'VPDMAX: Maximum Vapor Pressure Deficit ' +
                 '(hPa)(PRISM)', 'value': 18},
       {'label': 'VPDMIN: Minimum Vapor Pressure Deficit ' +
                 '(hPa)(PRISM)', 'value': 20}]
