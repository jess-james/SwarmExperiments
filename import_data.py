import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4
from netCDF4 import num2date
import pickle
import xarray as xr
import re
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid


def gistemp_to_lat(str):
    if str in ['Year',  'Glob',  'NHem',  'SHem' ]:
        return str
    sign = -1
    if 'N' in str:
        sign = 1
    mag = np.sum(str.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", s))/len(str.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", s))
    return sign * mag

def import_gistemp():
    file = 'data/gistemp/ZonAnn.Ts+dSST.csv'
    nasa = pd.read_csv(file)
    nasa.columns = ['Year', 'Glob', 'NHem', 'SHem', '57', \
                        '0', '-66', '77', '54', '34', '12', '-12', '-34', '-54', '-77']

    #Remove large areas
    nasa.drop('Glob', axis=1, inplace=True)
    nasa.drop('NHem', axis=1, inplace=True)
    nasa.drop('SHem', axis=1, inplace=True)
    nasa = nasa.set_index('Year')
    nasa.to_pickle('data/nasagistemp.pkl')
    
def to_datetime(d):
    if isinstance(d, dt.datetime):
        return d
    if isinstance(d, cftime.DatetimeNoLeap):
        return dt.datetime(d.year, d.month, d.day, d.hour, d.minute, d.second)
    elif isinstance(d, cftime.DatetimeGregorian):
        return dt.datetime(d.year, d.month, d.day, d.hour, d.minute, d.second)
    elif isinstance(d, str):
        errors = []
        for fmt in (
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%SZ"):
            try:
                return dt.datetime.strptime(d, fmt)
            except ValueError as e:
                errors.append(e)
                continue
        raise Exception(errors)
    elif isinstance(d, np.datetime64):
        return d.astype(dt.datetime)
    else:
        raise Exception("Unknown value: {} type: {}".format(d, type(d)))

def get_month(d):
    return d.month

def format_mean(ta, latbin, lonbin, month ,ref):
    return ta - ref.get((latbin, lonbin, month))

def std_ta(df, ref):
      return pd.Series([ ref.at[(latbin, lonbin, month),'ta']
        for (latbin, lonbin, month) in zip(df['latbin'], df['lonbin'], df['month']) ])


def format_gcm(name):
  file = '/data/{}.nc'.format(name)
  arr = xr.open_dataset(file)
  arr = arr.drop_vars(['lon_bnds','lat_bnds','time_bnds'])
  df = arr.to_dataframe().dropna(0,'any')
  df['lat'] = df.index.get_level_values(0)
  df['lon'] = df.index.get_level_values(1)
  df['time'] = df.index.get_level_values(3)
  df.to_pickle('/data/{}_fmt.pkl'.format(name))
  
def format_anomalies(name)
  model_df = pd.read_pickle("{}_fmt.pkl".format(name))
  
  #Convert time to correct data type if necessary 
  if(model_df['time'].dtype == cftime.DatetimeNoLeap ):
      models[model_name] = model_df[(model_df['time'] > cftime.DatetimeNoLeap(1980, 1, 1)) & (model_df['time'] < cftime.DatetimeNoLeap(2000, 1, 1))]
  else:
      models[model_name] = model_df[(model_df['time'] > np.datetime64('1980-01-01')) & (model_df['time'] < np.datetime64('2000-01-01'))]
      
  #Tag locations into bins
  step = 1
  to_bin = lambda x: np.floor(x / step) * step
  model_df["latbin"] = model_df['lat'].map(to_bin)
  model_df["lonbin"] = model_df['lon'].map(to_bin)
  model_df['time'] = model_df['time'].apply(to_datetime)
  
  #Drop extraneaous dates
  model_df = model_df[(model_df['time'] > np.datetime64('1980-01-01')) & (model_df['time'] < np.datetime64('2000-01-01'))]
  print("Drop dates out of range")
  print(model_df)
  col = model_df['time'].apply(get_month)
  model_df = model_df.assign(month=col.values)

  #Calculate Temperature Anomalies
  model_df['month'] = model_df['time'].apply(get_month, as_index=False)
  ref = model_df[(model_df['time'] > np.datetime64('1980-01-01')) & (model_df['time'] < np.datetime64('1990-01-01'))].groupby(["latbin", "lonbin", "month"]).mean()
  print("Std ref")
  print(ref)
  newmodel = model_df[(model_df['time'] > np.datetime64('1990-01-01')) & (model_df['time'] < np.datetime64('2000-01-01'))]
  for lon in newmodel['lonbin'].unique():
      for lat in newmodel['latbin'].unique():
          for month in newmodel['month'].unique():
              newmodel.loc[(newmodel['latbin'] ==lat)&(newmodel['lonbin'] ==lon)&(newmodel['month'] ==month)]['std_temp'] = ref.get_group((lat, lon, month))['ta'].mean()

  newmodel = newmodel.merge(ref, on=['latbin','lonbin','month'], how='left')
  newmodel['ta_anom'] = newmodel['ta'] - newmodel['std_temp']
  newmodel.to_pickle("data/"+model_name+"_anom_month.pkl")


  
    
import_gistemp()
model_names = ["bccrbcm20", "ccmacgcm31t47", "csiromk35", "gissaom","mpiecham5", "mricgcm232a", "ncarccsm30", "cnrmcm3run1","gissmodelerrun4","ncarccsm30run9", "inmcm30", "cnrmcm3", "iapfgoals10g", "gissmodeler", "miroc32medres"]

