"""Feature engineering deterministic, uses ONLY features known when trip begins.
No trip_id/trip_duration or target aggregates; usable identically on train, val, test, UI.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path

from input_validation import RAW_COLUMNS, check_schema

ANCHORS = {
    'midtown':(-73.9855,40.7580),
    'lower':(-74.0086,40.7060),
    'uptown':(-73.951,40.793),
    'brooklyn':(-73.985,40.695),
    'lga':(-73.873,40.776),
    'jfk':(-73.778,40.641),
}

def features(df: pd.DataFrame) -> pd.DataFrame:
    """Strictly per-record / deterministic features; no learned state and no label leakage."""
    check_schema(df)
    x = pd.DataFrame(index=df.index)
    for name in RAW_COLUMNS:
        if name == 'store_and_fwd_flag':
            x['store_and_fwd_Y'] = df[name].eq('Y').to_numpy(dtype='float32')
        else:
            x[name] = pd.to_numeric(df[name],errors='raise').to_numpy(dtype='float32')
    lo1 = x.pickup_lon_approx.to_numpy('float64'); la1 = x.pickup_lat_approx.to_numpy('float64')
    lo2 = x.dropoff_lon_approx.to_numpy('float64'); la2 = x.dropoff_lat_approx.to_numpy('float64')
    dx = (lo2-lo1)*111.32*np.cos(np.deg2rad((la1+la2)/2))
    dy = (la2-la1)*111.32
    km = x.straight_line_km_approx.to_numpy('float64')
    straight_geom = np.sqrt(dx*dx+dy*dy)
    x['delta_x_km'] = dx; x['delta_y_km'] = dy
    x['abs_dx'] = np.abs(dx); x['abs_dy'] = np.abs(dy)
    x['euclid_coords_km'] = straight_geom
    x['manhattan_km'] = np.abs(dx)+np.abs(dy)
    x['same_grid'] = ((lo1==lo2)&(la1==la2)).astype('float32')
    x['reported_zero_km'] = (km<0.25).astype('float32')
    x['dist_log'] = np.log1p(km)
    x['dist_sqrt'] = np.sqrt(np.maximum(km,0))
    x['dist_ratio_reported_vs_coords'] = (km+0.3)/(straight_geom+0.3)
    x['dist_ratio_manhattan'] = (np.abs(dx)+np.abs(dy)+0.3)/(km+0.3)
    x['route_direction_x'] = np.sign(dx)
    x['route_direction_y'] = np.sign(dy)
    x['route_angle_sin'] = dy/np.maximum(straight_geom,0.1)
    x['route_angle_cos'] = dx/np.maximum(straight_geom,0.1)
    for degree in (17, 29, 45):
        rad=np.deg2rad(degree)
        a=dx*np.cos(rad)+dy*np.sin(rad)
        b=-dx*np.sin(rad)+dy*np.cos(rad)
        x[f'manhattan_rot{degree}'] = np.abs(a)+np.abs(b)
        x[f'rot{degree}_east'] = a
    x['mid_lon'] = (lo1+lo2)/2
    x['mid_lat'] = (la1+la2)/2
    # These are ~1km ORIGINAL bins. Finer bins cannot recover lost precision.
    for scale in (100,50):
        p_x=np.round(lo1*scale).astype('int32'); p_y=np.round(la1*scale).astype('int32')
        d_x=np.round(lo2*scale).astype('int32'); d_y=np.round(la2*scale).astype('int32')
        x[f'p_grid_lon_{scale}']=p_x
        x[f'p_grid_lat_{scale}']=p_y
        x[f'd_grid_lon_{scale}']=d_x
        x[f'd_grid_lat_{scale}']=d_y
    h=x.pickup_hour.to_numpy('float64')+x.pickup_minute_bucket.to_numpy('float64')/60
    wd=x.pickup_dayofweek.to_numpy('float64')
    mo=x.pickup_month.to_numpy('float64')
    x['hour_float']=h
    x['minute_week']=wd*24*60+h*60
    x['is_weekend']=(wd>=5).astype('float32')
    x['is_commute'] = ((h>=7)&(h<=10)|((h>=16)&(h<=19))).astype('float32')
    for base,name in ((24,'hour'),(7,'dow'),(12,'month')):
        value = {'hour':h,'dow':wd,'month':mo}[name]
        x[f'{name}_sin'] = np.sin(2*np.pi*value/base)
        x[f'{name}_cos'] = np.cos(2*np.pi*value/base)
    x['pickup_hour_week']=wd*24+h
    for name,(lng,lat) in ANCHORS.items():
        for prefix,lons,lats in [('p',lo1,la1),('d',lo2,la2)]:
            ix=(lons-lng)*111.32*np.cos(np.deg2rad((lats+lat)/2))
            iy=(lats-lat)*111.32
            x[f'{prefix}_to_{name}'] = np.sqrt(ix*ix+iy*iy)
    # Distinct route via landmark; derived only from coordinates
    for name in ('midtown','lower','lga','jfk'):
        x[f'via_{name}']=x[f'p_to_{name}']+x[f'd_to_{name}']
    for col in x.columns:
        x[col] = x[col].astype('float32')
    if x.isna().any().any() or not np.isfinite(x.to_numpy()).all():
        raise ValueError('Nonfinite engineered features')
    return x


def predict_seconds(model: lgb.Booster | object, df: pd.DataFrame, feature_names:list[str]) -> np.ndarray:
    x=features(df)
    if list(x.columns)!=list(feature_names):
        raise ValueError('Feature schema mismatch between training and prediction')
    raw=np.asarray(model.predict(x),dtype=np.float64)
    # clip in log-space for numerical stability; never negative predictions
    return np.maximum(np.expm1(np.clip(raw,0,12)),0.01)


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y=np.asarray(y_true,dtype=np.float64)
    p=np.asarray(y_pred,dtype=np.float64)
    if not np.isfinite(p).all() or (p<0).any(): raise ValueError('Invalid prediction')
    return float(np.sqrt(np.mean((np.log1p(p)-np.log1p(y))**2)))
