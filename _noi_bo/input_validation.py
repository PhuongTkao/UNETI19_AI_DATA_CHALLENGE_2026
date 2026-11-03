"""Strict, model-safe validation of inputs known at pickup time.

Do not fit/learn any values from Test.  Geographic range is a deliberately broad
NYC-area input sanity bound, not a claim about the precision of rounded GPS.
Training and Test must both pass the same checks.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

RAW_COLUMNS = [
    'vendor_id','pickup_month','pickup_dayofweek','pickup_hour',
    'pickup_minute_bucket','passenger_count','pickup_lon_approx',
    'pickup_lat_approx','dropoff_lon_approx','dropoff_lat_approx',
    'straight_line_km_approx','store_and_fwd_flag'
]
INTEGRAL = ('vendor_id','pickup_month','pickup_dayofweek','pickup_hour',
            'pickup_minute_bucket','passenger_count')
RANGES = {
    'vendor_id':(1,2),
    'pickup_month':(1,6),  # dataset spans Jan–Jun only; do not extrapolate silently
    'pickup_dayofweek':(0,6),
    'pickup_hour':(0,23),
    'passenger_count':(1,6),
    'pickup_lon_approx':(-75.0,-72.0),
    'dropoff_lon_approx':(-75.0,-72.0),
    'pickup_lat_approx':(40.0,42.0),
    'dropoff_lat_approx':(40.0,42.0),
    'straight_line_km_approx':(0,45),
}

def check_schema(df:pd.DataFrame, labelled:bool=False)->None:
    if not isinstance(df,pd.DataFrame):
        raise ValueError('Input must be a pandas DataFrame')
    absent=sorted(set(RAW_COLUMNS)-set(df.columns))
    if absent:
        raise ValueError(f'Missing features: {absent}')
    if not df.columns.is_unique:
        raise ValueError('Duplicate input columns')
    if 'trip_id' in df:
        if df['trip_id'].isna().any() or df['trip_id'].astype(str).duplicated().any():
            raise ValueError('Missing or duplicate trip_id')
    if labelled and 'trip_duration' not in df:
        raise ValueError('Missing target trip_duration')
    if df.loc[:,RAW_COLUMNS].isna().any().any():
        raise ValueError('Missing input feature(s)')
    for col in RAW_COLUMNS:
        if col=='store_and_fwd_flag':
            if not df[col].isin(['N','Y']).all():
                raise ValueError(f'{col} must be exactly N or Y')
            continue
        try:
            num=pd.to_numeric(df[col],errors='raise').to_numpy(dtype='float64')
        except (TypeError,ValueError,OverflowError) as exc:
            raise ValueError(f'{col} is not numeric') from exc
        if not np.isfinite(num).all():
            raise ValueError(f'{col} must be finite')
        if col in INTEGRAL and not np.all(num==np.floor(num)):
            raise ValueError(f'{col} must be an integer')
        if col=='pickup_minute_bucket' and not np.isin(num,[0,15,30,45]).all():
            raise ValueError(f'{col} must be one of 0/15/30/45')
        if col in RANGES:
            lo,hi=RANGES[col]
            if ((num<lo)|(num>hi)).any():
                raise ValueError(f'{col} outside valid range [{lo}, {hi}]')
    if labelled:
        try:
            target=pd.to_numeric(df['trip_duration'],errors='raise').to_numpy(dtype='float64')
        except (TypeError,ValueError,OverflowError) as exc:
            raise ValueError('trip_duration must be numeric') from exc
        if not np.isfinite(target).all() or (target<=0).any():
            raise ValueError('trip_duration must be finite and positive')
