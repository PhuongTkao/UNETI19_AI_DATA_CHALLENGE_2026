"""Selected target-independent spatial/temporal features inspired by original AI.rar.
No extra position resolution is claimed: coordinates remain on the given .01 degree grid.
"""
import numpy as np
import pandas as pd

def old_geo_features(df):
    p_lon=df['pickup_lon_approx'].to_numpy(dtype=np.float64)
    p_lat=df['pickup_lat_approx'].to_numpy(dtype=np.float64)
    d_lon=df['dropoff_lon_approx'].to_numpy(dtype=np.float64)
    d_lat=df['dropoff_lat_approx'].to_numpy(dtype=np.float64)
    dx=(d_lon-p_lon)*84.18;dy=(d_lat-p_lat)*111.04
    angle=np.arctan2(dy,dx)
    hour=df.pickup_hour.to_numpy(dtype=np.float64)+df.pickup_minute_bucket.to_numpy(dtype=np.float64)/60
    out=pd.DataFrame(index=df.index)
    out['sin_2angle']=np.sin(2*angle)
    out['cos_2angle']=np.cos(2*angle)
    out['sin_2hour']=np.sin(4*np.pi*hour/24)
    out['cos_2hour']=np.cos(4*np.pi*hour/24)
    out['direction16']=np.floor((angle+np.pi)/(2*np.pi)*16).astype(np.float32)
    out['quarter_hour']=(hour*4).astype(np.float32)
    out['hour_group']=np.digitize(hour,[6,10,16,20]).astype(np.float32)
    for nm,lo,la,rad in [('jfk',-73.78,40.64,.03),('lga',-73.87,40.77,.02)]:
        out['pickup_near_'+nm]=((np.abs(p_lon-lo)<=rad)&(np.abs(p_lat-la)<=rad)).astype(np.float32)
        out['dropoff_near_'+nm]=((np.abs(d_lon-lo)<=rad)&(np.abs(d_lat-la)<=rad)).astype(np.float32)
    return out.astype(np.float32)
