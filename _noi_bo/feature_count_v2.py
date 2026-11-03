"""Train-only frequency encoding (no labels); deterministically replayed in the App.
Fit with FIT indices only for Validation. Fit on all Train for final deployment.
"""
import numpy as np
import pandas as pd

GROUPS=('p_cell','d_cell','route','p_hour','d_hour')

def keys(df):
    arr=lambda c: np.rint(df[c].to_numpy(dtype=np.float64)*100).astype(np.int64)
    # Full globe unique geographical keys; no finer spatial resolution than raw coordinates.
    plon,plat,dlon,dlat=(arr(c) for c in ('pickup_lon_approx','pickup_lat_approx','dropoff_lon_approx','dropoff_lat_approx'))
    p=(plon+18000)*18001+(plat+9000)
    d=(dlon+18000)*18001+(dlat+9000)
    h=df['pickup_hour'].to_numpy(dtype=np.int64)
    return dict(p_cell=p,d_cell=d,route=p*1_000_000_000+d,p_hour=p*24+h,d_hour=d*24+h)

class CountEncoder:
    def __init__(self): self.tables={}
    def fit(self,df):
        self.tables={}
        for name,vals in keys(df).items():
            u,c=np.unique(vals,return_counts=True)
            self.tables[name]=(u.astype(np.int64),c.astype(np.int32))
        return self
    def transform(self,df):
        k=keys(df)
        d={}
        for name,v in k.items():
            unique,counts=self.tables[name]
            pos=np.searchsorted(unique,v)
            valid=(pos<len(unique))
            safe=np.minimum(pos,len(unique)-1)
            valid &= unique[safe]==v
            frequency=np.where(valid,counts[safe],0)
            d['log_count_'+name]=np.log1p(frequency).astype('float32')
        return pd.DataFrame(d,index=df.index)
    def save(self,path):
        content={}
        for name,(u,c) in self.tables.items(): content[name+'_keys']=u;content[name+'_counts']=c
        np.savez_compressed(path,**content)
    @classmethod
    def load(cls,path):
        a=np.load(path,allow_pickle=False);e=cls()
        e.tables={name:(a[name+'_keys'],a[name+'_counts']) for name in GROUPS}
        return e
