"""Single production predictor: all batch CSV and Streamlit calls use this class.
The decision whether to blend is LOCKED in models/v3_metadata.json. No test memorization.
"""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
import lightgbm as lgb
from predictor_v2 import Inference as V2Inference
from feature_count_v2 import CountEncoder
from model_utils import features,check_schema
from old_geo_features import old_geo_features
ROOT=Path(__file__).resolve().parent

def checksum(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def as_seconds(z):
    z=np.asarray(z,dtype='float64')
    if not np.isfinite(z).all():raise ValueError('Non-finite model output')
    return np.maximum(np.expm1(np.clip(z,0,12)),0.01)

class InferenceV3:
    def __init__(self,root=ROOT):
        self.root=Path(root)
        self.meta=json.loads((self.root/'models/v3_metadata.json').read_text(encoding='utf8'))
        self.v2=V2Inference(root)
        self.weights=self.meta['weights']
        self.encoder=self.v2.encoder
        self.geo=None;self.cat=None
        if 'geo' in self.weights:
            f=self.root/'models'/self.meta['geo_model_file']
            if checksum(f)!=self.meta['geo_sha256']:raise ValueError('Geo checkpoint SHA256 does not match')
            self.geo=lgb.Booster(model_file=str(f))
        if 'cat' in self.weights:
            f=self.root/'models'/self.meta['cat_model_file']
            if checksum(f)!=self.meta['cat_sha256']:raise ValueError('CatBoost checkpoint SHA256 does not match')
            from catboost import CatBoostRegressor
            self.cat=CatBoostRegressor()
            self.cat.load_model(str(f))
        if abs(sum(self.weights.values())-1)>1e-10 or min(self.weights.values())<0:raise ValueError('Bad blend weights')
    def predict(self,df):
        check_schema(df)
        p={'v2':self.v2.predict(df)}
        if 'cat' in self.weights or 'geo' in self.weights:
            X=pd.concat([features(df),self.encoder.transform(df)],axis=1)
            if 'cat' in self.weights:
                assert list(X.columns)==self.meta['cat_features']
                p['cat']=as_seconds(self.cat.predict(X.to_numpy(dtype='float32'),thread_count=2))
            if 'geo' in self.weights:
                X=pd.concat([X,old_geo_features(df)],axis=1)
                if list(X.columns)!=self.geo.feature_name():raise ValueError('Geo features mismatch')
                p['geo']=as_seconds(self.geo.predict(X,num_threads=2))
        out=np.expm1(sum(w*np.log1p(p[n]) for n,w in self.weights.items()))
        if not np.isfinite(out).all() or (out<=0).any():raise ValueError('Invalid final prediction')
        return out
