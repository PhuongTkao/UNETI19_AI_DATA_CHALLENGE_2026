"""Unified predictor used for ALL test rows and every App input. No memorized predictions."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
import lightgbm as lgb
from model_utils import features,check_schema
from feature_count_v2 import CountEncoder
ROOT=Path(__file__).resolve().parent

def read_meta(root=ROOT):
    root=Path(root)
    return json.loads((root/'models'/'metadata.json').read_text('utf-8'))

def _hash_check(path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise ValueError('Checkpoint or frequency table checksum mismatch: '+str(path))

class Inference:
    def __init__(self,root=ROOT):
        self.root=Path(root);self.meta=read_meta(root)
        mp=self.root/'models'/self.meta['model_file']
        cp=self.root/'models'/self.meta['count_file']
        _hash_check(mp,self.meta['model_sha256']);_hash_check(cp,self.meta['count_sha256'])
        self.model=lgb.Booster(model_file=str(mp))
        self.encoder=CountEncoder.load(cp)
        if self.model.feature_name()!=self.meta['feature_names']:
            raise ValueError('Feature name mismatch')
    def predict(self,df):
        check_schema(df)
        x=pd.concat([features(df),self.encoder.transform(df)],axis=1)
        if x.columns.tolist()!=self.meta['feature_names']:
            raise ValueError('Feature ordering mismatch')
        p=np.asarray(self.model.predict(x,num_threads=2),dtype='float64')
        seconds=np.maximum(np.expm1(np.clip(p,0,12)),0.01)
        if not np.isfinite(seconds).all() or (seconds<=0).any():
            raise ValueError('Invalid prediction')
        return seconds
