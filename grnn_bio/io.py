"""Portable JSON/NPZ artifacts. No pickle, hidden cache, or silent overwrites."""
from __future__ import annotations
import hashlib,json,os,platform,sys
from pathlib import Path
from datetime import datetime,timezone
import numpy as np


def jsonable(obj):
    if isinstance(obj,dict): return {str(k):jsonable(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)): return [jsonable(v) for v in obj]
    if isinstance(obj,np.ndarray): return jsonable(obj.tolist())
    if isinstance(obj,np.generic): return jsonable(obj.item())
    if isinstance(obj,float) and not np.isfinite(obj): return None
    if isinstance(obj,Path): return str(obj)
    return obj


def write_json(path,obj):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(jsonable(obj),ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
    _replace(tmp,p)


def _replace(tmp,target,attempts=50):
    """os.replace with retries: on Windows a reader holding the target open
    (indexer, antivirus, a progress script) briefly blocks the rename."""
    import time
    for i in range(attempts):
        try:
            os.replace(tmp,target);return
        except PermissionError:
            if i==attempts-1: raise
            time.sleep(.05*(i+1))


def sha256_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''): h.update(chunk)
    return h.hexdigest()


def object_hash(obj):
    return hashlib.sha256(json.dumps(jsonable(obj),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def source_fingerprint(root):
    return object_hash({str(p.relative_to(root)):sha256_file(p) for p in sorted((Path(root)/'grnn_bio').glob('*.py'))})


def environment():
    import scipy,pandas
    return {'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,
            'scipy':scipy.__version__,'pandas':pandas.__version__,
            'utc_time':datetime.now(timezone.utc).isoformat(),
            'blas_thread_limits':{k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']}}


def save_npz(path,**arrays):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.npz.tmp')
    with tmp.open('wb') as f: np.savez_compressed(f,**arrays)
    _replace(tmp,path)


def save_checkpoint(path,circuit,optimizer,meta):
    arrays={'model__'+k:v for k,v in circuit.state_dict().items()}
    opt=optimizer.state_dict()
    metadata={k:v for k,v in opt.items() if not isinstance(v,np.ndarray)}
    arrays.update({'optimizer__'+k:v for k,v in opt.items() if isinstance(v,np.ndarray)})
    arrays['metadata_json']=np.array(json.dumps(jsonable({'optimizer':metadata,'run':meta}),sort_keys=True))
    save_npz(path,**arrays)


def load_checkpoint(path,circuit,optimizer,expected_config_hash):
    with np.load(path,allow_pickle=False) as raw:
        meta=json.loads(str(raw['metadata_json']))
        if meta['run']['config_hash']!=expected_config_hash:
            raise ValueError('Checkpoint config hash mismatch; refusing an inconsistent resume')
        circuit.load_state_dict({k[7:]:raw[k] for k in raw.files if k.startswith('model__')})
        opt=meta['optimizer']
        opt.update({k[11:]:raw[k] for k in raw.files if k.startswith('optimizer__')})
        optimizer.load_state_dict(opt)
    return meta['run']
