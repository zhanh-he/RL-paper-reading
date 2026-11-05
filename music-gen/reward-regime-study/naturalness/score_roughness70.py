"""Unmodified MoSQITo DW implementation at one explicit70dB RMS SPL assumption."""
import argparse,hashlib,json,time,os,resource,platform
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import numpy as np,soundfile as sf
from mosqito.sq_metrics import roughness_dw

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(v):return dict(mean=float(np.mean(v)),p50=float(np.median(v)),p95=float(np.quantile(v,.95)),maximum=float(np.max(v)))
def one(row):
 start=time.perf_counter();path=Path(row['path']);assert sha(path)==row['excerpt_sha256'];a,sr=sf.read(path,dtype='float64',always_2d=True);x=a.mean(1);rms=float(np.sqrt(np.mean(x*x)));assert sr==48000
 if rms<1e-5:return dict(id=row['id'],audio_sha256=row['excerpt_sha256'],valid=False,reason='silent / no naturalness score',seconds=time.perf_counter()-start)
 v,spec,bark,t=roughness_dw(x*(20e-6*10**(70/20))/rms,sr,overlap=.5)
 return dict(id=row['id'],audio_sha256=row['excerpt_sha256'],valid=True,asper=stats(v),time_s=np.asarray(t).tolist(),time_asper=np.asarray(v).tolist(),specific_asper=np.asarray(spec).tolist(),bark_axis=np.asarray(bark).tolist(),seconds=time.perf_counter()-start,worker_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args();start=time.perf_counter();m=json.loads(Path(a.manifest).read_text());rows=[]
 with ProcessPoolExecutor(max_workers=a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
  for r in pool.map(one,m['samples']):
   rows.append(r);Path(a.output).write_text(json.dumps(dict(status='running',samples=rows),allow_nan=False)+'\n');print(json.dumps(dict(id=r['id'],seconds=r['seconds'])),flush=True)
 Path(a.output).write_text(json.dumps(dict(status='complete',profile=dict(method='MoSQITo roughness_dw unmodified',assumed_rms_spl_db=70,workers=a.workers,blas_threads=os.environ.get('OPENBLAS_NUM_THREADS'),total_seconds=time.perf_counter()-start,manifest_sha256=sha(a.manifest),python=platform.python_version()),samples=rows),allow_nan=False)+'\n')
if __name__=='__main__':main()
