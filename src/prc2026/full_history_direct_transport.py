"""Segmented CDN DIRECT transfer; the tiny GitHub redirect never carries payload."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import json

import shutil
import threading
import time
from datetime import datetime
from urllib.parse import urlsplit,urlunsplit,parse_qs

import urllib3

from .full_history_acquisition import ROOT,save,now,log,validate_asset_url,HTTPStop
from .adsblol_archive_range import validate_range_headers

CONFIG=ROOT/'configs/full_history_direct_transport_v1.yaml'


def segment_ranges(size,chunk):
    if size<=0 or chunk<=0:raise ValueError('Positive byte sizes required')
    return [(start,min(start+chunk,size)-1) for start in range(0,size,chunk)]


def pinned_url(url,ip,host):
    value=urlsplit(url)
    if (value.scheme!='https' or value.hostname!=host or value.username or value.password or value.fragment
            or value.port not in (None,443) or not value.path.startswith('/github-production-release-asset/')):
        raise ValueError('CDN redirect differs from registered public host')
    return urlunsplit(('https',ip,value.path,value.query,''))


def redirect_url(part,credential,*,fresh=False,stop=None):
    from scripts.acquire_july_archive_v259 import asset_session,PROXY_PORT
    validate_asset_url(part['url'])
    last_error=None
    for attempt in range(6):
        if stop is not None and stop.is_set():raise RuntimeError('Transfer stopped before metadata retry')
        try:
            with asset_session(PROXY_PORT,credential) as session:
                params={'download':'1','_prc_refresh':str(int(time.time()))} if fresh else None
                with session.get(part['url'],params=params,headers={'Cache-Control':'no-cache'},stream=True,allow_redirects=False,timeout=(15,35)) as response:
                    if response.status_code in (401,403,429):raise HTTPStop(response.status_code)
                    if response.status_code not in (301,302,303,307,308):
                        raise ValueError('Metadata request must return a redirect without reading any body')
                    url=response.headers['Location']
                    pinned_url(url,'185.199.111.133','release-assets.githubusercontent.com')
                    return url
        except HTTPStop:raise
        except ValueError:raise
        except Exception as error:
            last_error=type(error).__name__
            time.sleep(min(2*(attempt+1),10))
    raise RuntimeError('Metadata transport attempts exhausted: '+str(last_error))


class CDNLease:
    """Refresh public signed locations before expiry, never after an HTTP refusal."""
    def __init__(self,part,credential,stop=None):
        self.part=part;self.credential=credential;self.stop=stop;self.url=None;self.refresh_at=0.;self.lock=threading.Lock()
    def current(self):
        with self.lock:
            if self.stop is not None and self.stop.is_set():raise RuntimeError('Transfer stopped before metadata request')
            stamp=time.time()
            if self.url is None or stamp>=self.refresh_at:
                try:value=redirect_url(self.part,self.credential,fresh=True,stop=self.stop)
                except HTTPStop:
                    if self.stop is not None:self.stop.set()
                    raise
                expiry=parse_qs(urlsplit(value).query).get('se')
                expires=datetime.fromisoformat(expiry[0].replace('Z','+00:00')).timestamp() if expiry else stamp+1800
                if expires<stamp+120:raise RuntimeError('Public CDN returned a stale download location')
                self.url=value;self.refresh_at=min(stamp+1200,expires-120)
            return self.url


class TransferState:
    def __init__(self,folder,cfg):
        self.cfg=cfg;self.folder=folder;self.stop=threading.Event();self.lock=threading.Lock()
        self.managers={}
        self.body_bytes=sum(json.loads(p.read_text()).get('body_bytes',0) for p in folder.glob('segments/*/attempt_*.json'))
        accounting=folder/'checkpoint_accounting.json'
        adjustment=json.loads(accounting.read_text()) if accounting.exists() else {}
        self.reserved_body_bytes=int(adjustment.get('reserved_unobserved_body_bytes',0))
        recovered=int(adjustment.get('checkpoint_recovered_body_bytes',0))
        if self.reserved_body_bytes<0 or recovered<0:raise ValueError('Negative checkpoint accounting')
        self.body_bytes+=recovered
    def consume(self,n):
        with self.lock:
            if self.stop.is_set() or self.body_bytes+n+self.reserved_body_bytes>self.cfg['maximum_bulk_body_bytes']:
                self.stop.set()
                raise RuntimeError('Transfer stopped or finite byte budget reached')
            self.body_bytes+=n
    def manager(self,port,credential,ip):
        from scripts.acquire_july_archive_v259 import AuthAdapter
        with self.lock:
            if ip not in self.managers:
                self.managers[ip]=urllib3.ProxyManager(f'http://127.0.0.1:{port}',
                    proxy_headers=AuthAdapter(credential).proxy_headers(f'http://127.0.0.1:{port}'),
                    server_hostname=self.cfg['cdn_host'],assert_hostname=self.cfg['cdn_host'],cert_reqs='CERT_REQUIRED',
                    maxsize=self.cfg.get('asset_workers',1)*self.cfg.get('segments_per_asset',1),block=True)
            return self.managers[ip]
    def close(self):
        for manager in self.managers.values():manager.clear()


def fetch_segment(cfg,task,part,signed_url,port,credential,state):
    from scripts.acquire_july_archive_v259 import AuthAdapter
    index,start,stop=task;folder=state.folder/'segments'/part['name'];folder.mkdir(parents=True,exist_ok=True)
    target=folder/f'{index:04d}.bin';expected=stop-start+1
    if target.exists() and target.stat().st_size==expected:return target
    partial=folder/f'{index:04d}.partial'
    prior=sorted(folder.glob(f'attempt_{index:04d}_*.json'))
    if any(json.loads(p.read_text()).get('http_stop') for p in prior):raise RuntimeError('Prior refusal preserved')
    # A zero-byte partial previously caused an implementation FileExistsError. Preserve
    # those receipts, but do not charge them as failed network recovery attempts.
    charged=sum(json.loads(p.read_text()).get('error_type')!='FileExistsError' for p in prior)
    for attempt in range(len(prior)+1,len(prior)+1+max(0,cfg['maximum_attempts_per_segment']-charged)):
        if state.stop.is_set():raise RuntimeError('Transfer stopped')
        existing=partial.stat().st_size if partial.exists() else 0
        if existing==expected:partial.replace(target);return target
        if existing>expected:raise ValueError('Oversized segment preserved')
        ip=cfg['cdn_ipv4_candidates'][(attempt-1)%len(cfg['cdn_ipv4_candidates'])]
        location=signed_url.current() if isinstance(signed_url,CDNLease) else signed_url
        request_url=pinned_url(location,ip,cfg['cdn_host'])
        marker=folder/f'attempt_{index:04d}_{attempt:02d}.json'
        record={'started_utc':now(),'range_start':start+existing,'range_stop':stop,'body_bytes':0,
                'connect_ip':ip,'payload_route':'DIRECT','status':'running'}
        save(marker,record);response=None
        manager=state.manager(port,credential,ip)
        try:
            response=manager.request('GET',request_url,
                headers={'Host':cfg['cdn_host'],'Range':f'bytes={start+existing}-{stop}','Accept-Encoding':'identity'},
                preload_content=False,redirect=False,retries=False,
                pool_timeout=cfg.get('pool_timeout_seconds',35),
                timeout=urllib3.Timeout(connect=15,read=cfg['read_timeout_seconds']))
            if response.status in (401,403,429):raise HTTPStop(response.status)
            validate_range_headers(response.status,response.headers,start+existing,stop,part['size'])
            with partial.open('ab' if partial.exists() else 'xb') as stream:
                checkpoint=0;began=time.monotonic()
                while True:
                    # read1 returns available bytes instead of waiting to fill a MiB while
                    # a connection trickles forever below the socket inactivity timeout.
                    block=response.read1(65536,decode_content=False)
                    if not block:break
                    state.consume(len(block))
                    if stream.tell()+len(block)>expected:raise ValueError('Segment too large')
                    stream.write(block);record['body_bytes']+=len(block)
                    if record['body_bytes']-checkpoint>=4*1024*1024:
                        stream.flush();save(marker,record);checkpoint=record['body_bytes']
                        if shutil.disk_usage(folder).free<cfg['minimum_free_bytes']:raise RuntimeError('Disk reserve reached')
                    elapsed=time.monotonic()-began
                    if elapsed>cfg.get('segment_deadline_seconds',600):raise TimeoutError('Finite segment transfer deadline')
                    if elapsed>cfg.get('slow_window_seconds',90) and record['body_bytes']/elapsed<cfg.get('minimum_segment_bytes_per_second',16384):
                        raise TimeoutError('Reconnect slow DIRECT stream with preserved prefix')
            if partial.stat().st_size!=expected:raise ValueError('Truncated segment')
            partial.replace(target);record['status']='complete'
        except HTTPStop as error:
            record.update(status='failed',http_stop=error.status);state.stop.set()
        except Exception as error:
            record.update(status='failed',error_type=type(error).__name__)
        finally:
            if response:
                try:response.close()
                finally:response.release_conn()
            record['finished_utc']=now();save(marker,record)
        if record['status']=='complete':return target
        if state.stop.is_set():break
        time.sleep(min(attempt*2,10))
    raise RuntimeError('Segment attempts exhausted: '+part['name']+f' #{index}')


def transfer_part(cfg,part,port,credential,state):
    target=ROOT/part['path'];target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if target.stat().st_size!=part['size']:raise ValueError('Existing final file has wrong size')
        return {'name':part['name'],'status':'reused','bytes':part['size']}
    # Segment staging and the final raw archive can be on different volumes.
    # Reserve enough room for every concurrently assembling asset on this disk.
    required = cfg.get('target_minimum_free_bytes', cfg['minimum_free_bytes']) + part['size'] * cfg['asset_workers']
    if shutil.disk_usage(target.parent).free < required:
        state.stop.set()
        raise RuntimeError('Raw archive volume reserve reached')
    signed_url=CDNLease(part,credential,state.stop)
    ranges=segment_ranges(part['size'],cfg['segment_bytes'])
    started=time.monotonic()
    with ThreadPoolExecutor(max_workers=cfg['segments_per_asset']) as pool:
        futures=[pool.submit(fetch_segment,cfg,(i,a,b),part,signed_url,port,credential,state) for i,(a,b) in enumerate(ranges)]
        paths=[future.result() for future in futures]
    assembly=target.with_name(target.name+'.assembling')
    if shutil.disk_usage(target.parent).free < required:
        state.stop.set()
        raise RuntimeError('Raw archive volume reserve reached before assembly')
    with assembly.open('wb') as output:
        for path,(a,b) in zip(paths,ranges):
            if path.stat().st_size!=b-a+1:raise ValueError('Segment size changed before assembly')
            with path.open('rb') as source:shutil.copyfileobj(source,output,length=4*1024*1024)
    if assembly.stat().st_size!=part['size']:raise ValueError('Assembled public asset size differs')
    assembly.replace(target)
    for path in paths:path.unlink()  # Only this run's intermediate chunks; original archives stay intact.
    log(event='direct_part_complete',name=part['name'],bytes=part['size'],seconds=round(time.monotonic()-started,1))
    return {'name':part['name'],'status':'downloaded','bytes':part['size']}


def run(cfg,manifest,port,credential,folder,sha):
    state=TransferState(folder,cfg);parts=[p for d in manifest['days'] for p in d['parts']]
    if cfg['payload_route']!='DIRECT' or cfg['proxy_payload_fallback'] or not cfg['tls_hostname_verification']:
        raise ValueError('User DIRECT/TLS boundary changed')
    result={'source_git_sha':sha,'started_utc':now(),'payload_route':'DIRECT',
            'metadata_route':cfg['metadata_redirect_route'],'metadata_body_bytes':0,'expected_parts':len(parts)}
    done=[];errors=[]
    with ThreadPoolExecutor(max_workers=cfg['asset_workers']) as pool:
        pending={pool.submit(transfer_part,cfg,p,port,credential,state):p for p in parts}
        while pending:
            finished,_=wait(pending,timeout=5,return_when=FIRST_COMPLETED)
            for future in finished:
                part=pending.pop(future)
                try:done.append(future.result())
                except HTTPStop as error:
                    state.stop.set();errors.append({'part':part['name'],'error_type':'HTTPStop','http_status':error.status})
                except Exception as error:errors.append({'part':part['name'],'error_type':type(error).__name__})
            if state.stop.is_set():
                for future in pending:future.cancel()
            save(folder/'status.json',{**result,'status':'running','completed_parts':len(done),
                 'bulk_body_bytes':state.body_bytes,'errors':errors,'updated_utc':now()})
    result.update(status='files_complete_awaiting_decode' if len(done)==len(parts) else 'incomplete',
                  completed_parts=len(done),bulk_body_bytes=state.body_bytes,errors=errors,finished_utc=now())
    save(folder/'status.json',result)
    state.close()
    if len(done)!=len(parts):raise RuntimeError('DIRECT acquisition incomplete; state preserved')
    return result
