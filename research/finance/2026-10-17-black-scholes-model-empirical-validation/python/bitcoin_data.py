"""Rights-gated official CSV access and explicit-schema adapters. No scraper.

No file-column mapping below is claimed to describe the actual UWA attachments.
Rights records document human-reviewed evidence; a Boolean cannot grant rights.
"""
from dataclasses import dataclass
from datetime import datetime,timezone
from pathlib import Path
import csv,hashlib,json,re,time,codecs
from urllib.request import Request,build_opener,HTTPRedirectHandler
from urllib.parse import urlparse
from urllib.error import HTTPError,URLError
import numpy as np
import pandas as pd

FILES=('instrument_name.csv','Deribit_w_IV.csv','Price_5m.csv','Deribit_BTC_option_all.rds')

def require_rights(record:dict,publication:bool=False)->None:
    for key in ['processing']+(['publication'] if publication else []):
        entry=record.get(key,{})
        if entry.get('authorized') is not True or not entry.get('evidence_reference') or not entry.get('reviewed_by') or not entry.get('reviewed_on'):
            raise PermissionError('Documented '+key+' authorization required; CC BY label alone does not settle the review')

def utc_times(values,*,source_timezone:str|None=None,unit:str|None=None)->pd.Series:
    s=pd.Series(values).copy()
    if unit is not None:
        if unit not in ['s','ms','us','ns']:raise ValueError('Unsupported epoch unit')
        return pd.to_datetime(pd.to_numeric(s,errors='coerce'),unit=unit,utc=True,errors='coerce')
    if pd.api.types.is_numeric_dtype(s):raise ValueError('Epoch unit must be supplied, never guessed')
    if s.dropna().astype(str).str.contains(r'(?:Z|[+-]\d{2}:?\d{2})$',regex=True).all():
        return pd.to_datetime(s,utc=True,format='mixed',errors='coerce')
    if source_timezone is None:raise ValueError('Naive timestamps require verified source timezone')
    parsed=pd.to_datetime(s,format='mixed',errors='coerce')
    if isinstance(parsed.dtype,pd.DatetimeTZDtype):return parsed.dt.tz_convert('UTC')
    return parsed.dt.tz_localize(source_timezone,ambiguous='raise',nonexistent='raise').dt.tz_convert('UTC')

@dataclass(frozen=True)
class Instrument:
    symbol:str
    strike:float
    kind:str
    expiration:pd.Timestamp
    # Exercise style and multiplier are NOT inferred from an identifier.

def parse_instrument(symbol:str,*,expiry_hour_utc:int=8)->Instrument:
    m=re.fullmatch(r'BTC-(\d{1,2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{2})-(\d+(?:\.\d+)?)-([CP])',str(symbol))
    if not m or not 0<=expiry_hour_utc<=23:raise ValueError('Unsupported BTC option identifier')
    day,month,year,strike,side=m.groups();k=float(strike)
    if not np.isfinite(k) or k<=0:raise ValueError('Positive strike required')
    months=['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
    expiry=pd.Timestamp(datetime(2000+int(year),months.index(month)+1,int(day),expiry_hour_utc,tzinfo=timezone.utc))
    return Instrument(str(symbol),k,'call' if side=='C' else 'put',expiry)

def match_underlying(trades:pd.DataFrame,history:pd.DataFrame,max_lag='5min')->pd.DataFrame:
    from bitcoin_volatility import clean_history
    h=clean_history(history).rename(columns={'timestamp':'underlying_at'})
    if trades.timestamp.isna().any():raise ValueError('Filter invalid trade timestamps before matching')
    result=pd.merge_asof(trades.sort_values('timestamp'),h,left_on='timestamp',right_on='underlying_at',direction='backward',
        allow_exact_matches=True,tolerance=pd.Timedelta(max_lag))
    result['lag_seconds']=(result.timestamp-result.underlying_at).dt.total_seconds()
    if (result.lag_seconds.dropna()<0).any():raise AssertionError('Future underlying match')
    return result

def file_sha256(path:Path)->str:
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def official_url(url:str)->None:
    u=urlparse(url)
    if u.scheme!='https' or not u.hostname or not (u.hostname=='research-repository.uwa.edu.au' or u.hostname.endswith('.uwa.edu.au')) or u.username or u.password:
        raise ValueError('Only reviewed HTTPS university attachment hosts are permitted; inspect unexpected redirects manually')

class AcademicRedirects(HTTPRedirectHandler):
    def __init__(self):self.chain=[]
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        official_url(newurl);self.chain.append({'status':code,'from':req.full_url,'to':newurl})
        if len(self.chain)>5:raise ValueError('Too many redirects')
        return super().redirect_request(req,fp,code,msg,headers,newurl)

def download_csv(url:str,destination:Path,rights:dict,*,attempts:int=2)->dict:
    """Stream to .part, at most one transient retry; no auth bypass or scraping.

    HTTP 401/403/404, HTML/login responses and non-university redirects stop.
    Failed .part files remain for inspection; an existing original is not replaced.
    """
    require_rights(rights);official_url(url)
    destination=Path(destination)
    if destination.name not in FILES[:3] or attempts not in [1,2]:raise ValueError('Unexpected filename or retry count')
    if destination.exists() or destination.with_suffix('.csv.part').exists():raise FileExistsError('Inspect existing original/partial file first')
    destination.parent.mkdir(parents=True,exist_ok=True)
    part=destination.with_suffix('.csv.part');redirects=AcademicRedirects();opener=build_opener(redirects)
    for attempt in range(attempts):
        try:
            with opener.open(Request(url,headers={'User-Agent':'AcademicResearchDownload/1.0'}),timeout=40) as response:
                if response.status!=200:raise ValueError('Expected complete HTTP 200 attachment')
                first=response.read(65536);sniff=first.lstrip().lower()
                if b'text/html' in response.headers.get('Content-Type','').lower().encode() or sniff.startswith((b'<!doctype html',b'<html')):raise ValueError('HTML/login response, not CSV')
                with part.open('wb') as stream:
                    stream.write(first)
                    for block in iter(lambda:response.read(1024*1024),b''):stream.write(block)
                size=part.stat().st_size;length=response.headers.get('Content-Length')
                if size==0 or (length and int(length)!=size):raise ValueError('Empty/truncated attachment')
                receipt=dict(source_url=url,final_url=response.url,status=response.status,redirects=redirects.chain,bytes=size,
                    content_type=response.headers.get('Content-Type'),retrieved_utc=datetime.now(timezone.utc).isoformat(),sha256=file_sha256(part))
            part.replace(destination)
            destination.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
            return receipt
        except (HTTPError,URLError,TimeoutError) as error:
            transient=isinstance(error,HTTPError) and error.code in [429,500,502,503,504] or isinstance(error,TimeoutError) or isinstance(getattr(error,'reason',None),TimeoutError)
            if not transient or attempt+1==attempts:raise
            time.sleep(2)
    raise AssertionError('Unreachable')

def inspect_csv(path:Path,rights:dict,*,timestamp_field=None,contract_field=None,id_field=None,source_timezone=None,unit=None,chunksize=100000)->dict:
    """Chunked descriptive inventory only; semantics still require source evidence."""
    require_rights(rights);path=Path(path)
    with path.open('rb') as stream:prefix=stream.read(65536)
    encoding='utf-8-sig'
    try:codecs.getincrementaldecoder(encoding)(errors='strict').decode(prefix,final=False)
    except UnicodeDecodeError:raise ValueError('Not UTF-8; document and explicitly implement the actual encoding')
    preview=prefix.decode(encoding,errors='ignore').lstrip().lower()
    if preview.startswith(('<!doctype html','<html','<head','<body')):raise ValueError('HTML/access page is not a CSV dataset')
    with path.open(encoding=encoding,newline='') as stream:header=next(csv.reader(stream),None)
    if not header:raise ValueError('Empty CSV; no header verified')
    if len(header)!=len(set(header)):raise ValueError('Duplicate CSV headings require an explicit adapter')
    if any(field and field not in header for field in [timestamp_field,contract_field,id_field]):raise ValueError('Requested inventory field is absent from actual CSV header')
    count=0;missing={k:0 for k in header};types={k:set() for k in header};ids=set();contracts=set();dates=set();duplicates=0;earliest=latest=None;calls=puts=unparsed=0
    for chunk in pd.read_csv(path,encoding=encoding,chunksize=chunksize,dtype={k:'string' for k in [id_field,contract_field] if k}):
        count+=len(chunk)
        for k in header:missing[k]+=int(chunk[k].isna().sum());types[k].add(str(chunk[k].dtype))
        if id_field:
            for value in chunk[id_field].dropna().astype(str):
                if value in ids:duplicates+=1
                ids.add(value)
        if contract_field:
            for value,n in chunk[contract_field].dropna().value_counts().items():
                contracts.add(str(value))
                try:
                    instrument=parse_instrument(str(value));calls+=int(n) if instrument.kind=='call' else 0;puts+=int(n) if instrument.kind=='put' else 0
                except ValueError:unparsed+=int(n)
        if timestamp_field:
            times=utc_times(chunk[timestamp_field],source_timezone=source_timezone,unit=unit).dropna()
            if len(times):
                earliest=min(earliest,times.min()) if earliest is not None else times.min();latest=max(latest,times.max()) if latest is not None else times.max();dates.update(times.dt.strftime('%Y-%m-%d'))
    return dict(file=path.name,bytes=path.stat().st_size,sha256=file_sha256(path),encoding='utf-8-sig' if prefix.startswith(codecs.BOM_UTF8) else 'utf-8',rows=count,columns=header,column_count=len(header),
        storage_types={k:sorted(v) for k,v in types.items()},missing_percent={k:100*v/count if count else None for k,v in missing.items()},
        earliest=str(earliest) if earliest is not None else None,latest=str(latest) if latest is not None else None,dates=len(dates) if timestamp_field else None,
        contracts=len(contracts) if contract_field else None,calls=calls if contract_field else None,puts=puts if contract_field else None,unparsed=unparsed if contract_field else None,
        duplicate_ids=duplicates if id_field else None,price_semantics='Must be supplied and verified separately; no inference from a column name')
