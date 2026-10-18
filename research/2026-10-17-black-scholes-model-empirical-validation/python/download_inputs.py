"""Fetch public inputs into an explicitly external location; no package install."""
import argparse,os,urllib.request,zipfile
from pathlib import Path
import shutil

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',default=str(Path(os.environ.get('TEMP','/tmp'))/'bsm-research-20261017'));p.add_argument('--direct',action='store_true',help='Ignore a broken system proxy; retain TLS certificate verification');a=p.parse_args()
    root=Path(a.directory).resolve();project=Path(__file__).resolve().parents[1]
    if root.is_relative_to(project):raise SystemExit('Inputs must be outside the project directory')
    root.mkdir(parents=True,exist_ok=True)
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({})) if a.direct else urllib.request.build_opener()
    urls={'options_sample_2022H2.zip':'https://historicaldata.net/file/options_sample_2022H2.zip','spx.csv':'https://cdn.cboe.com/api/global/us_indices/daily_prices/SPX_History.csv','fred.csv':'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO&cosd=2022-07-01&coed=2022-12-30'}
    for name,url in urls.items():
        dest=root/name
        if dest.exists():print('Keep existing input:',name);continue
        with opener.open(url,timeout=120) as response,open(dest,'wb') as out:shutil.copyfileobj(response,out)
        print('Downloaded:',name)
    with zipfile.ZipFile(root/'options_sample_2022H2.zip') as z:
        source=(root/'source').resolve()
        for n in z.namelist():
            if not (source/n).resolve().is_relative_to(source):raise SystemExit('Unsafe ZIP member')
        z.extractall(source)
    print('Extracted outside project:',source)
    print('Read README.md, LICENSE.txt, day_by_date/manifest.json and verify.py before running the provider checks.')
if __name__=='__main__':main()
