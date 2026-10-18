"""Strictly available UTC inputs. Daily windows contain W consecutive returns.

A day's last observed price is available as that day's close only at next UTC
midnight. Gaps are never forward-filled into volatility estimates.
"""
import numpy as np
import pandas as pd

def clean_history(history:pd.DataFrame)->pd.DataFrame:
    h=history[['timestamp','spot']].copy().sort_values('timestamp')
    if not isinstance(h.timestamp.dtype,pd.DatetimeTZDtype):raise ValueError('UTC-aware history required')
    h['timestamp']=h.timestamp.dt.tz_convert('UTC')
    if h.timestamp.isna().any() or h.timestamp.duplicated().any():raise ValueError('Missing/duplicate underlying timestamps')
    if not (np.isfinite(h.spot)&(h.spot>0)).all():raise ValueError('Invalid underlying prices')
    return h

def daily_estimators(history:pd.DataFrame,windows=(30,60))->pd.DataFrame:
    h=clean_history(history).set_index('timestamp')
    daily=h.spot.resample('1D').last()
    returns=np.log(daily/daily.shift(1))
    result=pd.DataFrame({'available_at':daily.index+pd.Timedelta(days=1),'daily_close':daily.to_numpy(),'log_return':returns.to_numpy()})
    for window in windows:
        if not isinstance(window,int) or window<2:raise ValueError('Window must be at least two returns')
        result['H'+str(window)]=(returns.rolling(window,min_periods=window).std(ddof=1)*np.sqrt(365)).to_numpy()
    return result

def match_volatility(trades:pd.DataFrame,history:pd.DataFrame,*,estimates:pd.DataFrame|None=None)->pd.DataFrame:
    if estimates is None:estimates=daily_estimators(history)
    return pd.merge_asof(trades.sort_values('timestamp'),estimates[['available_at','H30','H60']],left_on='timestamp',right_on='available_at',
        direction='backward',allow_exact_matches=True,tolerance=pd.Timedelta(days=1))

def realized_five_minute(history:pd.DataFrame,days:int=30)->pd.DataFrame:
    """Optional uncentered quadratic-variation estimate, not daily sample std.

    UTC midnight grid: 288 returns per day. Require a price at every grid point
    and W complete preceding days. Annual sigma=sqrt(365/W * sum(r_5m^2)).
    No interpolation, no hidden annualization by sqrt(252).
    """
    if not isinstance(days,int) or days<1:raise ValueError('Positive day window required')
    h=clean_history(history).set_index('timestamp')
    if (h.index!=h.index.floor('5min')).any():raise ValueError('Not an exact five-minute sample; inspect source first')
    series=h.spot.reindex(pd.date_range(h.index.min(),h.index.max(),freq='5min'))
    ret=np.log(series/series.shift(1));squared=ret**2
    # Label each (00:00, next 00:00] interval at its right boundary.
    count=squared.resample('1D',closed='right',label='right').count()
    variance=squared.resample('1D',closed='right',label='right').sum().where(count==288)
    rv=np.sqrt(variance.rolling(days,min_periods=days).sum()*365/days)
    return pd.DataFrame({'available_at':rv.index,'RV'+str(days):rv.to_numpy()})
