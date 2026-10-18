"""Unit-explicit simplified European benchmark; no empirical contract inference.

USD numeraire, frictionless spot conversion, constant USD rate and carry proxy.
An inverse payoff is USD intrinsic divided by the terminal conversion index.
One unit of notional is one BTC. Quantity, fees and collateral are separate.
"""
import numpy as np
from pricing import price,implied,greeks

def _valid_kind(kind):
    if np.any(~np.isin(np.asarray(kind),['call','put'])):raise ValueError('Expected call or put')

def premium_usd(premium,spot,currency:str,*,multiplier:float,per_contract:bool=False):
    p,s=np.broadcast_arrays(np.asarray(premium,float),np.asarray(spot,float))
    if not np.isfinite(multiplier) or multiplier<=0 or np.any(~np.isfinite(p)) or np.any(p<0) or np.any(~np.isfinite(s)) or np.any(s<=0):raise ValueError('Invalid units or values')
    if currency not in ['BTC','USD']:raise ValueError('Premium currency must be verified BTC or USD')
    return p*(s if currency=='BTC' else 1)/(multiplier if per_contract else 1)

def inverse_payoff(terminal_spot,strike,kind='call',*,multiplier:float=1.):
    _valid_kind(kind)
    s,k=np.broadcast_arrays(np.asarray(terminal_spot,float),np.asarray(strike,float))
    if np.any(~np.isfinite([s,k])) or np.any(s<=0) or np.any(k<=0) or not np.isfinite(multiplier) or multiplier<=0:raise ValueError('Invalid payoff input')
    sign=np.where(np.asarray(kind)=='call',1.,-1.)
    return multiplier*np.maximum(sign*(s-k),0)/s

def benchmark(spot,strike,T,rate,sigma,carry=0.,kind='call'):
    _valid_kind(kind)
    usd=price(spot,strike,T,rate,sigma,carry,kind)
    return dict(usd=usd,btc=usd/np.asarray(spot,float))

def forward_price(forward,strike,T,rate,sigma,kind='call'):
    """Black-76 USD value when a separately observed forward is actually available."""
    _valid_kind(kind)
    return price(forward,strike,T,rate,sigma,rate,kind)

def market_iv(observed_usd,spot,strike,T,rate,carry=0.,kind='call'):
    _valid_kind(kind)
    return implied(observed_usd,spot,strike,T,rate,carry,kind)

def rate_continuous(annual_percent):
    """Explicit semiannual-equivalent approximation, not a zero-curve bootstrap."""
    y=np.asarray(annual_percent,float)/100
    if np.any(~np.isfinite(y)) or np.any(y<=-2):raise ValueError('Invalid yield')
    return 2*np.log1p(y/2)

def maturity(trade,expiry):
    import pandas as pd
    a,b=pd.Timestamp(trade),pd.Timestamp(expiry)
    if a.tzinfo is None or b.tzinfo is None:raise ValueError('Timezone-aware timestamps required')
    return (b-a).total_seconds()/(365*86400)
