"""Independent BSM implementation. Prices in points; Greeks per unit/year."""
import math
from scipy.special import ndtr
import numpy as np

def price(S, K, T, r, sigma, q=0., kind='call'):
    S,K,T,r,sigma,q = np.broadcast_arrays(*map(lambda x:np.asarray(x,dtype=float),(S,K,T,r,sigma,q)))
    if np.any(~np.isfinite([S,K,T,r,sigma,q])) or np.any((S<=0)|(K<=0)|(T<0)|(sigma<0)):
        raise ValueError('Nonfinite or invalid pricing input')
    a=S*np.exp(-q*T); b=K*np.exp(-r*T)
    v=sigma*np.sqrt(T); safe=np.where(v>0,v,1)
    d1=(np.log(S/K)+(r-q+.5*sigma*sigma)*T)/safe; d2=d1-v
    sign=np.where(np.asarray(kind)=='call',1.,-1.)
    return np.where(v>0,sign*(a*ndtr(sign*d1)-b*ndtr(sign*d2)),np.maximum(sign*(a-b),0))

def greeks(S,K,T,r,sigma,q=0.,kind='call'):
    if min(S,K,T,sigma)<=0: raise ValueError('Greeks require interior parameters')
    d1=(math.log(S/K)+(r-q+sigma*sigma/2)*T)/(sigma*math.sqrt(T)); d2=d1-sigma*math.sqrt(T)
    a=math.exp(-q*T); b=math.exp(-r*T); n=math.exp(-d1*d1/2)/math.sqrt(2*math.pi)
    z=1 if kind=='call' else -1
    return dict(delta=z*a*ndtr(z*d1),gamma=a*n/(S*sigma*math.sqrt(T)),vega=S*a*n*math.sqrt(T),
        theta=-S*a*n*sigma/(2*math.sqrt(T))+z*q*S*a*ndtr(z*d1)-z*r*K*b*ndtr(z*d2),rho=z*K*T*b*ndtr(z*d2))

def implied(price_mid,S,K,T,r,q=0.,kind='call'):
    """Bracketed bisection, independent of vendor IV. NaN if outside bounds/bracket."""
    target,S,K,T,r,q=np.broadcast_arrays(*map(lambda x:np.asarray(x,float),(price_mid,S,K,T,r,q)))
    z=np.where(np.asarray(kind)=='call',1.,-1.)
    low=np.maximum(z*(S*np.exp(-q*T)-K*np.exp(-r*T)),0)
    high=np.where(z==1,S*np.exp(-q*T),K*np.exp(-r*T))
    ok=np.isfinite(target)&(target>low+1e-8)&(target<high)&(T>0)&(S>0)&(K>0)
    lo=np.full(target.shape,1e-6); hi=np.full(target.shape,5.)
    ok &= price(S,K,T,r,hi,q,kind)>=target
    for _ in range(60):
        mid=(lo+hi)/2; below=price(S,K,T,r,mid,q,kind)<target
        lo=np.where(below,mid,lo); hi=np.where(below,hi,mid)
    return np.where(ok,(lo+hi)/2,np.nan)
