import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
import numpy as np
import pandas as pd
from scipy.special import ndtr
from pricing import price,greeks,implied
from analyze import metrics,quote_mask

class PricingTests(unittest.TestCase):
    def test_reference_and_parity(self):
        self.assertAlmostEqual(float(price(100,100,1,.05,.2)),10.450583572185565,places=10)
        for s in [50,100,180]:
            self.assertAlmostEqual(float(price(s,100,.7,.04,.3,.02)-price(s,100,.7,.04,.3,.02,'put')),s*np.exp(-.02*.7)-100*np.exp(-.04*.7),places=10)
    def test_limits(self):
        self.assertEqual(price(100,110,0,.05,.2),0)
        self.assertAlmostEqual(float(price(100,100,1,.05,0)),100-100*np.exp(-.05))
        self.assertLess(float(price(100,100,1,.05,.2)),float(price(101,100,1,.05,.2)))
        self.assertLess(float(price(100,100,1,.05,.2)),float(price(100,100,1,.05,.3)))
    def test_cdf(self):
        self.assertEqual(ndtr(0),.5); self.assertAlmostEqual(ndtr(2)+ndtr(-2),1)
        self.assertTrue(np.all(np.diff(ndtr(np.linspace(-10,10,100)))>=0))
    def test_bad_inputs(self):
        for s in [-1,0,np.nan]:
            with self.assertRaises(ValueError): price(s,100,1,.03,.2)
        self.assertTrue(np.isnan(implied(-1,100,100,1,.03)))
        self.assertTrue(np.isnan(implied(np.nan,100,100,1,.03)))
    def test_greeks_finite_difference(self):
        for kind in ['call','put']:
            x=[102.,100.,.7,.04,.3,.015]; g=greeks(*x,kind)
            for name,i,sign in [('delta',0,1),('theta',2,-1),('rho',3,1),('vega',4,1)]:
                h=1e-4; a=x.copy(); b=x.copy(); a[i]+=h;b[i]-=h
                fd=sign*(price(*a,kind)-price(*b,kind))/(2*h)
                self.assertAlmostEqual(float(fd),g[name],places=5)
            h=.01; fd=(price(x[0]+h,*x[1:],kind)-2*price(*x,kind)+price(x[0]-h,*x[1:],kind))/h**2
            self.assertAlmostEqual(float(fd),g['gamma'],places=7)
    def test_inversion(self):
        for k in [80,100,120]:
            for kind in ['call','put']:
                p=price(100,k,.8,.03,.27,.02,kind)
                self.assertAlmostEqual(float(implied(p,100,k,.8,.03,.02,kind)),.27,places=9)
    def test_filters(self):
        d=pd.DataFrame(dict(bid=[9,0,11,9,np.nan],ask=[10,1,10,10,10],mid=[9.5,.5,10.5,9.5,10],bid_size=[1]*5,ask_size=[1]*5))
        self.assertEqual(quote_mask(d,.2).tolist(),[True,False,False,True,False])
        self.assertEqual(int(quote_mask(d,.05).sum()),0)
    def test_metrics(self):
        d=pd.DataFrame(dict(mid=[10,20],A=[12,18],S=[100,100],bid=[9,19],ask=[11,21],quote_date=['a','b']))
        m=metrics(d,'A'); self.assertEqual(m['mae'],2);self.assertEqual(m['rmse'],2);self.assertEqual(m['bias'],0);self.assertEqual(m['within'],0);self.assertEqual(m['n'],2)

if __name__=='__main__': unittest.main()
