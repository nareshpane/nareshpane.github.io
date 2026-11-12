"""Publication formatting only: these strings never enter model calculations."""
import math

def ordinary(value, decimals=1):
    if value is None or not math.isfinite(float(value)):
        return 'N/A'
    return format(float(value), f',.{decimals}f')

def informative(value):
    """One decimal, retaining the sign and scale of a small nonzero quantity."""
    x=float(value)
    return format(x,'.1e') if 0<abs(x)<.05 else ordinary(x)

def probability(value):
    x=float(value)
    if .95<x<1:
        return '1 − '+format(1-x,'.1e')
    return informative(x)

def percentage(value, *, proportion=False):
    return ordinary(float(value)*(100 if proportion else 1))+'%'

def integer(value):
    return ordinary(value,0)
