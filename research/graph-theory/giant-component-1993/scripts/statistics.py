"""Binomial Monte Carlo summaries; constants are limits, not finite-n targets."""
import math

LIMITS = {'snapshot': math.sqrt(2 / 3) * math.cosh(math.sqrt(5 / 18)),
          'trajectory': 5 * math.pi / 18}
NS = [50, 100, 500, 1000, 5000, 10000, 20000, 30000, 40000,
      50000, 60000, 70000, 80000, 90000, 100000]


def wilson(successes, trials):
    if not 0 <= successes <= trials or trials <= 0:
        raise ValueError('Invalid binomial counts')
    z = 1.959963984540054
    p = successes / trials
    d = 1 + z * z / trials
    c = (p + z * z / (2 * trials)) / d
    h = z / d * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    return (0.0 if successes == 0 else max(0, c - h),
            1.0 if successes == trials else min(1, c + h))


def summarize(row):
    p = row['successes'] / row['trials']
    limit = LIMITS[row['experiment']]
    lo, hi = wilson(row['successes'], row['trials'])
    return dict(row, probability=p, theoretical_limit=limit,
                signed_difference=p-limit, absolute_difference=abs(p-limit),
                standard_error=math.sqrt(p*(1-p)/row['trials']), ci_low=lo, ci_high=hi)
