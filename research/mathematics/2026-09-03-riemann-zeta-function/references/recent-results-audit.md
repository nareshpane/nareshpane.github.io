# Recent Results Audit

**Checked through: September 3, 2026.**

This record classifies each recent result used on the page. A zero-free region,
zero-density estimate, critical-line proportion, finite computation, bound on
zeta, and zero-statistics theorem answer different questions. None of the
items below proves the Riemann Hypothesis.

## Guth and Maynard: Zero Density

- **Authors:** Larry Guth and James Maynard
- **Title:** *New large value estimates for Dirichlet polynomials*
- **Year/status:** 2026, peer-reviewed, *Annals of Mathematics* 203, 623-675
- **Source:** <https://doi.org/10.4007/annals.2026.203.2.6>; arXiv:2405.20552
- **Result used:** Their estimates imply
  `N(sigma,T) <= T^(15(1-sigma)/(3+5sigma)+o(1))`; combined with Ingham's
  estimate this gives the uniform exponent `30/13`, improving Huxley's `12/5`.
- **Classification:** Unconditional zero-density estimate
- **Implication:** Among other consequences, primes are obtained asymptotically
  in intervals of length at least `x^(17/30+epsilon)`.
- **Does not imply:** The number of zeros off the critical line is zero.

## Mossinghoff, Trudgian, and Yang: Explicit Zero-Free Regions

- **Authors:** Michael J. Mossinghoff, Timothy S. Trudgian, Andrew Yang
- **Title:** *Explicit zero-free regions for the Riemann zeta-function*
- **Year/status:** 2024, peer-reviewed, *Research in Number Theory* 10
- **Source:** <https://doi.org/10.1007/s40993-023-00498-y>; arXiv:2212.06867
- **Result used:** Published explicit regions include the Korobov-Vinogradov
  denominator constant `55.241` for `|t| >= 3` and the classical denominator
  constant `5.558691` for `|t| >= 2`, in the ranges stated on the page.
- **Classification:** Unconditional explicit zero-free region, using rigorous
  finite zero verification in its explicit argument
- **Implication:** Improved explicit estimates in prime-distribution problems.
- **Does not imply:** Zero-freeness throughout `1/2 < Re(s) < 1`.

## Bellotti: Zeta Bound and Zero-Free Region

- **Author:** Chiara Bellotti
- **Title:** *Explicit bounds for the Riemann zeta function and a new zero-free region*
- **Year/status:** 2024, peer-reviewed, *Journal of Mathematical Analysis and Applications* 536, 128249
- **Source:** <https://doi.org/10.1016/j.jmaa.2024.128249>; arXiv:2306.10680
- **Result used:** The published paper gives an explicit Korobov-Vinogradov
  zero-free denominator constant `53.989` and an explicit bound
  `|zeta(sigma+it)| <= 70.7 |t|^(4.438(1-sigma)^(3/2)) log(|t|)^(2/3)`
  for the paper's stated range.
- **Previous benchmark:** This sharpens the published `55.241` constant above.
- **Classification:** Unconditional zeta bound and zero-free region
- **Does not imply:** RH or a global bound for all points in the critical strip.
- **Version caution:** arXiv v1 stated `54.004`; the page uses the sharper final
  published value `53.989`.

## Bellotti, Trudgian, and Yang: 2026 Preprint

- **Authors:** Chiara Bellotti, Timothy S. Trudgian, Andrew Yang
- **Title:** *Zero-free regions inspired by work of Heath-Brown*
- **Year/status:** 2026 preprint; no peer-reviewed publication found by cutoff
- **Source:** arXiv:2603.21490
- **Claim used:** The preprint states zero-freeness for `t >= 3` and
  `sigma >= 1 - 1/(4.896 log t)`.
- **Classification:** Unconditional claim in a preprint; classical zero-free region
- **Implication:** If validated in publication, it improves the published
  classical constant `5.558691`.
- **Does not imply:** RH; the region remains close to `Re(s)=1`.

## Pratt, Robles, Zaharescu, and Zeindler: Critical-Line Proportion

- **Authors:** Kyle Pratt, Nicolas Robles, Alexandru Zaharescu, Dirk Zeindler
- **Title:** *More than five-twelfths of the zeros of zeta are on the critical line*
- **Year/status:** 2020, peer-reviewed, *Research in the Mathematical Sciences* 7, article 2
- **Source:** <https://doi.org/10.1007/s40687-019-0199-8>; arXiv:1802.10521
- **Result used:** `liminf N_0(T)/N(T) > 0.417293`; the paper also gives
  `0.407511` for the proportion that are simple and on the line.
- **Classification:** Unconditional critical-line proportion
- **Implication:** More than 41.7 percent are known to be on the line under the
  paper's counting convention.
- **Does not imply:** The required 100 percent, or absence of even one off-line zero.

## Platt and Trudgian: Certified Computation

- **Authors:** Dave Platt and Timothy Trudgian
- **Title:** *The Riemann hypothesis is true up to 3 x 10^12*
- **Year/status:** 2021, peer-reviewed, *Bulletin of the London Mathematical Society* 53, 792-797
- **Source:** <https://doi.org/10.1112/blms.12460>; arXiv:2004.09765
- **Result used:** Rigorous interval arithmetic verifies every zero with
  `0 < Im(rho) <= 3 x 10^12` lies on the critical line.
- **Classification:** Unconditional finite computational verification
- **Implication:** A vast finite initial range satisfies RH.
- **Does not imply:** Anything about the infinitely many higher zeros.
- **Record check:** No authoritative superseding certified-height paper was
  located by the cutoff; the 2026 zero-free preprint still cites this height.

## Baluyot, Goldston, Suriajaya, and Turnage-Butterbaugh: Pair Correlation

- **Authors:** Siegfred Alan C. Baluyot, Daniel Alan Goldston, Ade Irma
  Suriajaya, Caroline L. Turnage-Butterbaugh
- **Title:** *An unconditional Montgomery theorem for pair correlation of zeros of the Riemann zeta-function*
- **Year/status:** 2024, peer-reviewed, *Acta Arithmetica* 214, 357-376
- **Source:** <https://doi.org/10.4064/aa230612-20-3>; arXiv:2306.04799
- **Result used:** An unconditional pair-correlation framework. The paper's
  61.7 percent simple-zero application requires an additional near-critical-line
  or strong zero-density hypothesis and is therefore labeled conditional.
- **Classification:** Unconditional zero-statistics theorem with a conditional application
- **Does not imply:** RH or that the zeros counted statistically lie on the line.

## Open Status

- **Authority:** Clay Mathematics Institute, *Riemann Hypothesis*
- **Source:** <https://www.claymath.org/millennium/riemann-hypothesis/>
- **Status at cutoff:** Unsolved.

## Excluded Recent Claim

The page does not promote the August 2026 preprint by Levent Alpoge and Ralph
Furman claiming that more than two thirds of the zeros are simple and on the
critical line (arXiv:2608.13637). At the cutoff it was a very recent unreviewed
claim with unusual stated proof provenance and no independent authoritative
validation located. Omitting it is preferable to presenting it as an
established replacement for the peer-reviewed 2020 benchmark.
