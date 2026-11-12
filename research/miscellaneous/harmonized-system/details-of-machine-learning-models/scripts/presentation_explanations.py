"""Presentation-only explanations drawn from existing saved arrays; no model fitting."""
import numpy as np

def transformations(d,t,one,eq,p,sig,f):
    s='<h3>From a stored observation to the fitted input</h3>'
    s+=p('Use observation 263, Canada → United States wheat, to follow the final Elastic Net pipeline. Its raw economic inputs are from 2024. A feature is one input column; the order of the columns matters. Let i label a trade observation and j label a feature. Write uᵢⱼ for a stored quantity, xᵢⱼ for the numeric feature after its specified transformation and any missing-value replacement, and zᵢⱼ for its standardized input. The transformed target y is a different object: yᵢ = log(1 + Vᵢ), where Vᵢ is trade measured in USD.')
    s+='<h4>1 / Natural logarithms: turn ratios into differences</h4>'
    s+=eq(r'x_{ij}=\log(u_{ij}),\quad u_{ij}>0,\qquad \log(ab)=\log a+\log b')
    s+=p('Here log is the natural logarithm (base e). It is the number q for which exp(q)=u. The same percentage change in GDP creates the same change in log GDP regardless of the original country size. Logs also compress the enormous range of dollar values. GDP, population and distance use this transformation; manufacturing and internet percentages do not.')
    s+=eq(r'x_{263,\mathrm{destination\ GDP}}\simeq\log('+sig(one.destination_gdp)+r')\simeq '+sig(np.log(one.destination_gdp)))
    s+=p('The argument is destination GDP in USD, not GDP converted to millions first. This log value becomes the GDP entry x in the numeric input vector. It will be centered and scaled before multiplication by the learned Elastic Net slope.')
    s+='<h4>2 / Logarithms of one plus a value: keep zeros eligible</h4>'
    s+=eq(r'x_{ij}=\log(1+u_{ij}),\quad u_{ij}\ge0,\qquad \log(1+0)=0')
    s+=p('For a trade amount or external product aggregate u in USD, the “1” is one USD. Adding it avoids the undefined log(0), while resembling log(u) for large u. External exporter supply, importer demand and outside-network world demand use log1p. The regression target also uses log1p trade, except Model 6’s explicitly positive-only size stage, which uses log V.')
    s+=eq(r'x_{263,\mathrm{external\ supply}}\simeq\log(1+'+sig(one.external_supply_usd)+r')\simeq '+sig(np.log1p(one.external_supply_usd)))
    zero=d[(d.year<2025)&d.observed_exports_usd.eq(0)].iloc[0]
    s+=p(f'Historical observation {int(zero.observation_id)} has an explicitly flagged BACI grid-zero target. Its transformed target is log(1+0)=0. This zero is retained, not removed to make logs convenient. For observation 263 the external-supply log above enters x; the unavailable 2025 outcome does not enter any fitted input.')
    s+='<h4>3 / Missing covariates: replace only the missing input</h4>'
    s+=eq(r'x_{ij}^{filled}=\begin{cases}x_{ij},&x_{ij}\text{ is available},\\m_j,&x_{ij}\text{ is missing},\end{cases}\quad m_j=\operatorname{median}_{i\in train}(x_{ij}\text{ available})')
    s+=p('The training median mⱼ is the middle available value in that feature’s training column. A missing manufacturing share is not zero manufacturing. Observation 263 has a missing exporter manufacturing share, so the final Elastic Net uses the stored training median '+sig(t['imputation_medians'][4])+'% before scaling. Missingness remains flagged in the dataset; this is an imputed explanatory input, not an observed manufacturing value or an imputed 2025 trade outcome.')
    s+='<h4>4 / Center and standardize numeric features</h4>'
    s+=eq(r'\mu_j=\frac1n\sum_{i\in train}x_{ij}^{filled},\qquad s_j=\sqrt{\frac1n\sum_{i\in train}(x_{ij}^{filled}-\mu_j)^2},\qquad z_{ij}=\frac{x_{ij}^{filled}-\mu_j}{s_j}')
    s+=p('n is the number of training rows (252 for this final Elastic Net). μⱼ is their feature mean, after median imputation. sⱼ is the population standard deviation: the software divides the sum of squared deviations by n, not n−1. The superscript “filled” identifies the pre-scaling numeric value. zᵢⱼ is the final standardized numeric input. If a training column has zero variance, the software uses scale 1 instead of dividing by zero.')
    s+=p('Subtracting μⱼ moves the training column’s mean to zero: the centered value says how far this observation lies above or below a typical training input. Dividing by sⱼ expresses that difference in training standard deviations, rather than GDP-log units, kilometres or percentage points. This makes the numeric slopes’ Elastic Net penalties more comparable. It changes the coordinates used to fit the rule, not the underlying economy.')
    x=float(np.log(one.destination_gdp));mu=t['scaling_means'][1];sd=t['scaling_sd'][1];z=t['standardized_features'][1]
    s+=p('Numerical substitutions are rounded approximations. In particular, subtracting two displayed logs can magnify the effect of rounding; the saved training means, scales and transformed values are used at full precision.')
    s+=eq(r'x_{263,j}-\mu_j\simeq '+sig(x)+'-'+sig(mu)+r'\simeq '+sig(x-mu))
    s+=eq(r'z_{263,j}\simeq\frac{'+sig(x)+'-'+sig(mu)+'}{'+sig(sd)+r'}\simeq '+sig(z)+r'\quad(j=\mathrm{destination\ GDP})')
    s+=p('Use the same training μⱼ, sⱼ and median on validation and 2025 inputs. Recomputing them on the evaluated rows would change the coordinates of a rule whose coefficients were already learned; using held-out observations to estimate preprocessing would also breach the training boundary. Different models or fitting stages can have different preprocessing values because they use different training populations.')
    s+=p('The Elastic Net intercept a is fitted separately. It is not a column passed through StandardScaler and is not penalized. Standardizing the numeric feature columns therefore does not standardize the intercept. The prediction is a plus the coefficient-weighted transformed inputs. In Model 1’s matrix notation an explicit column of ones represents its intercept; that column likewise remains one.')
    s+='<h4>5 / Binary indicators and categorical one-hot encoding</h4>'
    s+=eq(r'B_i=\mathbf1\{\text{a shared border exists}\},\qquad L_i=\mathbf1\{\text{a shared official language exists}\}')
    s+=p('The indicator symbol 1{condition} equals 1 when the condition is true and 0 otherwise. In observation 263, the GeoDist border and language indicators are both 1. They enter the numeric branch, so this implementation standardizes them too; a binary raw input need not remain 0 or 1 after scaling.')
    j=9;s+=eq(r'z_{263,\mathrm{border}}\simeq\frac{1-'+sig(t['scaling_means'][j])+'}{'+sig(t['scaling_sd'][j])+r'}\simeq '+sig(t['standardized_features'][j]))
    s+=eq(r'C_{i,10}=\mathbf1\{HS2_i=10\},\quad C_{i,87}=\mathbf1\{HS2_i=87\},\quad HS4_{263}=1001\Rightarrow(C_{263,10},C_{263,87})=(1,0)')
    s+=p('HS2 is the first two digits of the four-character HS4 code. The code is a label, not a magnitude: chapter 87 is not “8.7 times” chapter 10. One-hot encoding gives each chapter its own membership column. These two indicators pass through the categorical branch without scaling. The final Elastic Net uses both; Model 1 drops the chapter-10 indicator when including its intercept. This distinction is part of the actual fitted specification.')
    s+='<h4>6 / Assemble one vector, then stack vectors into a matrix</h4>'
    s+=eq(r'z_i^{full}=[z_{i1},\ldots,z_{i14},C_{i,10},C_{i,87}],\qquad X=\begin{bmatrix}(z_1^{full})\\\vdots\\(z_n^{full})\end{bmatrix},\qquad \widehat y_i=a+\sum_{j=1}^{16}X_{ij}b_j')
    s+=p('The first 14 entries are numeric transformed inputs, in the saved feature order; the last two are chapter memberships. A complete feature vector is one ordered row. Stacking 252 such training rows creates a 252 × 16 input matrix X: rows are observations and columns are features. bⱼ is the learned slope for column j. The vector y contains the corresponding 252 log1p targets. Observation 263 supplies a new row with exactly the same ordering, not a new set of fitted coefficients. The next table shows every entry, with display rounding only.')
    return s

def metric_notation(d,preds,eq,p,table,sig,f):
    s='<h4>Understanding the Symbols and Error Measures</h4>'
    s+=table(['Symbol','Meaning'],[
      [r'\(V_i\)','Actual trade target in USD for evaluated observation i, with disclosed BACI grid-zero assumptions retained.'],
      [r'\(\widehat V_i\)','Model-predicted trade value in USD for the same observation.'],
      [r'\(e_i=\widehat V_i-V_i\)','Signed error: positive is overprediction; negative is underprediction.'],
      [r'\(n\)','Number of observations included in the evaluation, not the training count.'],
      [r'\(\sum_i\)','Add the indicated quantity across the evaluated observations.'],
      [r'\(|e_i|\)','Absolute error: ignore the sign and retain the magnitude.'],
      [r'\(d,\;i:dest(i)=d\)','A destination and the set of evaluated cells exporting to that destination.'],
      [r'\(\overline V\)','Mean actual value in the evaluation set; used in the R² denominator.']
    ],'Notation for prediction errors','text-table')
    s+=p('<strong>MSE — Mean Squared Error.</strong> Square each dollar error and average the squares. Squaring gives larger misses disproportionately more influence: doubling an error quadruples its squared contribution. With trade in USD, MSE has USD² units; it is not an average dollar error.')
    s+=p('<strong>RMSE — Root Mean Squared Error.</strong> Take the square root of MSE. Since the square root of USD² is USD, RMSE returns to the original trade-value units while retaining the influence of large squared errors.')
    s+=p('<strong>MAE — Mean Absolute Error.</strong> Average the absolute dollar errors. A negative error and a positive error both contribute positive magnitudes, so they cannot cancel. MAE gives a more direct average size of a miss in USD.')
    s+=p('<strong>WAPE — Weighted Absolute Percentage Error.</strong> Divide total absolute dollar error by total actual trade and multiply by 100. The denominator here is ΣVᵢ, equal to Σ|Vᵢ| because trade is nonnegative. This is not the simple average of individual cell percentage errors: where Vᵢ>0, each cell’s absolute percentage error is weighted by its share of total actual trade. A zero-valued cell still contributes its absolute prediction error to the numerator; its individual percentage error is undefined. If all actual values sum to zero, WAPE is undefined and displayed as N/A.')
    s+=p('<strong>Cell WAPE and Destination WAPE.</strong> Cell WAPE takes each cell’s absolute error before summing. Destination WAPE first sums the actual and predicted cells by destination, then takes each destination total’s absolute error. Opposite signed errors within a destination can therefore cancel before Destination WAPE is calculated. The two measures answer different questions and are not interchangeable; this implementation pools all exporters within each destination.')
    s+='<h4>A short example from the saved predictions</h4>'
    g=preds[(preds.model==5)&(preds.split=='validation_2024_observed')]
    # Prefer two cells whose errors cancel after destination aggregation; these are existing outputs.
    chosen=None
    for _,group in g.groupby('destination',sort=True):
        pos=group[group.signed_error_usd>0];neg=group[group.signed_error_usd<0]
        if len(pos) and len(neg):chosen=[pos.iloc[0],neg.iloc[0]];break
    if chosen is None:chosen=[g.iloc[0],g.iloc[1]]
    a=np.array([r.observed_exports_usd for r in chosen]);pr=np.array([r.predicted_exports_usd for r in chosen]);e=pr-a
    rows=[[int(r.observation_id),r.exporter+' → '+r.destination,r.hs4_code,float(r.observed_exports_usd),float(r.predicted_exports_usd),float(r.signed_error_usd)] for r in chosen]
    s+=table(['Observation ID','Pair','HS4','Actual (USD)','Predicted (USD)','Signed error (USD)'],rows,'Two actual validation cells: single-stage gradient boosting','prediction-table')
    s+=p('Step 1: subtract actual dollars from predicted dollars to obtain the two signed errors above. Step 2: square them, add them, and divide by n=2. For compact arithmetic, let qᵢ=eᵢ/1,000,000; each q is an error in millions of USD.')
    q=e/1e6;mse=float(np.mean(q*q));mae=float(np.mean(abs(q)));wape=float(100*sum(abs(e))/sum(a));destwape=float(100*abs(sum(e))/sum(a))
    s+=eq(r'MSE\simeq\frac{('+sig(q[0])+')^2+('+sig(q[1])+r')^2}{2}\simeq '+sig(mse)+r'\ (\mathrm{million\ USD})^2')
    s+=eq(r'RMSE\simeq\sqrt{'+sig(mse)+r'}\simeq '+sig(np.sqrt(mse))+r'\ \mathrm{million\ USD},\quad MAE\simeq\frac{|'+sig(q[0])+'|+|'+sig(q[1])+r'|}{2}\simeq '+sig(mae)+r'\ \mathrm{million\ USD}')
    s+=p('Step 3: divide the sum of absolute USD errors by the sum of actual USD values; dollars cancel, giving a ratio. Multiplying by 100 expresses that ratio as a percentage. Step 4: because these two cells share a destination, add their signed errors before taking the absolute value for the destination measure.')
    s+=eq(r'Cell\ WAPE\simeq100\frac{'+sig(sum(abs(e)))+'}{'+sig(sum(a))+r'}\simeq '+sig(wape)+r'\%,\quad Destination\ WAPE\simeq100\frac{|'+sig(sum(e))+'|}{'+sig(sum(a))+r'}\simeq '+sig(destwape)+r'\%')
    s+=p('These calculations illustrate two existing rows, not the full 84-row score. The per-model sections apply the same definitions to their saved predictions; the formulas and evaluation populations are unchanged.')
    return s
