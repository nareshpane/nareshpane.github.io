"""Direct seven-day LSTM forecasts with chronological calibration and baselines.

Run: python 03_time_series_forecasting.py --quick --output ../outputs/forecasting
All electricity demand is simulated, in MWh/day. Future realized weather is NEVER
an input. Independent program: its NumPy numerical functions need no TensorFlow.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
from pathlib import Path
import platform
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np


# SECTION 1 — Explicit, additive synthetic process (not a universal decomposition).
def generate_series(seed: int = 42, n: int = 2190):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    weather_deviation = np.zeros(n)
    disturbance = np.zeros(n)
    for i in range(1, n):
        weather_deviation[i] = .70 * weather_deviation[i-1] + rng.normal(0, 2.)
        disturbance[i] = .65 * disturbance[i-1] + rng.normal(0, 25.)
    temperature = 12 + 11 * np.sin(2*np.pi*(t-80)/365.25) + weather_deviation
    components = {
        "trend": 1500 + .12 * t,
        "weekly": np.where(t.astype(int) % 7 < 5, 48., -120.),
        "annual": 90 * np.cos(2*np.pi*t/365.25),
        "weather": 18*np.maximum(10-temperature, 0) + 14*np.maximum(temperature-20, 0),
        "noise": disturbance,
        "shocks": (rng.random(n) < .008) * rng.choice([-1., 1.], n) * rng.uniform(100, 220, n),
    }
    demand = np.sum(np.stack(list(components.values())), axis=0)
    # No clipping: these documented parameters produce positive synthetic demand.
    if demand.min() <= 0:
        raise ValueError("A modified generator produced nonpositive demand; revisit the process")
    return demand, temperature, components


def boundaries(n):
    return {"train_end": int(.60*n), "validation_end": int(.75*n),
            "calibration_end": int(.85*n), "n": n}


def input_features(demand, temperature, train_end):
    """Fit means/scales ONLY on training observations; past demand/temp + calendar."""
    t = np.arange(len(demand))
    observed = np.column_stack([demand, temperature])
    mean, scale = observed[:train_end].mean(axis=0), observed[:train_end].std(axis=0)
    scale = np.maximum(scale, 1e-8)
    standardized = (observed-mean)/scale
    features = np.column_stack([standardized, np.sin(2*np.pi*t/365.25), np.cos(2*np.pi*t/365.25),
                                np.sin(2*np.pi*t/7), np.cos(2*np.pi*t/7)]).astype(np.float32)
    return features, mean, scale


# SECTION 2 — Origin o is the FIRST unobserved index (the first target index).
def make_windows(features, demand, lookback, horizon, target_start, target_end, stride=1):
    """All targets lie in [target_start,target_end); context may precede that interval.

    X[o]=features[o-L:o], Y[o]=demand[o:o+H]. Hence every input timestamp
    is <o and every target timestamp is >=o. Default daily origins overlap.
    """
    origins = np.arange(max(lookback, target_start), target_end-horizon+1, stride, dtype=int)
    if not len(origins):
        raise ValueError("Not enough observations for the requested window")
    history_indices = origins[:, None] - np.arange(lookback, 0, -1)
    target_indices = origins[:, None] + np.arange(horizon)
    assert np.all(history_indices < origins[:, None])
    assert np.all(target_indices >= origins[:, None])
    assert target_indices.min() >= target_start and target_indices.max() < target_end
    return features[history_indices], np.asarray(demand)[target_indices], origins


def baseline_forecasts(demand, origins, horizon=7):
    if horizon > 7:
        raise ValueError("This seasonal baseline is implemented for horizons up to seven days")
    last = np.repeat(np.asarray(demand)[origins-1, None], horizon, axis=1)
    seasonal = np.asarray(demand)[origins[:, None] + np.arange(horizon) - 7]
    return last, seasonal


def conformal_quantiles(absolute_errors, alpha=.2):
    """Horizonwise exact finite-sample rank, with no interpolated quantile.

    This is an algorithm borrowed from split conformal, not a coverage theorem for
    this dependent time series. The calibration set must not fit the point model.
    """
    errors = np.asarray(absolute_errors, dtype=np.float64)
    if errors.ndim != 2 or not 0 < alpha < 1 or not np.isfinite(errors).all():
        raise ValueError("Need a finite [n_origins,horizon] score matrix and 0<alpha<1")
    count = len(errors)
    rank = math.ceil((count+1)*(1-alpha))
    if rank > count:
        return np.full(errors.shape[1], np.inf)
    return np.sort(errors, axis=0)[rank-1]


def forecast_metrics(actual, predicted, mase_scale, lower=None, upper=None):
    error = np.asarray(predicted) - np.asarray(actual)
    metrics = {"mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.mean(error**2))),
               "mase": float(np.abs(error).mean()/mase_scale), "mase_training_scale": float(mase_scale),
               "by_horizon_mae": np.abs(error).mean(axis=0).tolist(),
               "by_horizon_rmse": np.sqrt(np.mean(error**2, axis=0)).tolist(),
               "mean_error": float(error.mean()), "by_horizon_bias": error.mean(axis=0).tolist()}
    if lower is not None and upper is not None:
        covered = (actual >= lower) & (actual <= upper)
        metrics.update(coverage=float(covered.mean()), by_horizon_coverage=covered.mean(axis=0).tolist(),
                       mean_interval_width=float(np.mean(upper-lower)))
    return metrics


def autocorrelation(values, max_lag=28):
    centered = np.asarray(values) - np.mean(values)
    denominator = float(centered @ centered)
    return [1.] + [float(centered[k:] @ centered[:-k] / denominator) if denominator else 0.
                   for k in range(1, max_lag+1)]


def write_forecast_svg(path, actual, neural, last, seasonal, lower, upper):
    """One truly held-out seven-step forecast; all chart points come from arrays."""
    width, height = 760, 360
    minimum = min(float(np.min(lower)), float(np.min(actual))) - 30
    maximum = max(float(np.max(upper)), float(np.max(actual))) + 30
    px = lambda i: 70 + i*100
    py = lambda value: 290 - (value-minimum)/(maximum-minimum)*220
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 360" role="img"><title>Held-out electricity forecast in MWh per day</title><rect width="760" height="360" fill="#fff8f2"/><g font-family="sans-serif" font-size="13" fill="#26332f"><text x="35" y="25">Held-out seven-day forecast • MWh/day</text>']
    polygon = " ".join(f"{px(i)},{py(lower[i]):.2f}" for i in range(7)) + " " + " ".join(f"{px(i)},{py(upper[i]):.2f}" for i in reversed(range(7)))
    svg.append(f'<polygon points="{polygon}" fill="#d7e8d3"/>')
    for values, color, name, row in [(actual, "#26332f", "Observed", 0),(neural,"#427d49","LSTM",1),
                                      (last,"#b6773a","Last value",2),(seasonal,"#5b75a2","Seasonal naive",3)]:
        points = " ".join(f"{px(i)},{py(value):.2f}" for i,value in enumerate(values))
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/><text x="{35+row*175}" y="345" fill="{color}">{html.escape(name)}</text>')
    for i in range(7):
        svg.append(f'<text x="{px(i)}" y="315" text-anchor="middle">h={i+1}</text>')
    for value in np.linspace(minimum,maximum,4):
        svg.append(f'<text x="58" y="{py(value)+4:.1f}" text-anchor="end">{value:.0f}</text>')
    path.write_text("".join(svg)+"</g></svg>", encoding="utf-8")


# SECTION 3 — Direct multi-output forecasting with no future realized covariates.
def build_model(tf, lookback=56, features=6, horizon=7):
    inputs = tf.keras.layers.Input((lookback, features))
    x = tf.keras.layers.LSTM(32)(inputs)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(horizon)(x)
    return tf.keras.Model(inputs, outputs, name="direct_seven_day_lstm")


def run(args):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    n, lookback, horizon = (1460 if args.quick else 2190), 56, 7
    demand, temperature, components = generate_series(args.seed, n)
    split = boundaries(n)
    features, mean, scale = input_features(demand, temperature, split["train_end"])
    ranges = [(0,split["train_end"],1),(split["train_end"],split["validation_end"],1),
              (split["validation_end"],split["calibration_end"],1),(split["calibration_end"],n,1)]
    train, validation, calibration, test = [make_windows(features,demand,lookback,horizon,a,b,s) for a,b,s in ranges]
    x_train, y_train, _ = train
    x_val, y_val, _ = validation
    x_cal, y_cal, o_cal = calibration
    x_test, y_test, o_test = test
    model = build_model(tf, lookback, features.shape[1], horizon)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3, clipnorm=1.), loss="mse", jit_compile=False)
    start = time.perf_counter()
    history = model.fit(x_train,(y_train-mean[0])/scale[0],
                        validation_data=(x_val,(y_val-mean[0])/scale[0]),batch_size=32,
                        epochs=2 if args.quick else 40, verbose=2, shuffle=True,
                        callbacks=[tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True),
                                   tf.keras.callbacks.TerminateOnNaN()])
    cal_prediction = model.predict(x_cal, verbose=0)*scale[0]+mean[0]
    test_prediction = model.predict(x_test, verbose=0)*scale[0]+mean[0]
    assert test_prediction.shape == y_test.shape and np.isfinite(test_prediction).all()
    q = conformal_quantiles(np.abs(y_cal-cal_prediction),args.alpha)
    if not np.isfinite(q).all():
        raise ValueError("Too few calibration origins for a finite interval at this alpha")
    last_test, seasonal_test = baseline_forecasts(demand,o_test,horizon)
    last_cal, seasonal_cal = baseline_forecasts(demand,o_cal,horizon)
    mase_scale = np.mean(np.abs(demand[7:split["train_end"]]-demand[:split["train_end"]-7]))
    metrics = {"neural": forecast_metrics(y_test,test_prediction,mase_scale,test_prediction-q,test_prediction+q)}
    for name,prediction,calpred in [("last_value",last_test,last_cal),("seasonal_naive",seasonal_test,seasonal_cal)]:
        bq = conformal_quantiles(np.abs(y_cal-calpred),args.alpha)
        metrics[name] = forecast_metrics(y_test,prediction,mase_scale,prediction-bq,prediction+bq)
    args.output.mkdir(parents=True,exist_ok=True)
    model.save(args.output / "demand.keras")
    reloaded = tf.keras.models.load_model(args.output / "demand.keras", compile=False)
    np.testing.assert_allclose(reloaded.predict(x_test[:2],verbose=0)*scale[0]+mean[0],test_prediction[:2],atol=1e-3)
    (args.output / "normalization.json").write_text(json.dumps({"mean":mean.tolist(),"scale":scale.tolist()}),encoding="utf-8")
    result = {"provenance":"actual TensorFlow training; synthetic demand", "seed":args.seed,
              "environment":{"python":platform.python_version(),"numpy":np.__version__,"tensorflow":tf.__version__},
              "configuration":{"n":n,"lookback":lookback,"horizon":horizon,"splits":split,
                               "calibration_and_test_stride":1,"alpha":args.alpha,"normalization_fitted_through":split["train_end"]-1},
              "metrics":metrics,"conformal":{"q":q.tolist(),"n_calibration":len(o_cal),
                                             "rank":math.ceil((len(o_cal)+1)*(1-args.alpha)),
                                             "caveat":"dependent time series: no exchangeability-based coverage guarantee; horizonwise, not simultaneous"},
              "history":history.history,"seconds":time.perf_counter()-start,"reload_verified":True,
              "residual_acf_horizon1":autocorrelation(y_test[:,0]-test_prediction[:,0],min(12,len(y_test)-1)),
              "examples":[{"origin":int(o),"actual":y.tolist(),"neural":p.tolist(),"lower":(p-q).tolist(),"upper":(p+q).tolist()}
                          for o,y,p in zip(o_test[:5],y_test[:5],test_prediction[:5])]}
    (args.output / "results.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    np.savez_compressed(args.output / "held_out_forecasts.npz",origins=o_test,actual=y_test,neural=test_prediction,
                        lower=test_prediction-q,upper=test_prediction+q,last_value=last_test,seasonal_naive=seasonal_test)
    write_forecast_svg(args.output / "held_out_forecast.svg",y_test[0],test_prediction[0],last_test[0],seasonal_test[0],test_prediction[0]-q,test_prediction[0]+q)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick",action="store_true",help="four years, at most two epochs; a smoke test")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--alpha",type=float,default=.2)
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parents[1]/"outputs/forecasting")
    args=parser.parse_args()
    try:
        run(args)
    except ModuleNotFoundError as exc:
        raise SystemExit("Install requirements.txt in an isolated environment first.") from exc
