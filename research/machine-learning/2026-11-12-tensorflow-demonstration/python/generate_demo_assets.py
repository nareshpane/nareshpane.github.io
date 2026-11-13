"""Generate honest browser assets using NumPy only: no fitted neural outputs.

Run from any folder: python generate_demo_assets.py
Imports canonical application files located beside this script. Writes only the
monograph's data/ JSON assets and images/ numerical diagnostics. Reproducible seed42.
The language baseline IS fitted multinomial Naive Bayes; all other browser results
are explicit formulas, fixed heuristics, or classical numerical calculations.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
import importlib.util
import json
from pathlib import Path
import platform
import sys

import numpy as np


def load_module(filename):
    name=Path(filename).stem
    spec=importlib.util.spec_from_file_location(name,Path(__file__).resolve().parent/filename)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path,value):
    path.write_text(json.dumps(value,separators=(",",":"),ensure_ascii=False,allow_nan=False),encoding="utf-8")


def generate(output: Path,seed=42):
    data=output/"data"; images=output/"images"
    data.mkdir(parents=True,exist_ok=True); images.mkdir(parents=True,exist_ok=True)
    cv=load_module("01_computer_vision.py")
    nlp=load_module("02_natural_language_processing.py")
    forecast=load_module("03_time_series_forecasting.py")
    scientific=load_module("04_scientific_computing.py")
    provenance={"python":platform.python_version(),"numpy":np.__version__,"seed":seed,
                "tensorflow_training":"not executed by this generator; only NumPy, exact formulas, and classical baselines",
                "tensorflow_installed":importlib.util.find_spec("tensorflow") is not None,
                "generator":"python/generate_demo_assets.py"}

    # SECTION 1 — Synthetic imagery and a declared fixed excess-green heuristic.
    scene_seeds=[seed,seed+39,seed+1984]
    scenes=[]; truth=[]; scores=[]; rgb=[]
    for i,s in enumerate(scene_seeds):
        image,mask=cv.generate_scene(s)
        score=cv.color_score(image)
        scene={"id":f"scene-{i+1}","label":["Mixed neighborhood","Roof and canopy confounders","Shadows and dry vegetation"][i],
               "width":64,"height":64,"seed":s,
               "rgb":np.rint(image*255).astype(int).reshape(-1,3).tolist(),
               "mask":mask.astype(int).ravel().tolist(),"score":np.round(score.ravel(),6).tolist(),
               "metrics":cv.segmentation_metrics(mask,score,.5)}
        scenes.append(scene); truth.append(mask); scores.append(score); rgb.append(image)
    vision={"provenance":"Actual deterministic synthetic images and fixed color-score baseline; NOT U-Net predictions.",
            "seed":seed,"threshold":.5,"score_formula":"clip((2G-R-B+0.05)/0.55,0,1); RGB intensities in [0,1]",
            "scenes":scenes,"aggregate":cv.segmentation_metrics(np.stack(truth),np.stack(scores),.5)}
    write_json(data/"vision-demo.json",vision)
    cv.write_examples(images/"vision-baseline-diagnostics.png",np.stack(rgb),np.stack(truth),np.stack(scores))

    # SECTION 2 — A truly fitted, train-only count baseline, not a neural substitute.
    corpus=nlp.make_corpus(seed)
    train_text,train_y,_=corpus["train"]
    test_text,test_y,_=corpus["test"]
    baseline=nlp.fit_baseline(train_text,train_y)
    test_prob=nlp.baseline_predict(baseline,test_text)
    index={word:i for i,word in enumerate(baseline["vocabulary"])}
    probes=list(nlp.CHALLENGES)
    errors=np.where(test_prob.argmax(axis=1)!=test_y)[0]
    probes += [(test_text[i],int(test_y[i])) for i in errors[:4]]
    probe_prob=nlp.baseline_predict(baseline,[t for t,_ in probes])
    examples=[]
    for (text,y),p in zip(probes,probe_prob):
        tokens=nlp.tokenize(text)
        encoded=[index.get(w,1) for w in tokens[:48]]
        examples.append({"text":text,"true_label":None if y is None else nlp.CLASSES[y],
                         "predicted_label":nlp.CLASSES[int(p.argmax())],"probabilities":p.tolist(),
                         "tokens":tokens,"encoded":encoded+[0]*(48-len(encoded))})
    baseline_metrics=nlp.classification_metrics(test_y,test_prob)
    language={"provenance":"Actual fitted multinomial Naive Bayes baseline on synthetic requests; NOT neural predictions.",
              "classes":nlp.CLASSES,"vocabulary":baseline["vocabulary"],"sequence_length":48,
              "examples":examples,"baseline_metrics":baseline_metrics,"template_families":nlp.FAMILIES,
              "counts":{k:len(v[0]) for k,v in corpus.items()},
              "log_prior":baseline["log_prior"].tolist(),"log_likelihood":baseline["log_likelihood"].tolist(),
              "free_text_caveat":"Browser can evaluate this exported count baseline exactly; probabilities are not calibrated or neural.",
              "misclassified_examples":[{"text":test_text[i],"true_label":nlp.CLASSES[test_y[i]],
                                         "predicted_label":nlp.CLASSES[int(test_prob[i].argmax())],
                                         "probabilities":test_prob[i].tolist()} for i in errors[:10]]}
    write_json(data/"nlp-demo.json",language)
    nlp.write_confusion(images/"nlp-baseline-confusion.svg",baseline_metrics["confusion_matrix"],"Held-out families: count baseline")

    # SECTION 3 — Six years of coherent demand components and honest baselines.
    demand,temperature,components=forecast.generate_series(seed,2190)
    split=forecast.boundaries(len(demand))
    features,mean,scale=forecast.input_features(demand,temperature,split["train_end"])
    _,cal_actual,cal_origins=forecast.make_windows(features,demand,56,7,split["validation_end"],split["calibration_end"],1)
    _,actual,origins=forecast.make_windows(features,demand,56,7,split["calibration_end"],len(demand),1)
    cal_last,cal_seasonal=forecast.baseline_forecasts(demand,cal_origins)
    last,seasonal=forecast.baseline_forecasts(demand,origins)
    q=forecast.conformal_quantiles(np.abs(cal_actual-cal_seasonal),.2)
    q_last=forecast.conformal_quantiles(np.abs(cal_actual-cal_last),.2)
    mase_scale=np.mean(np.abs(demand[7:split["train_end"]]-demand[:split["train_end"]-7]))
    examples=[]
    for i in range(0,len(origins),7):
        o=origins[i]
        examples.append({"origin":int(o),"history":demand[o-56:o].tolist(),"actual":actual[i].tolist(),
                         "last_value":last[i].tolist(),"seasonal_naive":seasonal[i].tolist(),
                         "lower":(seasonal[i]-q).tolist(),"upper":(seasonal[i]+q).tolist()})
    forecasting={"provenance":"Actual synthetic observations, last-value and seasonal-naive baselines, chronological baseline calibration; NO trained LSTM output.",
                 "seed":seed,"units":"MWh/day","start_date":"2020-01-06",
                 "components":{k:v.tolist() for k,v in components.items()},"temperature":temperature.tolist(),"demand":demand.tolist(),
                 "splits":split,"lookback":56,"horizon":7,"origins":[e["origin"] for e in examples],"examples":examples,
                 "evaluation_origins":origins.tolist(),"n_test_origins":len(origins),"evaluation_stride":1,
                 "gallery_stride":7,
                 "metrics":{"last_value":forecast.forecast_metrics(actual,last,mase_scale,last-q_last,last+q_last),
                            "seasonal_naive":forecast.forecast_metrics(actual,seasonal,mase_scale,seasonal-q,seasonal+q)},
                 "conformal":{"alpha":.2,"n_calibration":len(cal_origins),"q":q.tolist(),
                              "rank":int(np.ceil((len(cal_origins)+1)*.8)),
                              "method":"Horizonwise absolute errors; exact ceil((m+1)(1-alpha)) order statistic",
                              "caveat":"Chronological dependence breaks exchangeability; empirical horizonwise coverage only, not simultaneous coverage."},
                 "autocorrelation":forecast.autocorrelation(demand,28),
                 "residual_autocorrelation_horizon1":forecast.autocorrelation(actual[:,0]-seasonal[:,0],12)}
    write_json(data/"forecasting-demo.json",forecasting)

    # SECTION 4 — Analytic solution and actual stable finite differences.
    x,t,fd,r=scientific.sample_ftcs(.1,41,np.linspace(0,1,21))
    xx,tt=np.meshgrid(x,t)
    exact=scientific.analytical(xx,tt,.1)
    error=np.abs(fd-exact)
    convergence=[]
    for nx in [21,41,81]:
        cx,ct,cf,cr=scientific.ftcs(.1,nx)
        convergence.append({"nx":nx,"dx":float(cx[1]-cx[0]),"dt":float(ct[1]-ct[0]),"r":cr,
                            **scientific.field_metrics(scientific.analytical(cx,1,.1),cf[-1])})
    heat={"provenance":"Analytical formula and actual FTCS computation; NO fitted PINN output.","kappa":.1,
          "x":x.tolist(),"t":t.tolist(),"exact":exact.tolist(),"finite_difference":fd.tolist(),"error":error.tolist(),
          "metrics":{**scientific.field_metrics(exact,fd),"r":r,"boundary_max_abs":float(np.max(np.abs(fd[:,[0,-1]])))},
          "convergence":convergence,"profiles":[{"t":float(t[j]),"exact":exact[j].tolist(),"finite_difference":fd[j].tolist()} for j in [0,5,10,15,20]],
          "unstable_example":scientific.unstable_trace()}
    write_json(data/"scientific-demo.json",heat)
    scientific.write_heatmap(images/"heat-ftcs-error.png",error,signed=True)
    scientific.write_heatmap(images/"heat-exact.png",exact)

    manifest={**provenance,"assets":["vision-demo.json","nlp-demo.json","forecasting-demo.json","scientific-demo.json"],
              "validation":"generated from canonical numerical functions; see validate_experiments.py",
              "metrics":{"vision_color_baseline":vision["aggregate"],"nlp_count_baseline":baseline_metrics,
                         "forecast_baselines":forecasting["metrics"],"heat_ftcs":heat["metrics"]}}
    write_json(data/"generation-manifest.json",manifest)
    print(json.dumps({"output":str(output.resolve()),"environment":provenance,
                      "metrics":{"cv_iou":vision["aggregate"]["iou"],"nlp_accuracy":baseline_metrics["accuracy"],
                                 "forecast_mae":forecasting["metrics"]["seasonal_naive"]["mae"],
                                 "ftcs_relative_l2":heat["metrics"]["relative_l2"]}},indent=2))
    return manifest


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed",type=int,default=42)
    args=parser.parse_args()
    sys.dont_write_bytecode=True
    generate(args.output,args.seed)
