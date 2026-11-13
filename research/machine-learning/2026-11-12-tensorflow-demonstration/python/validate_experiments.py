"""Meaningful numerical checks; optionally exercise TensorFlow forward/gradient APIs.

Run: python validate_experiments.py
     python validate_experiments.py --tensorflow
This does not claim that full training converges. No bytecode caches are written.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent


def load(name):
    spec=importlib.util.spec_from_file_location(Path(name).stem,ROOT/name)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def numerical_checks():
    checks=[]
    for path in sorted(ROOT.glob("*.py")):
        source=path.read_text(encoding="utf-8")
        ast.parse(source,filename=str(path))
        compile(source,str(path),"exec")
    checks.append("all Python sources parse and compile without writing bytecode")
    cv=load("01_computer_vision.py")
    nlp=load("02_natural_language_processing.py")
    forecast=load("03_time_series_forecasting.py")
    heat=load("04_scientific_computing.py")

    # Dataset reproducibility, shapes, semantics/range, and distinct split seeds.
    first=cv.make_dataset(8,42)
    repeated=cv.make_dataset(8,42)
    second=cv.make_dataset(8,43)
    np.testing.assert_array_equal(first[0],repeated[0])
    np.testing.assert_array_equal(first[1],repeated[1])
    assert not np.array_equal(first[0],second[0])
    assert first[0].shape==(8,64,64,3) and first[1].shape==(8,64,64,1)
    assert np.all((first[0]>=0)&(first[0]<=1))
    assert .02<float(first[1].mean())<.8
    simple=cv.segmentation_metrics([1,1,0,0],[.9,.1,.9,.1])
    assert (simple["tp"],simple["fp"],simple["fn"],simple["tn"])==(1,1,1,1)
    np.testing.assert_allclose([simple["iou"],simple["dice"],simple["accuracy"]],[1/3,.5,.5])
    assert cv.segmentation_metrics([0,0],[0,0])["dice"]==1
    checks.append("vision: deterministic aligned geometry, 64x64 tensor dimensions, independent scenes, binary masks, metric hand calculation")

    corpus=nlp.make_corpus(42,quick=True)
    baseline=nlp.fit_baseline(corpus["train"][0],corpus["train"][1])
    train_words={w for t in corpus["train"][0] for w in nlp.tokenize(t)}
    assert set(baseline["vocabulary"][2:])<=train_words
    assert not set(corpus["train"][2])&set(corpus["test"][2])
    probabilities=nlp.baseline_predict(baseline,corpus["test"][0])
    assert probabilities.shape==(48,4) and np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1),1,atol=1e-12)
    assert nlp.tokenize("Wi-Fi, ACCOUNT_42!")==["wifi","account42"]
    metrics=nlp.classification_metrics(corpus["test"][1],probabilities)
    assert sum(map(sum,metrics["confusion_matrix"]))==48
    checks.append("language: disjoint template families and normalized texts, train-only vocabulary, NB probability normalization, confusion accounting")

    demand,temperature,components=forecast.generate_series(42)
    np.testing.assert_allclose(demand,np.stack(list(components.values())).sum(axis=0),atol=1e-10)
    assert np.all(demand>0)
    split=forecast.boundaries(len(demand))
    features,mean,scale=forecast.input_features(demand,temperature,split["train_end"])
    mutated=demand.copy(); mutated[split["train_end"]:]+=100000
    _,mean_changed,scale_changed=forecast.input_features(mutated,temperature,split["train_end"])
    np.testing.assert_array_equal(mean,mean_changed)
    np.testing.assert_array_equal(scale,scale_changed)
    x,y,origins=forecast.make_windows(features,demand,56,7,split["calibration_end"],len(demand),7)
    for i,o in enumerate(origins):
        np.testing.assert_array_equal(x[i],features[o-56:o])
        np.testing.assert_array_equal(y[i],demand[o:o+7])
    # A sentinel test proves changing the first target cannot change THAT origin's features.
    sentinel=np.arange(300,dtype=float)
    sx,sy,so=forecast.make_windows(sentinel[:,None],sentinel,56,7,200,260)
    changed=sentinel.copy(); changed[so[0]:]+=99999
    sx2,_,_=forecast.make_windows(changed[:,None],changed,56,7,200,260)
    np.testing.assert_array_equal(sx[0],sx2[0])
    assert sy[0,0]==so[0] and sx[0,-1,0]==so[0]-1
    last,seasonal=forecast.baseline_forecasts(demand,origins)
    np.testing.assert_array_equal(seasonal[0],demand[origins[0]-7:origins[0]])
    scores=np.arange(1,11,dtype=float)[:,None]
    assert forecast.conformal_quantiles(scores,.2)[0]==9 # ceil(11*.8)=9.
    assert math.isinf(forecast.conformal_quantiles(scores[:2],.1)[0])
    checks.append("forecast: additive process, positive demand, train-only scaling, chronological target boundaries, future sentinel isolation, weekday baseline, exact conformal rank")

    xx=np.linspace(.1,.9,9); tt=np.linspace(.1,.9,9)
    ut,uxx=heat.analytical_derivatives(xx,tt,.1)
    np.testing.assert_allclose(ut-.1*uxx,0,atol=1e-15)
    h=1e-4
    independent_uxx=(heat.analytical(xx+h,tt)-2*heat.analytical(xx,tt)+heat.analytical(xx-h,tt))/(h*h)
    independent_ut=(heat.analytical(xx,tt+h)-heat.analytical(xx,tt-h))/(2*h)
    np.testing.assert_allclose(independent_uxx,uxx,atol=2e-7,rtol=3e-8)
    np.testing.assert_allclose(independent_ut,ut,atol=3e-9,rtol=3e-9)
    np.testing.assert_allclose(heat.analytical(xx,0),np.sin(np.pi*xx))
    assert abs(heat.analytical(1,.5))<1e-15
    errors=[]
    for nx in [21,41,81]:
        x,t,field,r=heat.ftcs(.1,nx)
        assert 0<r<=.5 and np.isfinite(field).all()
        np.testing.assert_array_equal(field[:,[0,-1]],0)
        errors.append(heat.field_metrics(heat.analytical(x,1),field[-1])["relative_l2"])
    assert errors[2]<errors[1]<errors[0]
    assert 3.8<errors[0]/errors[1]<4.2
    assert (1-2*.4)*1+.4*(.5+.5)==.6
    try:
        heat.ftcs(requested_r=.6)
        raise AssertionError("Unstable FTCS unexpectedly accepted")
    except ValueError:
        pass
    unstable=heat.unstable_trace()
    assert unstable["trace"][-1]["max_abs"]>1000
    assert len(unstable["trace"])<=100
    checks.append("heat: exact boundary/initial conditions and derivatives, independent difference derivatives, stable FTCS, grid convergence, hand update, rejected unstable setting and guarded instability demonstration")

    # Assets must contain finite values and internally consistent experimental data.
    data=ROOT.parent/"data"
    if (data/"generation-manifest.json").exists():
        assets={p.name:json.loads(p.read_text(encoding="utf-8")) for p in data.glob("*.json")}

        def finite(value):
            if isinstance(value,float):
                assert math.isfinite(value)
            elif isinstance(value,dict):
                for v in value.values(): finite(v)
            elif isinstance(value,list):
                for v in value: finite(v)
        finite(assets)
        vision=assets["vision-demo.json"]
        for scene in vision["scenes"]:
            assert len(scene["rgb"])==4096 and len(scene["mask"])==4096 and len(scene["score"])==4096
            actual=cv.segmentation_metrics(scene["mask"],scene["score"])
            for key in ["tp","fp","fn","tn"]:
                assert actual[key]==scene["metrics"][key]
        language=assets["nlp-demo.json"]
        exported={"vocabulary":language["vocabulary"],"log_prior":np.array(language["log_prior"]),
                  "log_likelihood":np.array(language["log_likelihood"])}
        reconstructed=nlp.baseline_predict(exported,[e["text"] for e in language["examples"]])
        np.testing.assert_allclose(reconstructed,[e["probabilities"] for e in language["examples"]],atol=1e-14)
        forecast_asset=assets["forecasting-demo.json"]
        np.testing.assert_allclose(forecast_asset["demand"],np.stack(list(forecast_asset["components"].values())).sum(axis=0))
        heat_asset=assets["scientific-demo.json"]
        np.testing.assert_allclose(heat_asset["error"],np.abs(np.array(heat_asset["finite_difference"])-np.array(heat_asset["exact"])))
        checks.append("browser assets: finite JSON, pixel confusion counts, exported NB inference parity, coherent demand components, FTCS absolute-error consistency")
    return checks


def tensorflow_checks():
    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()
    checks=[]
    foundation=load("00_tensorflow_foundations.py")
    foundation.run()
    checks.append("TensorFlow foundations numerical assertions")
    cv=load("01_computer_vision.py")
    images,masks=cv.make_dataset(2,42)
    model=cv.build_model(tf)
    with tf.GradientTape() as tape:
        probabilities=model(images)
        loss=cv.composite_loss(tf,masks,probabilities)
    gradients=tape.gradient(loss,model.trainable_variables)
    assert tuple(probabilities.shape)==(2,64,64,1) and np.isfinite(float(loss))
    assert all(g is not None and np.isfinite(g.numpy()).all() for g in gradients)
    tf.keras.optimizers.Adam(1e-3).apply_gradients(zip(gradients,model.trainable_variables))
    checks.append("vision TensorFlow forward, finite composite loss, finite gradients, optimization step")
    nlp=load("02_natural_language_processing.py")
    corpus=nlp.make_corpus(42,True)
    classifier,vectorizer=nlp.build_model(tf,corpus["train"][0])
    logits=classifier(tf.constant(corpus["test"][0][:3]))
    assert tuple(logits.shape)==(3,4) and np.isfinite(logits.numpy()).all()
    assert "auniquelyheldouttoken" not in vectorizer.get_vocabulary()
    checks.append("language TensorFlow train-only adaptation, masked pooling forward, logits shape")
    forecast=load("03_time_series_forecasting.py")
    forecaster=forecast.build_model(tf)
    outputs=forecaster(tf.zeros((2,56,6)))
    assert tuple(outputs.shape)==(2,7) and np.isfinite(outputs.numpy()).all()
    checks.append("forecast TensorFlow direct seven-output shape and finite forward")
    heat=load("04_scientific_computing.py")
    pinn=heat.build_model(tf,16)
    points=tf.constant([[.2,.3],[.5,.4],[.8,.7]],dtype=tf.float64)
    with tf.GradientTape() as tape:
        u,ut,uxx=heat.differential_fields(tf,pinn,points)
        loss=tf.reduce_mean((ut-.1*uxx)**2)+tf.reduce_mean(u**2)
    gradients=tape.gradient(loss,pinn.trainable_variables)
    assert all(g is not None and np.isfinite(g.numpy()).all() for g in gradients)
    assert tuple(uxx.shape)==(3,1)
    # Check nested derivatives against the exact field using a differentiable lambda.
    def exact_model(xt,training=True):
        return tf.exp(-.1*np.pi**2*xt[:,1:2])*tf.sin(np.pi*xt[:,:1])
    _,at,axx=heat.differential_fields(tf,exact_model,points)
    np.testing.assert_allclose((at-.1*axx).numpy(),0,atol=1e-12)
    tf.keras.optimizers.Adam(1e-3).apply_gradients(zip(gradients,pinn.trainable_variables))
    checks.append("heat TensorFlow nested derivative parity with exact PDE, parameter gradients through differential operators, optimization step")
    return checks


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensorflow",action="store_true",help="also require and test TensorFlow APIs")
    parser.add_argument("--report",type=Path,help="optional JSON report path")
    args=parser.parse_args()
    passed=numerical_checks()
    tf_status="not run (use --tensorflow after installing requirements.txt)"
    if args.tensorflow:
        passed+=tensorflow_checks()
        tf_status="forward/loss/gradient checks passed; full application training is separate"
    result={"python":platform.python_version(),"numpy":np.__version__,"passed":passed,"tensorflow":tf_status}
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
