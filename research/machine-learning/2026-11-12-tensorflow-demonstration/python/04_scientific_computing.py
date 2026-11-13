"""Heat equation: exact solution, stable FTCS, and a physics-informed network.

Run: python 04_scientific_computing.py --quick --output ../outputs/scientific
Coordinates and temperature are NONDIMENSIONAL. Default diffusivity kappa=0.1.
Independent program. Analytical and FTCS functions can run without TensorFlow.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import struct
import time
import zlib

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np


# SECTION 1 — The benchmark and its independently checkable derivatives.
def analytical(x, t, kappa=.1):
    return np.exp(-kappa*np.pi**2*np.asarray(t))*np.sin(np.pi*np.asarray(x))


def analytical_derivatives(x, t, kappa=.1):
    u = analytical(x,t,kappa)
    return -kappa*np.pi**2*u, -np.pi**2*u # u_t, u_xx


def ftcs(kappa=.1, nx=41, final_time=1., requested_r=.4):
    """u[n+1,i]=(1-2r)u[n,i]+r(u[n,i-1]+u[n,i+1]).

    Choose an integer number of steps to hit final_time exactly and round dt DOWN.
    Thus actual r<=requested_r<=0.5. Boundaries remain exactly zero at every step.
    Returns x, time, field[time,space], and actual stability ratio.
    """
    if kappa<=0 or nx<3 or final_time<=0 or not 0<requested_r<=.5:
        raise ValueError("Require kappa>0, nx>=3, time>0, 0<r<=0.5")
    x=np.linspace(0,1,nx)
    dx=1/(nx-1)
    steps=max(1,math.ceil(kappa*final_time/(requested_r*dx*dx)))
    dt=final_time/steps
    r=kappa*dt/(dx*dx)
    assert r<=.5+1e-14
    field=np.zeros((steps+1,nx),dtype=np.float64)
    field[0]=np.sin(np.pi*x)
    field[0,[0,-1]]=0
    for j in range(steps):
        field[j+1,1:-1]=(1-2*r)*field[j,1:-1]+r*(field[j,:-2]+field[j,2:])
    assert np.isfinite(field).all() and np.max(np.abs(field))<=1+1e-12
    return x,np.linspace(0,final_time,steps+1),field,r


def sample_ftcs(kappa=.1,nx=41,times=None):
    if times is None:
        times=np.linspace(0,1,21)
    x,times_full,field,r=ftcs(kappa,nx,float(np.max(times)))
    sampled=np.stack([np.interp(times,times_full,field[:,i]) for i in range(nx)],axis=1)
    return x,np.asarray(times),sampled,r


def unstable_trace(kappa=.1,nx=41,requested_r=.65,steps=100):
    """Intentionally unstable FTCS, bounded by a safety stop (not a solver).

    Add a 1e-6 alternating perturbation so the unstable high-frequency mode is
    visible. A pure first sine mode alone need not visibly reveal the instability.
    """
    x=np.linspace(0,1,nx)
    u=np.sin(np.pi*x)+1e-6*(-1.)**np.arange(nx)
    u[[0,-1]]=0
    trace=[]
    for step in range(steps):
        amplitude=float(np.max(np.abs(u)))
        trace.append({"step":step,"max_abs":amplitude})
        if not np.isfinite(amplitude) or amplitude>1e6:
            break
        u[1:-1]=(1-2*requested_r)*u[1:-1]+requested_r*(u[:-2]+u[2:])
    return {"r":requested_r,"dt":requested_r/(kappa*(nx-1)**2),
            "perturbation":1e-6,"stop_amplitude":1e6,"trace":trace}


def field_metrics(exact, predicted):
    difference=np.asarray(predicted)-np.asarray(exact)
    denominator=np.linalg.norm(exact)
    return {"relative_l2":float(np.linalg.norm(difference)/denominator),
            "max_abs_error":float(np.max(np.abs(difference))),
            "rmse":float(np.sqrt(np.mean(difference**2))),
            "aggregation":"unweighted Euclidean norm across every evaluation grid point"}


# SECTION 2 — Smooth, row-independent neural field; float64 throughout.
def build_model(tf,width=32):
    inputs=tf.keras.layers.Input((2,),dtype=tf.float64)
    x=tf.keras.layers.Rescaling(2.,offset=-1.,dtype=tf.float64)(inputs)
    for _ in range(3):
        x=tf.keras.layers.Dense(width,activation="tanh",dtype=tf.float64)(x)
    output=tf.keras.layers.Dense(1,dtype=tf.float64)(x)
    return tf.keras.Model(inputs,output,name="heat_pinn")


def differential_fields(tf,model,xt):
    """Return u,u_t,u_xx as [N,1] tensors.

    gradient(non-scalar,xt) differentiates the SUM of outputs. That equals each
    sample's derivative only because Dense/tanh here never mix batch rows. This
    shortcut is invalid for BatchNormalization or attention across samples.
    Both the first-gradient operation AND its x slice occur inside outer tape.
    """
    with tf.GradientTape() as second:
        second.watch(xt)
        with tf.GradientTape() as first:
            first.watch(xt)
            u=model(xt,training=True)
        du=first.gradient(u,xt)
        ux=du[:,:1]
    d_ux=second.gradient(ux,xt)
    return u,du[:,1:2],d_ux[:,:1]


def sample_points(tf,rng,n_residual,n_boundary,n_initial):
    residual=rng.uniform(0,1,(n_residual,2))
    boundary=np.column_stack([rng.integers(0,2,n_boundary),rng.uniform(0,1,n_boundary)])
    initial=np.column_stack([rng.uniform(0,1,n_initial),np.zeros(n_initial)])
    return tuple(tf.convert_to_tensor(a,dtype=tf.float64) for a in (residual,boundary,initial))


def write_heatmap(path,values,signed=False):
    """A small standard PNG, with high values warm and low values pale blue."""
    values=np.asarray(values,dtype=float)
    normalized=values/(np.max(values) if signed and np.max(values)>0 else 1.)
    normalized=np.clip(normalized,0,1)
    cold=np.array([220,235,244]); warm=np.array([194,69,48])
    image=np.rint(cold+(warm-cold)*normalized[...,None]).astype(np.uint8)
    image=np.repeat(np.repeat(image,5,0),5,1)
    height,width,_=image.shape

    def chunk(kind,data):
        return struct.pack(">I",len(data))+kind+data+struct.pack(">I",zlib.crc32(kind+data)&0xffffffff)
    rows=b"".join(b"\x00"+row.tobytes() for row in image)
    path.write_bytes(b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR",struct.pack(">IIBBBBB",width,height,8,2,0,0,0))+
                     chunk(b"IDAT",zlib.compress(rows,9))+chunk(b"IEND",b""))


def write_profile_svg(path,x,exact,fd,pinn):
    svg=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 350" role="img"><title>Heat profiles at t=0.5 on independent grid</title><rect width="720" height="350" fill="#fff8f2"/><g font-family="sans-serif" fill="#26332f"><text x="35" y="25">Heat profiles at t=0.5 • nondimensional temperature</text>']
    for values,color,name,col in [(exact,"#26332f","Exact",0),(fd,"#5b75a2","FTCS",1),(pinn,"#b66e3e","PINN",2)]:
        points=" ".join(f"{50+620*a:.3f},{285-235*b:.3f}" for a,b in zip(x,values))
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.3"/><text x="{60+col*210}" y="330" fill="{color}">{name}</text>')
    for a in [0,.25,.5,.75,1]:
        svg.append(f'<text x="{50+620*a}" y="305" text-anchor="middle" font-size="12">{a:g}</text>')
    path.write_text("".join(svg)+"</g></svg>",encoding="utf-8")


# SECTION 3 — Parameter tape encloses the entire differential-operator graph.
def run(args):
    import tensorflow as tf

    if args.kappa<=0:
        raise ValueError("Thermal diffusivity must be positive")
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    rng=np.random.default_rng(args.seed)
    width=24 if args.quick else 32
    model=build_model(tf,width)
    schedule=tf.keras.optimizers.schedules.ExponentialDecay(1e-3,1000,.9,staircase=True)
    optimizer=tf.keras.optimizers.Adam(schedule,clipnorm=5.)
    kappa=tf.constant(args.kappa,dtype=tf.float64)
    weights=(1.,10.,10.) # residual, boundary, initial; fields already nondimensional.

    @tf.function
    def train_step(residual_points,boundary_points,initial_points):
        with tf.GradientTape() as parameters:
            _,ut,uxx=differential_fields(tf,model,residual_points)
            residual=ut-kappa*uxx
            lr=tf.reduce_mean(tf.square(residual))
            lb=tf.reduce_mean(tf.square(model(boundary_points,training=True)))
            target=tf.sin(tf.constant(np.pi,dtype=tf.float64)*initial_points[:,:1])
            l0=tf.reduce_mean(tf.square(model(initial_points,training=True)-target))
            loss=weights[0]*lr+weights[1]*lb+weights[2]*l0
        gradients=parameters.gradient(loss,model.trainable_variables)
        tf.debugging.assert_all_finite(loss,"Nonfinite PINN objective")
        for gradient in gradients:
            if gradient is None:
                raise ValueError("Disconnected parameter gradient")
            tf.debugging.assert_all_finite(gradient,"Nonfinite PINN parameter gradient")
        optimizer.apply_gradients(zip(gradients,model.trainable_variables))
        return loss,lr,lb,l0

    steps=200 if args.quick else 4000
    n_residual=256 if args.quick else 1024
    history=[]
    start=time.perf_counter()
    for step in range(steps):
        if step%100==0:
            points=sample_points(tf,rng,n_residual,64,64)
        values=train_step(*points)
        if step%20==0 or step==steps-1:
            row={"step":step+1,"total":float(values[0]),"residual":float(values[1]),"boundary":float(values[2]),"initial":float(values[3])}
            history.append(row)
            print(json.dumps(row),flush=True)

    # SECTION 4 — Independent uniform space-time grid, never used to fit parameters.
    x=np.linspace(0,1,81); times=np.linspace(0,1,51)
    xx,tt=np.meshgrid(x,times)
    evaluation=tf.constant(np.column_stack([xx.ravel(),tt.ravel()]),dtype=tf.float64)
    u,ut,uxx=differential_fields(tf,model,evaluation)
    neural=u.numpy().reshape(xx.shape)
    residual=(ut-kappa*uxx).numpy().reshape(xx.shape)
    exact=analytical(xx,tt,args.kappa)
    _,_,finite_difference,r=sample_ftcs(args.kappa,len(x),times)
    assert np.isfinite(neural).all() and np.isfinite(residual).all()
    neural_metrics=field_metrics(exact,neural)
    neural_metrics.update(residual_rms=float(np.sqrt(np.mean(residual**2))),
                          residual_max_abs=float(np.max(np.abs(residual))),
                          boundary_max_abs=float(np.max(np.abs(neural[:,[0,-1]]))),
                          initial_rmse=float(np.sqrt(np.mean((neural[0]-np.sin(np.pi*x))**2))))
    args.output.mkdir(parents=True,exist_ok=True)
    model.save(args.output/"heat.keras")
    restored=tf.keras.models.load_model(args.output/"heat.keras",compile=False)
    np.testing.assert_allclose(restored(evaluation[:10]).numpy(),u.numpy()[:10],atol=1e-10)
    convergence=[]
    for nx in [21,41,81]:
        cx,ct,cf,cr=ftcs(args.kappa,nx)
        convergence.append({"nx":nx,"dx":1/(nx-1),"dt":ct[1]-ct[0],"r":cr,
                            **field_metrics(analytical(cx,1,args.kappa),cf[-1])})
    result={"provenance":"actual TensorFlow PINN training and FTCS computation", "seed":args.seed,
            "environment":{"python":platform.python_version(),"numpy":np.__version__,"tensorflow":tf.__version__},
            "configuration":{"kappa":args.kappa,"steps":steps,"width":width,"weights":weights,
                             "n_residual":n_residual,"resample_every":100,"boundary_and_initial_each":64,"dtype":"float64"},
            "pinn_metrics":neural_metrics,"ftcs_metrics":field_metrics(exact,finite_difference),
            "ftcs_r":r,"convergence":convergence,"history":history,"seconds":time.perf_counter()-start,
            "reload_verified":True,"warning":"Optimizer completion is not a convergence or accuracy certificate."}
    (args.output/"results.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    np.savez_compressed(args.output/"evaluation_fields.npz",x=x,t=times,exact=exact,finite_difference=finite_difference,
                        pinn=neural,absolute_error=np.abs(neural-exact),residual=residual)
    write_heatmap(args.output/"pinn_temperature.png",neural)
    write_heatmap(args.output/"pinn_absolute_error.png",np.abs(neural-exact),signed=True)
    write_heatmap(args.output/"pinn_absolute_residual.png",np.abs(residual),signed=True)
    write_profile_svg(args.output/"heat_profiles.svg",x,exact[25],finite_difference[25],neural[25])
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick",action="store_true",help="200 steps and 256 collocation points; does not certify convergence")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--kappa",type=float,default=.1)
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parents[1]/"outputs/scientific")
    args=parser.parse_args()
    try:
        run(args)
    except ModuleNotFoundError as exc:
        raise SystemExit("Install requirements.txt in an isolated environment first.") from exc
