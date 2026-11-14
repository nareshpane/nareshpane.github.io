"""Infer a dimensionless river source through the same upwind-diffusion solver."""
from __future__ import annotations
import time
import numpy as np
from scipy.optimize import minimize
import torch
from common import cli, configure, environment, plot_lines, write_json


def grid(nx=61, final=1.2, velocity=.35, dispersion=.005):
    if nx < 5 or final <= 0 or velocity < 0 or dispersion <= 0:
        raise ValueError('Use nx>=5, final>0, v>=0, D>0')
    dx = 1/(nx-1)
    steps = int(np.ceil(final*(velocity/dx+2*dispersion/dx**2)/.75))
    dt = final/steps
    A, B = velocity*dt/dx, dispersion*dt/dx**2
    assert A+2*B <= .75+1e-12
    return np.linspace(0, 1, nx), np.linspace(0, final, steps+1), A, B


def numpy_simulator(phi, nx=61, velocity=.35, dispersion=.005, final=1.2):
    x, t, A, B = grid(nx, final, velocity, dispersion)
    location, intensity = phi
    source = intensity*np.exp(-.5*((x-location)/.04)**2)/(.04*np.sqrt(2*np.pi))
    c = np.zeros(nx)
    states = [c.copy()]
    for _ in t[1:]:
        center = ((1-A-2*B)*c[1:-1]+(A+B)*c[:-2]+B*c[2:]+(t[1]-t[0])*source[1:-1])
        c = np.concatenate([[0.], center, [0.]])
        states.append(c)
    return x, t, np.array(states)


def torch_simulator(phi, nx=61, velocity=.35, dispersion=.005, final=1.2):
    x, t, A, B = grid(nx, final, velocity, dispersion)
    tx = torch.tensor(x)
    location, intensity = phi.unbind()
    source = intensity*torch.exp(-.5*((tx-location)/.04)**2)/(.04*np.sqrt(2*np.pi))
    c = torch.zeros(nx)
    states = [c]
    for _ in t[1:]:
        center = (1-A-2*B)*c[1:-1]+(A+B)*c[:-2]+B*c[2:]+(t[1]-t[0])*source[1:-1]
        # New tensors at every step preserve history required by backward.
        c = torch.cat([torch.zeros(1), center, torch.zeros(1)])
        states.append(c)
    return torch.stack(states)


def finite_gradient(function, phi, step=1e-5):
    eye = np.eye(len(phi))*step
    return np.array([(function(phi+e)-function(phi-e))/(2*step) for e in eye])


def manufactured_refinement():
    # c=exp(-t) sin(pi x), homogeneous boundaries, exact forcing c_t+v c_x-D c_xx.
    errors = []
    for nx in [31, 61, 121]:
        x, t, A, B = grid(nx, final=.5)
        c = np.sin(np.pi*x)
        for time_value in t[:-1]:
            source = np.exp(-time_value)*((-1+.005*np.pi**2)*np.sin(np.pi*x)+.35*np.pi*np.cos(np.pi*x))
            center = (1-A-2*B)*c[1:-1]+(A+B)*c[:-2]+B*c[2:]+(t[1]-t[0])*source[1:-1]
            c = np.concatenate([[0.], center, [0.]])
        exact = np.exp(-t[-1])*np.sin(np.pi*x)
        errors.append(dict(nx=nx, dt=float(t[1]), combined=float(A+2*B),
                           relative_l2=float(np.linalg.norm(c-exact)/np.linalg.norm(exact))))
    assert errors[2]['relative_l2'] < errors[1]['relative_l2'] < errors[0]['relative_l2']
    return errors


def run(seed=42, quick=False, output=None):
    configure(seed)
    rng = np.random.default_rng(seed)
    truth = np.array([.26, .8])
    x, t, field = numpy_simulator(truth)
    sensors = np.array([27, 39, 51])
    observation_indices = np.unique(np.linspace(5, len(t)-1, 18, dtype=int))
    noise_sd = .025
    z = field[observation_indices[:, None], sensors]+rng.normal(0, noise_sd, (len(observation_indices), len(sensors)))
    train_times = np.arange(len(observation_indices))%3 != 0
    oi = observation_indices[train_times]
    observations = z[train_times]
    def objective(phi, station_subset=slice(None), measured=observations):
        prediction = numpy_simulator(phi)[2][oi[:, None], sensors[station_subset]]
        return float(np.mean((prediction-measured)**2)/(2*noise_sd**2))
    def automatic(phi):
        leaf = torch.tensor(phi, requires_grad=True)
        prediction = torch_simulator(leaf)[oi[:, None], sensors]
        loss = ((prediction-torch.tensor(observations))**2).mean()/(2*noise_sd**2)
        return loss.item(), torch.autograd.grad(loss, leaf)[0].numpy()
    probe = np.array([.32, .7])
    numerical = finite_gradient(objective, probe)
    _, gradient = automatic(probe)
    relative_error = float(np.linalg.norm(numerical-gradient)/max(1, np.linalg.norm(gradient)))
    assert relative_error < 1e-6
    forward_error = float(np.max(abs(field-torch_simulator(torch.tensor(truth)).numpy())))
    assert forward_error < 1e-12
    bounds = [(.1, .55), (.1, 2.)]
    initial = np.array([.4, .5])
    conventional_history, autodiff_history = [], []
    start = time.perf_counter()
    baseline = minimize(objective, initial, method='L-BFGS-B', jac='3-point', bounds=bounds,
                        callback=lambda p: conventional_history.append([*p.tolist(), objective(p)]),
                        options=dict(maxiter=30 if quick else 100, ftol=1e-12, gtol=1e-7))
    baseline_seconds = time.perf_counter()-start
    start = time.perf_counter()
    learned = minimize(automatic, initial, method='L-BFGS-B', jac=True, bounds=bounds,
                       callback=lambda p: autodiff_history.append([*p.tolist(), objective(p)]),
                       options=dict(maxiter=30 if quick else 100, ftol=1e-12, gtol=1e-7))
    torch_seconds = time.perf_counter()-start
    heldout_indices = observation_indices[~train_times]
    prediction = numpy_simulator(learned.x)[2]
    heldout_rmse = float(np.sqrt(np.mean((prediction[heldout_indices[:, None], sensors]-z[~train_times])**2)))
    # Profile: at each fixed location, intensity has a one-dimensional LS solution.
    locations = np.linspace(.16, .4, 45)
    profile = []
    for loc in locations:
        unit = numpy_simulator([loc, 1.])[2][oi[:, None], sensors]
        q = np.clip(np.sum(unit*observations)/np.sum(unit*unit), .1, 2.)
        profile.append([float(loc), float(q), objective([loc, q])])
    # Actual independent noise replicates and fewer-station sensitivity, not confidence intervals.
    recovery = []
    for sensor_count in [1, 3]:
        for noise in [.025, .10]:
            for replicate in range(3 if not quick else 1):
                sub = slice(0, sensor_count)
                measured = field[oi[:, None], sensors[sub]]+rng.normal(0, noise, (len(oi), sensor_count))
                fit = minimize(lambda p: objective(p, sub, measured), initial, bounds=bounds, method='L-BFGS-B', options={'maxiter': 70})
                recovery.append(dict(sensors=sensor_count, noise=noise, replicate=replicate,
                                     estimate=fit.x, absolute_error=abs(fit.x-truth), success=bool(fit.success)))
    _, _, A, B = grid()
    result = dict(status='executed NumPy and PyTorch differentiable solver', seed=seed, quick=quick, environment=environment(),
                  config=dict(nx=len(x), nt=len(t), velocity=.35, dispersion=.005, final=1.2, source_width=.04,
                              noise_sd=noise_sd, combined_cfl=A+2*B, boundary='zero Dirichlet at both ends', units='dimensionless'),
                  truth=truth, sensors=sensors, x=x, t=t, observations=z, observation_indices=observation_indices,
                  train_observation_rows=np.flatnonzero(train_times), field=field, fitted_field=prediction,
                  gradient=dict(probe=probe, numerical=numerical, autograd=gradient, relative_error=relative_error),
                  forward_max_error=forward_error, refinement=manufactured_refinement(),
                  conventional=dict(estimate=baseline.x, objective=float(baseline.fun), seconds=baseline_seconds,
                                    forward_evaluations=int(baseline.nfev), success=bool(baseline.success), history=conventional_history),
                  pytorch=dict(estimate=learned.x, objective=float(learned.fun), seconds=torch_seconds,
                               forward_backward_evaluations=int(learned.nfev), success=bool(learned.success), history=autodiff_history),
                  heldout_rmse=heldout_rmse, parameter_absolute_error=abs(learned.x-truth),
                  profile=profile, recovery_sensitivity=recovery)
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output/'river.json', result)
        np.savez_compressed(output/'river-data.npz', sensors=sensors, observation_indices=observation_indices, observations=z)
        plot_lines(output/'river-sensors.svg', t,
                   {f'station x={x[s]:.2f}': prediction[:, s] for s in sensors},
                   'Time (dimensionless)', 'Concentration (dimensionless)', 'Recovered-source prediction at three stations')
        plot_lines(output/'river-profile.svg', locations, {'profile objective': np.array(profile)[:, 2]},
                   'Fixed source position (dimensionless)', 'Scaled mean squared residual', 'Profiled intensity: identifiability diagnostic')
    return result


if __name__ == '__main__':
    args = cli(__doc__)
    result = run(args.seed, args.quick, args.output)
    print('Source truth / recovery:', result['truth'], result['pytorch']['estimate'])
