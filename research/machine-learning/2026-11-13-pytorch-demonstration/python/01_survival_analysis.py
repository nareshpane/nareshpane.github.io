"""Synthetic discrete-time competing risks: analytical and automatic gradients, neural extension."""
from __future__ import annotations
import copy
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
import torch
from common import cli, configure, environment, plot_lines, write_json

H = 12


def generate(n=1800, seed=42):
    rng = np.random.default_rng(seed)
    covariates = rng.normal(size=(n, 2))
    design = np.column_stack([np.ones(n), covariates])
    t = np.arange(H)[None, :]
    age, marker = covariates.T
    logits = np.zeros((n, H, 3))
    logits[:, :, 1] = -3.1+.07*t+.5*age[:, None]+.65*marker[:, None]+.5*np.sin(2*marker[:, None])
    logits[:, :, 2] = -3.5+.1*t+.25*age[:, None]-.5*marker[:, None]+.3*(marker[:, None]**2-1)
    p = softmax(logits, axis=2)
    latent_time = np.full(n, H+1, dtype=int)
    latent_cause = np.zeros(n, dtype=int)
    for h in range(H):
        u = rng.random(n)
        event = (u >= p[:, h, 0]) & (latent_time == H+1)
        cause = np.where(u < p[:, h, 0]+p[:, h, 1], 1, 2)
        latent_time[event], latent_cause[event] = h+1, cause[event]
    # Independent censoring at END of bin C; event in bin C is observed.
    censor = np.where(rng.random(n)<.25, rng.integers(3, H, n), H)
    times = np.minimum(latent_time, censor)
    causes = np.where(latent_time <= censor, latent_cause, 0)
    return dict(x=design, times=times, causes=causes,
                latent_time=latent_time, latent_cause=latent_cause, censor=censor)


def targets(times, causes):
    risk = np.arange(1, H+1)[None, :] <= times[:, None]
    label = np.zeros(risk.shape, dtype=int)
    events = causes > 0
    label[np.flatnonzero(events), times[events]-1] = causes[events]
    return risk, label


def numpy_objective(flat, x, times, causes, penalty=.001):
    weights = flat.reshape(H, x.shape[1], 2)
    event_logits = np.einsum('np,hpk->nhk', x, weights)
    logits = np.concatenate([np.zeros((*event_logits.shape[:2], 1)), event_logits], axis=2)
    logp = logits-logsumexp(logits, axis=2, keepdims=True)
    risk, label = targets(times, causes)
    selected = np.take_along_axis(logp, label[:, :, None], axis=2)[:, :, 0]
    value = -(selected*risk).sum()/len(x)+penalty*np.sum(weights**2)/2
    residual = np.exp(logp)
    residual[np.arange(len(x))[:, None], np.arange(H)[None, :], label] -= 1
    gradient = np.einsum('np,nhk->hpk', x, residual[:, :, 1:]*risk[:, :, None])/len(x)
    return float(value), (gradient+penalty*weights).ravel()


def torch_objective(weights, x, times, causes, penalty=.001):
    event_logits = torch.einsum('np,hpk->nhk', x, weights)
    logits = torch.cat([torch.zeros((*event_logits.shape[:2], 1)), event_logits], dim=2)
    logp = torch.log_softmax(logits, dim=2)
    risk = torch.arange(1, H+1)[None, :] <= times[:, None]
    label = torch.zeros(risk.shape, dtype=torch.long)
    events = causes > 0
    label[torch.where(events)[0], times[events]-1] = causes[events]
    return -(logp.gather(2, label[:, :, None]).squeeze(2)*risk).sum()/len(x)+penalty*weights.square().sum()/2


def probabilities(weights, x):
    event = np.einsum('np,hpk->nhk', x, weights)
    return softmax(np.concatenate([np.zeros((*event.shape[:2], 1)), event], 2), axis=2)


def incidence(p):
    survival = np.cumprod(p[:, :, 0], axis=1)
    before = np.column_stack([np.ones(len(p)), survival[:, :-1]])
    cif = np.cumsum(before[:, :, None]*p[:, :, 1:], axis=1)
    np.testing.assert_allclose(survival+cif.sum(2), 1, atol=2e-14)
    return survival, cif


def censor_survival(t):
    # P(C >= t), known by construction; early C is uniform on {3,...,11}.
    t = np.asarray(t)
    return .75+.25*np.maximum(0, H-np.maximum(t, 3))/(H-3)


def evaluate(p, data, indices, horizon=8):
    x, times, causes = (data[k][indices] for k in ('x', 'times', 'causes'))
    risk, label = targets(times, causes)
    nll = -np.sum(np.log(np.maximum(1e-300, np.take_along_axis(p, label[:, :, None], 2)[:, :, 0]))*risk)/len(p)
    _, cif = incidence(p)
    observed_event = (causes > 0) & (times <= horizon)
    known_free = (~observed_event) & (times >= horizon)
    weight = observed_event/censor_survival(times)+known_free/censor_survival(horizon)
    observed_y = ((causes == 1) & (times <= horizon)).astype(float)
    brier = np.mean(weight*(observed_y-cif[:, horizon-1, 0])**2)
    oracle = ((data['latent_cause'][indices] == 1) & (data['latent_time'][indices] <= horizon)).astype(float)
    groups = np.array_split(np.argsort(cif[:, horizon-1, 0]), 5)
    calibration = [dict(n=len(g), predicted=float(cif[g, horizon-1, 0].mean()),
                        latent_frequency=float(oracle[g].mean())) for g in groups]
    return dict(nll=float(nll), ipcw_brier_cause1_h8=float(brier), calibration=calibration)


class NeuralHazard(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.Tanh(), torch.nn.Linear(16, H*3))

    def forward(self, x):
        return self.network(x[:, 1:]).reshape(-1, H, 3)


def run(seed=42, quick=False, output=None):
    configure(seed)
    data = generate(600 if quick else 1800, seed)
    n = len(data['x'])
    order = np.random.default_rng(seed+1).permutation(n)
    train, valid, test = np.split(order, [int(.6*n), int(.8*n)])
    x, times, causes = (data[k][train] for k in ('x', 'times', 'causes'))
    tx, tt, tc = torch.tensor(x), torch.tensor(times), torch.tensor(causes)
    initial = np.zeros(H*3*2)
    probe = np.random.default_rng(seed+2).normal(0, .2, len(initial))
    leaf = torch.tensor(probe.reshape(H, 3, 2), requires_grad=True)
    loss = torch_objective(leaf, tx, tt, tc)
    gradient = torch.autograd.grad(loss, leaf)[0].numpy().ravel()
    value, analytical = numpy_objective(probe, x, times, causes)
    parity = float(np.max(abs(gradient-analytical)))
    np.testing.assert_allclose(gradient, analytical, atol=1e-12, rtol=1e-10)
    assert abs(loss.item()-value) < 1e-12
    options = dict(maxiter=80 if quick else 250, ftol=1e-11, gtol=1e-7)
    start = time.perf_counter()
    conventional = minimize(numpy_objective, initial, args=(x, times, causes), jac=True, method='L-BFGS-B', options=options)
    numpy_seconds = time.perf_counter()-start
    def auto(flat):
        w = torch.tensor(flat.reshape(H, 3, 2), requires_grad=True)
        objective = torch_objective(w, tx, tt, tc)
        return objective.item(), torch.autograd.grad(objective, w)[0].numpy().ravel()
    start = time.perf_counter()
    equivalent = minimize(auto, initial, jac=True, method='L-BFGS-B', options=options)
    torch_seconds = time.perf_counter()-start
    np.testing.assert_allclose(conventional.fun, equivalent.fun, atol=1e-9)
    model = NeuralHazard()
    optimizer = torch.optim.Adam(model.parameters(), lr=.018)
    risk, label = targets(data['times'], data['causes'])
    trisk, tlabel, allx = torch.tensor(risk), torch.tensor(label), torch.tensor(data['x'])
    history, best, state = [], np.inf, None
    epochs = 50 if quick else 350
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logs = model(allx[train]).log_softmax(2)
        objective = -(logs.gather(2, tlabel[train, :, None]).squeeze(2)*trisk[train]).sum()/len(train)
        objective += .0005*sum(w.square().sum() for w in model.parameters())
        objective.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            logs = model(allx[valid]).log_softmax(2)
            score = -(logs.gather(2, tlabel[valid, :, None]).squeeze(2)*trisk[valid]).sum()/len(valid)
        history.append([float(objective.detach()), float(score)])
        if score < best:
            best, state = float(score), copy.deepcopy(model.state_dict())
    neural_seconds = time.perf_counter()-start
    model.load_state_dict(state)
    model.eval()
    with torch.inference_mode():
        pn = model(allx[test]).softmax(2).numpy()
    pc = probabilities(conventional.x.reshape(H, 3, 2), data['x'][test])
    profiles = np.array([[1., -.8, -.8], [1., 0., 0.], [1., .8, .8]])
    with torch.inference_mode():
        profile_neural = model(torch.tensor(profiles)).softmax(2).numpy()
    result = dict(status='executed conventional and PyTorch CPU', seed=seed, quick=quick,
                  environment=environment(), n=n, splits=dict(train=train, valid=valid, test=test),
                  events=np.bincount(data['causes'], minlength=3), gradient_max_error=parity,
                  equivalent=dict(numpy_objective=float(conventional.fun), torch_objective=float(equivalent.fun),
                                  numpy_seconds=numpy_seconds, torch_seconds=torch_seconds,
                                  numpy_success=bool(conventional.success), torch_success=bool(equivalent.success),
                                  max_parameter_difference=float(np.max(abs(conventional.x-equivalent.x)))),
                  conventional=evaluate(pc, data, test), neural=evaluate(pn, data, test),
                  neural_training_seconds=neural_seconds, neural_epochs=epochs,
                  parameter_counts=dict(conventional=len(initial), neural=sum(p.numel() for p in model.parameters())),
                  profiles=dict(x=profiles, conventional=probabilities(conventional.x.reshape(H, 3, 2), profiles), neural=profile_neural),
                  test_predictions=dict(conventional=pc, neural=pn, times=data['times'][test], causes=data['causes'][test]),
                  timelines=[dict(time=int(data['times'][i]), cause=int(data['causes'][i])) for i in test[:14]],
                  history=history, weights=conventional.x.reshape(H, 3, 2))
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output/'survival.json', result)
        np.savez_compressed(output/'survival-data.npz', **data)
        plot_lines(output/'survival-training.svg', np.arange(epochs),
                   {'training + penalty': np.array(history)[:, 0], 'validation NLL': np.array(history)[:, 1]},
                   'Optimization epoch', 'Negative log-likelihood per subject', 'Neural hazard: selection uses validation only')
        s, f = incidence(result['profiles']['neural'])
        plot_lines(output/'survival-curves.svg', np.arange(1, H+1),
                   {'S: middle profile': s[1], 'F1: middle profile': f[1, :, 0], 'F2: middle profile': f[1, :, 1]},
                   'Time bin (synthetic)', 'Probability', 'Executed neural competing-risk prediction')
    return result


if __name__ == '__main__':
    args = cli(__doc__)
    result = run(args.seed, args.quick, args.output)
    print('Survival NLL:', result['conventional']['nll'], result['neural']['nll'])
