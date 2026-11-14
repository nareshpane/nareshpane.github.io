"""Chronological synthetic transaction scoring with logistic and native-PyTorch graph models."""
from __future__ import annotations
import copy
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, precision_recall_curve
import torch
from common import cli, configure, environment, plot_lines, write_json


def generate(seed=42, days=36, accounts=48, per_day=120):
    rng = np.random.default_rng(seed)
    attributes = rng.normal(size=(accounts, 2))
    institutions = rng.integers(0, 4, accounts)
    ring = rng.choice(accounts, size=7, replace=False)
    records = []
    for day in range(days):
        src = rng.integers(0, accounts, per_day)
        dst = rng.integers(0, accounts, per_day)
        dst = np.where(src == dst, (dst+1)%accounts, dst)
        # Latent regime affects relationships, with overlapping amount distributions.
        injected = rng.random(per_day) < (.10 if day < 28 else .13)
        index = rng.integers(0, len(ring), per_day)
        src[injected] = ring[index[injected]]
        dst[injected] = ring[(index[injected]+1)%len(ring)]
        label = (injected & (rng.random(per_day) < .75)).astype(int)
        amount = rng.lognormal(3.4+.18*injected, .75, per_day)
        times = day+np.sort(rng.uniform(.01, .99, per_day))
        records.append(np.column_stack([times, src, dst, amount, label]))
    return np.vstack(records), attributes, institutions


def numpy_aggregate(features, source, destination):
    output = np.zeros_like(features)
    np.add.at(output, destination, features[source])
    degree = np.bincount(destination, minlength=len(features))
    return output/np.maximum(1, degree[:, None])


def snapshots(records, attributes, institutions, days):
    n = len(attributes)
    result = []
    for day in range(3, days):
        history = records[(records[:, 0] >= day-3) & (records[:, 0] < day)]
        candidates = records[(records[:, 0] >= day) & (records[:, 0] < day+1)]
        assert history[:, 0].max() < candidates[:, 0].min()
        hs, hd = history[:, 1:3].astype(int).T
        incoming, outgoing = np.bincount(hd, minlength=n), np.bincount(hs, minlength=n)
        amount = np.zeros(n)
        np.add.at(amount, hs, history[:, 3])
        # All features come from preceding three days plus fixed account attributes.
        x = np.column_stack([np.log1p(incoming), np.log1p(outgoing),
                             np.log1p(amount)/5, attributes])
        src, dst = candidates[:, 1:3].astype(int).T
        ef = np.column_stack([np.log(candidates[:, 3])/5, institutions[src] == institutions[dst]])
        neighbor = numpy_aggregate(x, hs, hd)
        engineered = np.column_stack([x[src], x[dst], abs(x[src]-x[dst]), neighbor[src], neighbor[dst], ef])
        result.append(dict(day=day, x=x, hs=hs, hd=hd, src=src, dst=dst,
                           ef=ef, baseline=engineered, y=candidates[:, 4].astype(int),
                           candidate_times=candidates[:, 0], history_max_time=float(history[:, 0].max())))
    return result


class GraphScorer(torch.nn.Module):
    def __init__(self, width=12):
        super().__init__()
        self.self_layer = torch.nn.Linear(5, width)
        self.neighbor_layer = torch.nn.Linear(5, width, bias=False)
        self.edge = torch.nn.Sequential(torch.nn.Linear(2*width+2, width), torch.nn.Tanh(), torch.nn.Linear(width, 1))

    def embed(self, x, hs, hd):
        aggregate = torch.zeros_like(x).index_add(0, hd, x[hs])
        degree = torch.bincount(hd, minlength=len(x)).clamp_min(1)[:, None]
        aggregate = aggregate/degree
        return torch.tanh(self.self_layer(x)+self.neighbor_layer(aggregate)), aggregate

    def forward(self, graph):
        h, _ = self.embed(graph['x'], graph['hs'], graph['hd'])
        joined = torch.cat([h[graph['src']], h[graph['dst']], graph['ef']], dim=1)
        return self.edge(joined).squeeze(1)


def metrics(y, probability, threshold):
    predicted = probability >= threshold
    tp = np.sum(predicted & (y == 1))
    fp = np.sum(predicted & (y == 0))
    fn = np.sum(~predicted & (y == 1))
    capacity = max(1, int(.1*len(y)))
    selected = np.argsort(-probability)[:capacity]
    bins = np.minimum(9, (10*probability).astype(int))
    ece = sum(np.mean(bins == b)*abs(probability[bins == b].mean()-y[bins == b].mean())
              for b in range(10) if np.any(bins == b))
    precision, recall, _ = precision_recall_curve(y, probability)
    return dict(n=len(y), prevalence=float(y.mean()), average_precision=float(average_precision_score(y, probability)),
                precision=float(tp/max(1, tp+fp)), recall=float(tp/max(1, tp+fn)),
                false_positive_rate=float(fp/max(1, np.sum(y == 0))),
                precision_at_10_percent=float(y[selected].mean()), brier=float(np.mean((probability-y)**2)),
                ece_10_bins=float(ece), threshold=float(threshold),
                pr_curve=dict(precision=precision[::max(1, len(precision)//100)], recall=recall[::max(1, len(recall)//100)]))


def validation_threshold(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2*precision[:-1]*recall[:-1]/np.maximum(1e-15, precision[:-1]+recall[:-1])
    return float(thresholds[np.argmax(f1)])


def run(seed=42, quick=False, output=None):
    configure(seed)
    days, count = (24, 80) if quick else (36, 120)
    records, attributes, institutions = generate(seed, days=days, per_day=count)
    graphs = snapshots(records, attributes, institutions, days)
    a, b = int(.60*len(graphs)), int(.80*len(graphs))
    train, valid, test = graphs[:a], graphs[a:b], graphs[b:]
    scaler = StandardScaler().fit(np.vstack([g['baseline'] for g in train]))
    node_scaler = StandardScaler().fit(np.vstack([g['x'] for g in train]))
    def join(group, key):
        return np.concatenate([g[key] for g in group])
    start = time.perf_counter()
    baseline = LogisticRegression(C=1., max_iter=500).fit(scaler.transform(join(train, 'baseline')), join(train, 'y'))
    baseline_seconds = time.perf_counter()-start
    vp = baseline.predict_proba(scaler.transform(join(valid, 'baseline')))[:, 1]
    bt = validation_threshold(join(valid, 'y'), vp)
    bp = baseline.predict_proba(scaler.transform(join(test, 'baseline')))[:, 1]
    for g in graphs:
        g['tensor'] = {k: torch.tensor(node_scaler.transform(v) if k == 'x' else v)
                       for k, v in g.items() if k in ('x', 'hs', 'hd', 'src', 'dst', 'ef', 'y')}
        g['tensor']['y'] = g['tensor']['y'].to(torch.float64)
    model = GraphScorer()
    optimizer = torch.optim.Adam(model.parameters(), lr=.012, weight_decay=.001)
    def predict(group):
        model.eval()
        with torch.inference_mode():
            return np.concatenate([model(g['tensor']).sigmoid().numpy() for g in group])
    best, state, history = np.inf, None, []
    epochs = 25 if quick else 160
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = [torch.nn.functional.binary_cross_entropy_with_logits(model(g['tensor']), g['tensor']['y']) for g in train]
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
        pv = predict(valid)
        vy = join(valid, 'y')
        validation = float(-np.mean(vy*np.log(pv+1e-15)+(1-vy)*np.log(1-pv+1e-15)))
        history.append([float(loss.detach()), validation])
        if validation < best:
            best, state = validation, copy.deepcopy(model.state_dict())
    neural_seconds = time.perf_counter()-start
    model.load_state_dict(state)
    nt = validation_threshold(join(valid, 'y'), predict(valid))
    npred = predict(test)
    example = test[0]
    with torch.inference_mode():
        embedding, aggregate = model.embed(example['tensor']['x'], example['tensor']['hs'], example['tensor']['hd'])
    conventional_aggregate = numpy_aggregate(example['tensor']['x'].numpy(), example['hs'], example['hd'])
    parity = float(np.max(abs(aggregate.numpy()-conventional_aggregate)))
    assert parity < 1e-12
    # Entirely new accounts and new transaction realizations: generator transfer audit.
    shift_records, shift_attrs, shift_institutions = generate(seed+1000, days=12, per_day=count)
    shift = snapshots(shift_records, shift_attrs, shift_institutions, 12)
    for g in shift:
        g['tensor'] = {k: torch.tensor(node_scaler.transform(v) if k == 'x' else v)
                       for k, v in g.items() if k in ('x', 'hs', 'hd', 'src', 'dst', 'ef')}
    sp = baseline.predict_proba(scaler.transform(join(shift, 'baseline')))[:, 1]
    result = dict(status='executed conventional and PyTorch CPU', seed=seed, quick=quick, environment=environment(),
                  counts=dict(accounts=len(attributes), events=len(records), snapshots=len(graphs),
                              train_days=[g['day'] for g in train], validation_days=[g['day'] for g in valid], test_days=[g['day'] for g in test]),
                  conventional=metrics(join(test, 'y'), bp, bt), neural=metrics(join(test, 'y'), npred, nt),
                  new_accounts=dict(conventional=metrics(join(shift, 'y'), sp, bt), neural=metrics(join(shift, 'y'), predict(shift), nt)),
                  baseline_training_seconds=baseline_seconds, neural_training_seconds=neural_seconds,
                  aggregation_max_error=parity, neural_epochs=epochs, history=history,
                  parameter_counts=dict(conventional=int(baseline.coef_.size+1), neural=sum(p.numel() for p in model.parameters())),
                  gallery=dict(day=example['day'], x=example['tensor']['x'].numpy(), hs=example['hs'], hd=example['hd'],
                               src=example['src'], dst=example['dst'], ef=example['ef'], y=example['y'],
                               candidate_times=example['candidate_times'], history_max_time=example['history_max_time'],
                               aggregate=aggregate.numpy(), embedding=embedding.numpy(),
                               probability=npred[:len(example['y'])], baseline_probability=bp[:len(example['y'])]),
                  test_labels=join(test, 'y'), test_probability=npred, baseline_probability=bp)
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output/'financial-network.json', result)
        np.savez_compressed(output/'financial-data.npz', records=records, attributes=attributes, institutions=institutions)
        plot_lines(output/'financial-training.svg', np.arange(epochs),
                   {'training BCE': np.array(history)[:, 0], 'validation BCE': np.array(history)[:, 1]},
                   'Optimization epoch', 'Binary cross-entropy per edge', 'Graph model: historical information only')
        # Independent precision-recall coordinates; conventional and neural share test edges.
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8.8, 3.5), layout='constrained')
        for name, pred, color in [('Logistic', bp, '#B57A60'), ('Message passing', npred, '#477DA5')]:
            precision, recall, _ = precision_recall_curve(join(test, 'y'), pred)
            ax.step(recall, precision, where='post', label=name, color=color)
        ax.axhline(join(test, 'y').mean(), linestyle=':', color='#8D78A5', label='Prevalence')
        ax.set(xlabel='Recall', ylabel='Precision', title='Chronological test: synthetic latent-regime labels', xlim=(0, 1), ylim=(0, 1))
        ax.legend(frameon=False)
        fig.savefig(output/'financial-pr.svg', metadata={'Date': None})
        plt.close(fig)
    return result


if __name__ == '__main__':
    args = cli(__doc__)
    result = run(args.seed, args.quick, args.output)
    print('Graph average precision:', result['conventional']['average_precision'], result['neural']['average_precision'])
