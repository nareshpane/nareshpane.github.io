"""Exact arithmetic, tensor shapes, autograd, torch.func, and state round-trip checks."""
from __future__ import annotations
import io
import numpy as np
import torch
from common import ROOT, cli, configure, environment, write_json


def manual_step(w=1., b=0., rate=.05):
    x, y = 2., 5.
    prediction = w*x+b
    residual = prediction-y
    gradient = np.array([residual*x, residual])
    updated = np.array([w, b])-rate*gradient
    return dict(prediction=prediction, residual=residual, loss=.5*residual**2,
                gradient=gradient, updated=updated,
                updated_prediction=float(updated[0]*x+updated[1]),
                updated_loss=float(.5*(updated[0]*x+updated[1]-y)**2))


def autograd_step():
    theta = torch.tensor([1., 0.], requires_grad=True)
    loss = .5*(2*theta[0]+theta[1]-5)**2
    loss.backward()
    gradient = theta.grad.detach().numpy().copy()
    with torch.no_grad():
        theta -= .05*theta.grad
    return gradient, theta.detach().numpy()


def function_transforms():
    from torch.func import grad, vmap, jacrev, jacfwd, jvp, vjp, functional_call
    def f(x):
        return torch.stack((x[0]*x[1], x[0]**2+torch.sin(x[1])))
    x = torch.tensor([2., .3])
    jacobian = jacrev(f)(x)
    assert torch.allclose(jacobian, jacfwd(f)(x))
    direction = torch.tensor([1., -1.])
    _, tangent = jvp(f, (x,), (direction,))
    _, pullback = vjp(f, x)
    assert torch.allclose(tangent, jacobian@direction)
    assert torch.allclose(pullback(direction)[0], direction@jacobian)
    batch = torch.arange(12.).reshape(4, 3)/10
    scalar = lambda z: z.sin().sum()
    loop = torch.stack([grad(scalar)(row) for row in batch])
    batched = vmap(grad(scalar))(batch)
    assert torch.equal(loop, batched)
    layer = torch.nn.Linear(3, 1)
    params = dict(layer.named_parameters())
    assert torch.equal(functional_call(layer, params, (batch,)), layer(batch))
    hessian = jacfwd(jacrev(lambda z: (z*z).sum()))(x)
    assert torch.equal(hessian, 2*torch.eye(2))
    return dict(jacobian=jacobian.numpy(), jvp=tangent.numpy(),
                vjp=pullback(direction)[0].numpy(), vmap_max_error=float((loop-batched).abs().max()))


def run(seed=42):
    configure(seed)
    exact = manual_step()
    np.testing.assert_array_equal(exact['gradient'], [-6., -3.])
    np.testing.assert_allclose(exact['updated'], [1.3, .15], atol=1e-15, rtol=0)
    assert exact['loss'] == 4.5 and exact['updated_loss'] == 2.53125
    gradient, theta = autograd_step()
    np.testing.assert_allclose(gradient, exact['gradient'], atol=0, rtol=0)
    np.testing.assert_allclose(theta, exact['updated'], atol=1e-15, rtol=0)
    x, w = torch.ones(3, 2), torch.ones(2, 4)
    assert (x@w).shape == (3, 4)
    assert (x+torch.tensor([1., 2.])).shape == (3, 2)
    a = torch.arange(6).reshape(2, 3)
    assert not torch.equal(a.reshape(3, 2), a.permute(1, 0))
    model = torch.nn.Linear(2, 1)
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = torch.nn.Linear(2, 1)
    restored.load_state_dict(torch.load(stream, weights_only=True))
    assert torch.equal(model(x), restored(x))
    # Central differences: independently check a smooth scalar derivative.
    steps = np.logspace(-1, -12, 12)
    errors = [abs((np.sin(.7+h)-np.sin(.7-h))/(2*h)-np.cos(.7)) for h in steps]
    return dict(status='executed PyTorch and analytical validation', seed=seed,
                environment=environment(), exact=exact, transforms=function_transforms(),
                finite_difference=dict(steps=steps, absolute_errors=errors),
                checks=['exact scalar update', 'autograd parity', 'shapes and broadcasting',
                        'reshape versus permutation', 'torch.func identities', 'weights_only round-trip'])


if __name__ == '__main__':
    args = cli(__doc__)
    result = run(args.seed)
    write_json(args.output / 'foundations.json', result)
    print('Foundations:', ', '.join(result['checks']))
