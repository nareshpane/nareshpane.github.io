"""A bounded, directly executable tour of public Python documentation tools."""
import contextlib
import inspect
import io
import torch


def main():
    print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available())
    tensor = torch.tensor([1., 2.])
    print('Class:', tensor.__class__, '| Module:', torch.nn.Linear.__module__)
    print('Public tensor methods:', [n for n in dir(torch.Tensor)
                                     if not n.startswith('_')][:12])
    for obj in (torch.Tensor, torch.nn.Linear, torch.optim.Adam, torch.autograd.grad):
        print('\nObject:', obj.__name__)
        try:
            print('Signature:', inspect.signature(obj))
        except (TypeError, ValueError):
            print('Native signature unavailable; consult the versioned documentation.')
        print('Doc:', (inspect.getdoc(obj) or '')[:260])
        print('__doc__ present:', bool(obj.__doc__))
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        help(torch.nn.Linear)
    print('\nhelp() excerpt:', captured.getvalue()[:320])
    try:
        print('\nPython source excerpt:', inspect.getsource(torch.nn.Linear.forward)[:320])
    except (OSError, TypeError):
        print('Source is not available in this installation.')
    print('Documentation: https://docs.pytorch.org/docs/stable/')


if __name__ == '__main__':
    main()
