import torch
from torch import nn

from backpack import backpack, extend
from backpack.extensions import KFLR, KFLRFR, DiagGGNExact, DiagGGNExactFR
from backpack.core.layers import Flatten


def test(model, loss_fn, x, target, ext1, ext2):

    with backpack(ext1):
        output = model(x)
        loss = loss_fn(output, target)
        loss.backward()

    for param in model.parameters():
        value = getattr(param, ext1.savefield)
        setattr(param, 'last_value', value)

    # 1st run
    with backpack(ext2):
        output = model(x)
        loss = loss_fn(output, target)
        loss.backward()

    for module in model.children():
        for p_name, param in module.named_parameters():
            value = getattr(param, ext2.savefield)
            last_value = getattr(param, 'last_value')

            if isinstance(last_value, list):
                for i, (v, last_v) in enumerate(zip(value, last_value)):
                    err = (v - last_v).abs().max()
                    assert err == 0, f'{module.__class__.__name__}.{p_name} factor{i} err={err}'
            else:
                err = (value - last_value).abs().max()
                assert err == 0, f'{module.__class__.__name__}.{p_name} err={err}'

    # 2nd run
    with backpack(ext2):
        output = model(x)
        loss = loss_fn(output, target)
        loss.backward()

    for module in model.children():
        for p_name, param in module.named_parameters():
            value = getattr(param, ext2.savefield)
            last_value = getattr(param, 'last_value')

            if isinstance(last_value, list):
                for i, (v, last_v) in enumerate(zip(value, last_value)):
                    err = (v - last_v).abs().max()
                    assert err == 0, f'{module.__class__.__name__}.{p_name} factor{i} err={err}'
            else:
                err = (value - last_value).abs().max()
                assert err == 0, f'{module.__class__.__name__}.{p_name} err={err}'


def main():
    bs = 32
    input_shape = (3, 24, 24)
    n_hidden = 1
    hidden_ndim = 1000
    output_ndim = 1000
    loop = 100

    print('bs', bs)
    print('input_shape', input_shape)
    print('n_hidden', n_hidden)
    print('hidden_ndim', hidden_ndim)
    print('output_ndim', output_ndim)
    print('loop', loop)
    print('--------')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device', device)

    modules = [
        nn.Conv2d(input_shape[0], 32, 3),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Conv2d(32, 32, 3),
        nn.MaxPool2d(2),
        nn.ReLU(),
        Flatten(),
        nn.Linear(512, hidden_ndim),
        nn.ReLU()
    ]

    for i in range(n_hidden):
        modules.append(nn.Linear(hidden_ndim, hidden_ndim))
        modules.append(nn.ReLU())

    modules.append(nn.Linear(hidden_ndim, output_ndim))
    model = nn.Sequential(*modules)

    model = extend(model)
    loss_fn = extend(nn.CrossEntropyLoss())

    model = model.to(device)
    loss_fn = loss_fn.to(device)

    x = torch.rand(bs, *input_shape, device=device)
    target = torch.LongTensor(torch.randint(output_ndim - 1, (bs,)))
    target = target.to(device)

    test(model, loss_fn, x, target, KFLR(), KFLRFR())
    test(model, loss_fn, x, target, DiagGGNExact(), DiagGGNExactFR())


if __name__ == '__main__':
    main()
