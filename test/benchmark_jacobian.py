import torch
from torch import nn

from backpack import backpack, extend
from backpack.extensions import Jacobian

from timeit import timeit


def forward_postprocess(module, inp, output):
    inp0 = inp[0].clone().detach()

    def backward_hook(grad_output):
        grad_output = grad_output.clone().detach()

        if isinstance(module, nn.Linear):
            if module.weight.requires_grad:
                grads = torch.einsum('bi,bj->bij', grad_output, inp0)  # n x f_out x f_in
                setattr(module.weight, 'grads', grads)  # n x f_out x f_in

            if module.bias is not None and module.bias.requires_grad:
                setattr(module.bias, 'grads', grad_output)  # n x f_out

    if output.requires_grad:
        output.register_hook(backward_hook)


def jacobian_with_loop(model, x):

    output = model(x)

    bs = x.size(0)
    num_outputs = output.size(1)

    # Init Jacobian with zero matrix
    for p in model.parameters():
        jacobian = torch.zeros((num_outputs, bs, *p.data.shape), dtype=p.data.dtype, device=p.data.device)
        setattr(p, 'jacobian', jacobian)

    for i in range(num_outputs):
        retain_graph = False if i == num_outputs - 1 else True

        output_i = output[:, i].sum()
        output_i.backward(retain_graph=retain_graph)

        for p in model.parameters():
            p.jacobian[i] = p.grads


def jacobian_with_backpack(model, x, jac_ext):

    with backpack(jac_ext):
        output = model(x)
        loss = output.sum()
        loss.backward()

    for name, p in model.named_parameters():
        jac = getattr(p, jac_ext.savefield)

        if len(jac) == 1:
            [jac_out] = jac
            setattr(p, 'jacobian', jac_out)
        else:
            jac_out, inp0 = jac
            if jac_out is None:
                setattr(p, 'jacobian', inp0)
            else:
                setattr(p, 'jacobian', torch.einsum('bic,bj->bijc', (jac_out, inp0)))


def main():
    bs = 32
    input_ndim = 1000
    n_hidden = 3
    hidden_ndim = 1000
    output_ndim = 10
    trial = 100

    print(f'bs: {bs}')
    print(f'input_ndim: {input_ndim}')
    print(f'n_hidden: {n_hidden}')
    print(f'hidden_ndim: {hidden_ndim}')
    print(f'output_ndim: {output_ndim}')
    print(f'trial: {trial}')
    print('----------------')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device', device)

    modules = [nn.Linear(input_ndim, hidden_ndim), nn.ReLU()]

    for i in range(n_hidden):
        modules.append(nn.Linear(hidden_ndim, hidden_ndim))
        modules.append(nn.ReLU())

    modules.append(nn.Linear(hidden_ndim, output_ndim))
    model = nn.Sequential(*modules)

    model = model.to(device)

    x = torch.rand(bs, input_ndim, device=device)

    handles = []
    for module in model.children():
        handles.append(module.register_forward_hook(forward_postprocess))

    elapsed = timeit(lambda: jacobian_with_loop(model, x), number=trial)
    print(f'jacobian with loop: {elapsed:.3f}s')

    for handle in handles:
        handle.remove()

    model = extend(model)  # for BackPACK
    ext = Jacobian(start=modules[-1])  # for BackPACK

    elapsed = timeit(lambda: jacobian_with_backpack(model, x, ext), number=trial)
    print(f'jacobian with backpack: {elapsed:.3f}s')


if __name__ == '__main__':
    main()
