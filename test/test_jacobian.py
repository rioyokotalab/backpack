import torch
from torch import nn

from backpack import backpack, extend
from backpack.extensions import Jacobian, DiagGGNExact, LossHessianExact
from backpack.utils.utils import einsum


def test(model, loss_fn, x, target, jac_ext):

    diag_ggn_ext = DiagGGNExact()
    loss_hess_ext = LossHessianExact()

    with backpack(jac_ext, diag_ggn_ext, loss_hess_ext):
        output = model(x)
        loss = loss_fn(output, target)
        loss.backward()

    hess = loss_hess_ext.hessian

    for module in model.children():
        if isinstance(module, nn.Linear):
            w = module.weight
            diag_ggn_w = getattr(w, diag_ggn_ext.savefield)
            jac_w = getattr(w, jac_ext.savefield)
            jac_out, inp = jac_w
            if jac_out is None:
                jac_out = hess
            else:
                jac_out = einsum('bik,bkc->bic', (jac_out, hess))

            diag_ggn_w_test = einsum('bic,bj->ij', (jac_out ** 2, inp ** 2))
            err = (diag_ggn_w - diag_ggn_w_test).abs().max()
            assert err < 1e-8, err

            b = module.bias
            diag_ggn_b = getattr(b, diag_ggn_ext.savefield)
            [jac_out] = getattr(b, jac_ext.savefield)
            if jac_out is None:
                jac_out = hess
            else:
                jac_out = einsum('bik,bkc->bic', (jac_out, hess))

            diag_ggn_b_test = einsum('bic->i', (jac_out ** 2,))
            err = (diag_ggn_b - diag_ggn_b_test).abs().max()
            assert err < 1e-8, err


def main():
    bs = 32
    input_ndim = 100
    n_hidden = 3
    hidden_ndim = 1000
    output_ndim = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device', device)

    modules = [nn.Linear(input_ndim, hidden_ndim), nn.ReLU()]

    for i in range(n_hidden):
        modules.append(nn.Linear(hidden_ndim, hidden_ndim))
        modules.append(nn.ReLU())

    modules.append(nn.Linear(hidden_ndim, output_ndim))
    model = nn.Sequential(*modules)

    model = extend(model)
    loss_fn = extend(nn.CrossEntropyLoss())

    model = model.to(device)
    loss_fn = loss_fn.to(device)

    x = torch.rand(bs, input_ndim, device=device)
    target = torch.LongTensor(torch.randint(output_ndim - 1, (bs,)))
    target = target.to(device)

    ext = Jacobian(start=modules[-1])
    test(model, loss_fn, x, target, ext)


if __name__ == '__main__':
    main()
