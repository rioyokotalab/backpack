import torch
from backpack import backpack
from backpack.extensions import DiagGGN, DiagGGNMC, DiagGGNExact, KFAC, KFRA, KFLR


class SecondOrderOptimizer(torch.optim.Optimizer):

    def __init__(self, parameters, ext,
                 lr=1e-3, damping=1e-5, curv_ema_decay=0.95, weight_decay=0, momentum=0):
        self.ext = ext

        super().__init__(
            parameters,
            dict(lr=lr, damping=damping, curv_ema_decay=curv_ema_decay,
                 weight_decay=weight_decay, momentum=momentum)
        )

    def step(self, closure=None):

        rst = None

        if closure is not None:
            with backpack(self.ext):
                rst = closure()

        for group in self.param_groups:
            self.update_curvature_ema(group)
            self.precondition_grad(group)
            self.update_preprocess(group)
            self.update(group)

        return rst

    def update_curvature_ema(self, group):
        savefield = self.ext.savefield
        for p in group['params']:
            if p.grad is None:
                continue
            ema_decay = group['curv_ema_decay']
            curv = getattr(p, savefield)
            ema = getattr(p, 'ema', None)

            if ema is None or ema_decay == 1:
                if isinstance(curv, list):
                    ema = [c.clone() for c in curv]
                else:
                    ema = curv.clone()
            else:
                if isinstance(curv, list):
                    ema = [c.mul(ema_decay).add(1 - ema_decay, e)
                           for c, e in zip(curv, ema)]
                else:
                    ema = curv.mul(ema_decay).add(1 - ema_decay, ema)

            p.ema = ema

    @staticmethod
    def precondition_grad(group):
        raise NotImplementedError

    def update_preprocess(self, group):
        state = self.state

        def apply_weight_decay(p):
            weight_decay = group['weight_decay']
            if weight_decay != 0:
                if hasattr(p.grad, 'is_sparse') and p.grad.is_sparse:
                    raise RuntimeError(
                        'weight_decay option is not compatible with sparse gradients')
                p.grad.add_(weight_decay, p.data)

        def apply_momentum(p):
            momentum = group['momentum']

            if momentum != 0:
                param_state = state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(p.grad).detach()
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(p.grad)
                p.grad = buf

        for p in group['params']:
            if p.grad is None:
                continue
            apply_weight_decay(p)
            apply_momentum(p)

    def update(self, group):
        for p in group['params']:
            if p.grad is None:
                continue
            p.data.add_(-group['lr'], p.grad)


class DiagGGNOptimizer(SecondOrderOptimizer):

    def __init__(self, parameters, ext, *args, **kwargs):
        assert isinstance(ext, (DiagGGN, DiagGGNMC, DiagGGNExact))
        super().__init__(parameters, ext, *args, **kwargs)

    @staticmethod
    def precondition_grad(group):
        for p in group['params']:
            if p.grad is None or getattr(p, 'ema', None) is None:
                continue

            prec_grad = p.grad / (p.ema + group['damping'])
            p.grad.copy_(prec_grad)


class KronGGNOptimizer(SecondOrderOptimizer):

    def __init__(self, parameters, ext, *args, tikhonov_damping=True, **kwargs):
        assert isinstance(ext, (KFAC, KFRA, KFLR))
        self._tikhonov_damping = tikhonov_damping
        super().__init__(parameters, ext, *args, **kwargs)

    def precondition_grad(self, group):
        for p in group['params']:
            if p.grad is None or getattr(p, 'ema', None) is None:
                continue

            assert isinstance(p.ema, list)
            damping = group['damping']
            B = p.ema[0]
            if len(p.ema) > 1:
                A = p.ema[1]

                if self._tikhonov_damping:
                    pi = torch.sqrt((A.trace()/A.shape[0])/(B.trace()/B.shape[0]))
                else:
                    pi = 1.
                r = damping ** 0.5

                B_inv = _inv(_add_value_to_diagonal(B, r/pi))
                A_inv = _inv(_add_value_to_diagonal(A, r*pi))
                grad2d = p.grad.view(B_inv.size(0), -1)
                prec_grad = B_inv.mm(grad2d).mm(A_inv)
            else:
                B_inv = _inv(_add_value_to_diagonal(B, damping))
                prec_grad = torch.matmul(B_inv, p.grad)

            p.grad.copy_(prec_grad.reshape_as(p.grad))


def _inv(X):
    u = torch.cholesky(X)
    return torch.cholesky_inverse(u)


def _add_value_to_diagonal(X, value):
    if torch.cuda.is_available():
        indices = torch.cuda.LongTensor([[i, i] for i in range(X.shape[0])])
    else:
        indices = torch.LongTensor([[i, i] for i in range(X.shape[0])])
    values = X.new_ones(X.shape[0]).mul(value)
    return X.index_put(tuple(indices.t()), values, accumulate=True)
