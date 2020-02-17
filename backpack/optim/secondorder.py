import torch
from backpack import backpack
from backpack.extensions import DiagGGN, DiagGGNMC, DiagGGNExact


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
