import torch
from torch import nn

from backpack import extend, backpack
from backpack.extensions import BatchGrad, SumGradSquared, EmpKFAC, KFAC, KFLR
from backpack.extensions import FAIL_WARN


class LeNet(nn.Module):

    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.relu1 = nn.ReLU()
        self.max_pool2d1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.max_pool2d2 = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc3 = nn.Linear(16*5*5, 120)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(84, num_classes)

    def forward(self, x):
        h = self.relu1(self.conv1(x))
        h = self.max_pool2d1(h)
        h = self.relu2(self.conv2(h))
        h = self.max_pool2d2(h)
        h = self.flatten(h)
        h = self.relu3(self.fc3(h))
        h = self.relu4(self.fc4(h))
        out = self.fc5(h)

        return out


num_classes = 10
model = extend(LeNet(num_classes=num_classes))
loss_fn = extend(nn.CrossEntropyLoss())

bs = 10
x = torch.randn(bs, 3, 32, 32)
target = torch.randint(low=0, high=num_classes-1, size=(bs,))

ext_bg = BatchGrad()
ext_sgs = SumGradSquared(loss_reduction=loss_fn.reduction)
ext_empkfac = EmpKFAC(loss_reduction=loss_fn.reduction)
ext_mckfac = KFAC(fail_mode=FAIL_WARN, mc_samples=1)
ext_kflr = KFLR(fail_mode=FAIL_WARN)

with backpack(ext_bg, ext_sgs, ext_empkfac, ext_mckfac, ext_kflr):
    loss = loss_fn(model(x), target)
    loss.backward()


def compare(dsc, v1, v2):
    diff = v1 - v2
    max_diff = diff.abs().max()
    relative_diff = diff.norm() / v2.norm()
    print(f'{dsc}\n max_diff: {max_diff}, relative_diff: {relative_diff}')


for name, module in model.named_children():
    if len(list(module.parameters())) == 0:
        continue
    print('------------')
    print(name)
    for param in module.parameters():
        sum_grad_square = getattr(param, ext_sgs.savefield)
        emp_kron_factors = getattr(param, ext_empkfac.savefield)
        mc_kron_factors = getattr(param, ext_mckfac.savefield)
        exact_kron_factors = getattr(param, ext_kflr.savefield)
        if len(emp_kron_factors) > 1:
            A_emp = emp_kron_factors[1]
            A_mc = mc_kron_factors[1]
            A_exact = exact_kron_factors[1]
            assert (A_emp - A_mc).abs().max() == 0
            assert (A_emp - A_exact).abs().max() == 0
            B_emp = emp_kron_factors[0]
            B_mc = mc_kron_factors[0]
            B_exact = exact_kron_factors[0]
            compare('B_emp vs B_mc', B_emp, B_mc)
            compare('B_emp vs B_exact', B_emp, B_exact)
            compare('B_mc vs B_exact', B_mc, B_exact)

            diag_B = torch.diag(B_emp)
            diag_A = torch.diag(A_emp)
            kron_sum_grad_square = torch.einsum('i,j->ij', diag_B, diag_A).view_as(param.grad)
            compare('diag(empkfac) vs empdiag', kron_sum_grad_square, sum_grad_square.div(bs))

            grad_batch = getattr(param, ext_bg.savefield)
            compare('grad_batch.sum() vs grad', grad_batch.sum(0), param.grad)

            grad_batch *= bs
            grad_batch_squre = grad_batch ** 2
            compare('grad_batch**2.sum() vs sum_grad_square', grad_batch_squre.sum(0), sum_grad_square)

