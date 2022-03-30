from turtle import xcor
import torch
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn

def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon,
                  num_steps,
                  step_size):
    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    
    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(X.device)
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum()
    print('err pgd (white-box): ', err_pgd)
    return err, err_pgd

def pgd_blackbox(target_model, source_model, x, y, epsilon, step_num, step_size):
    # random noise
    x_adv = x.detach() + 0.001 * torch.randn(x.shape).cuda().detach()
    # pgd
    for _ in range(step_num):
        x_adv.requires_grad_()
        opt = optim.SGD([x_adv], lr=1e-3)
        opt.zero_grad()
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(source_model(x_adv), y)
        loss.backward()
        grad = x_adv.grad
        x_adv = x_adv + step_size * torch.sign(grad)
        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    err = (target_model(x_adv).data.max(1)[1] != y.data).float().sum()
    print('error pgd whitebox:', err)
    return err

def eval_adv_test_whitebox(model, device, test_loader, epsilon, num_steps, step_size):
    """
    evaluate model by white-box attack
    """
    model.eval()
    robust_err_total = 0
    natural_err_total = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        err_natural, err_robust = _pgd_whitebox(model, X, y, epsilon, num_steps, step_size)
        robust_err_total += err_robust
        natural_err_total += err_natural
    return robust_err_total
    print('natural_err_total: ', natural_err_total)
    print('robust_err_total: ', robust_err_total)
