from turtle import xcor
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn

def pgd_whitebox(model, x, y, epsilon, step_num, step_size):
    # random noise
    x_adv = x.detach() + 0.001 * torch.randn(x.shape).cuda().detach()
    # pgd
    for _ in range(step_num):
        x_adv.requires_grad_()
        opt = optim.SGD([x_adv], lr=1e-3)
        opt.zero_grad()
        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(x_adv), y)
        loss.backward()
        grad = x_adv.grad
        x_adv = x_adv + step_size * torch.sign(grad)
        x_adv = torch.min(torch.max(x_adv, x - epsilon), x + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    err = (model(x_adv).data.max(1)[1] != y.data).float().sum()
    print('error pgd whitebox:', err)
    return err

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

def PGD_attack(target_model, source_model, x, y, batch_size, device, mode = 'white'):
    epsilon = 0.03
    step_num = 20
    step_size = 0.003
    batch_num = int(np.ceil(x.shape[0] / batch_size))
    for idx in range(batch_num):
        start_idx = idx * batch_size
        end_idx = min((idx+1) * batch_size, x.shape[0])
        x_data = x[start_idx:end_idx, :].detach().to(device)
        y_data = y[start_idx:end_idx, :].detach().to(device)
        if mode == 'white':
            err = pgd_whitebox(target_model, x_data, y_data, epsilon, step_num, step_num)

