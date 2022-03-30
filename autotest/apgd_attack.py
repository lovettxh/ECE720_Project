import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import operator

class APGD_attack():
    def __init__(self, model, device, iter_step, epsilon=0.3):
        self.model = model
        self.device = device
        self.iter_step = iter_step
        self.epsilon = epsilon
        self.restarts = 1
    def setup(self, x):
        self.dims = len(list(x.shape[1:]))
        self.iter_step2 = max(int(0.22 * self.iter_step),1)
        self.iter_decr = max(int(0.03 * self.iter_step), 1)
        self.iter_min = max(int(0.06 * self.iter_step), 1)

    def L1_projection(self, x2, y2, eps1):
        '''
        x2: center of the L1 ball (bs x input_dim)
        y2: current perturbation (x2 + y2 is the point to be projected)
        eps1: radius of the L1 ball

        output: delta s.th. ||y2 + delta||_1 <= eps1
        and 0 <= x2 + y2 + delta <= 1
        '''
        x = x2.clone().float().view(x2.shape[0], -1)
        y = y2.clone().float().view(y2.shape[0], -1)
        sigma = y.clone().sign()
        u = torch.min(1 - x - y, x + y)
        #u = torch.min(u, epsinf - torch.clone(y).abs())
        u = torch.min(torch.zeros_like(y), u)
        l = -torch.clone(y).abs()
        d = u.clone()
        bs, indbs = torch.sort(-torch.cat((u, l), 1), dim=1)
        bs2 = torch.cat((bs[:, 1:], torch.zeros(bs.shape[0], 1).to(bs.device)), 1)
        inu = 2*(indbs < u.shape[1]).float() - 1
        size1 = inu.cumsum(dim=1)
        s1 = -u.sum(dim=1)
        c = eps1 - y.clone().abs().sum(dim=1)
        c5 = s1 + c < 0
        c2 = c5.nonzero().squeeze(1)
        s = s1.unsqueeze(-1) + torch.cumsum((bs2 - bs) * size1, dim=1)
        if c2.nelement != 0:
            lb = torch.zeros_like(c2).float()
            ub = torch.ones_like(lb) *(bs.shape[1] - 1)
        #print(c2.shape, lb.shape)
        nitermax = torch.ceil(torch.log2(torch.tensor(bs.shape[1]).float()))
        counter2 = torch.zeros_like(lb).long()
        counter = 0
        while counter < nitermax:
            counter4 = torch.floor((lb + ub) / 2.)
            counter2 = counter4.type(torch.LongTensor)
            c8 = s[c2, counter2] + c[c2] < 0
            ind3 = c8.nonzero().squeeze(1)
            ind32 = (~c8).nonzero().squeeze(1)
            #print(ind3.shape)
            if ind3.nelement != 0:
                lb[ind3] = counter4[ind3]
            if ind32.nelement != 0:
                ub[ind32] = counter4[ind32]
            #print(lb, ub)
            counter += 1
        lb2 = lb.long()
        alpha = (-s[c2, lb2] -c[c2]) / size1[c2, lb2 + 1] + bs2[c2, lb2]
        d[c2] = -torch.min(torch.max(-u[c2], alpha.unsqueeze(-1)), -l[c2])
        return (sigma * d).view(x2.shape)

    def L0_norm(self, x):
        return (x != 0.).view(x.shape[0], -1).sum(-1)
    def lp_norm(self, x):
        t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
        return t.view(-1, *([1] * self.dims))
    def normalize(self, x, norm):
        if norm == 'Linf':
            t = x.abs().view(x.shape[0], -1).max(1)[0]
            return x / (t.view(-1, *([1] * self.dims)) + 1e-12)

        elif norm == 'L2':
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
            return x / (t.view(-1, *([1] * self.dims)) + 1e-12)

        elif norm == 'L1':
            try:
                t = x.abs().view(x.shape[0], -1).sum(dim=-1)
            except:
                t = x.abs().reshape([x.shape[0], -1]).sum(dim=-1)
            return x / (t.view(-1, *([1] * self.dims)) + 1e-12)

    def check_oscillation(self, x, j, k, y5, k3=0.75):
        t = torch.zeros(x.shape[1]).to(self.device)
        for counter5 in range(k):
          t += (x[j - counter5] > x[j - counter5 - 1]).float()

        return (t <= k * k3 * torch.ones_like(t)).float()
    
    def dlr_loss(self, x, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()
        u = torch.arange(x.shape[0])

        return -(x[u, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (
            1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)
    
    def apgd_attack(self,x,y, norm, loss):
        loss_steps = torch.zeros([self.iter_step, x.shape[0]]).to(self.device)

        # x_adv = x.detach() + 0.001 * torch.randn(x.shape).cuda().detach()
        if norm == 'Linf':
            t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
            x_adv = x + self.epsilon * torch.ones_like(x
                ).detach() * self.normalize(t, norm)
        elif norm == 'L2':
            t = torch.randn(x.shape).to(self.device).detach()
            x_adv = x + self.epsilon * torch.ones_like(x
                ).detach() * self.normalize(t, norm)
        elif norm == 'L1':
            t = torch.randn(x.shape).to(self.device).detach()
            delta = self.L1_projection(x, t, self.epsilon)
            x_adv = x + t + delta
        
        
        x_adv = x_adv.clamp(0.0, 1.0)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        
        # loss ce
        if loss == 'ce':
            criterion = nn.CrossEntropyLoss(reduction='none')
        elif loss == 'dlr':
            criterion = self.dlr_loss
        x_adv.requires_grad_()

        # eot_iter numbers of iteration
        with torch.enable_grad():
            logits = self.model(x_adv)
            loss = criterion(logits, y)
            loss_sum = loss.sum()
        grad = torch.autograd.grad(loss_sum, [x_adv])[0].detach()
        acc = logits.detach().max(1)[1] == y
        grad_best = grad.clone()
        loss_best = loss.detach().clone()
        alpha = 2. if norm in ['Linf', 'L2'] else 1. if norm in ['L1'] else 2e-2
        step_size = alpha * self.epsilon * torch.ones([x.shape[0], *([1]*self.dims)]).to(self.device).detach()  # ???
        x_adv_old = x_adv.clone()
        count = 0
        iter_s = self.iter_step2 
        if norm == 'L1':
            iter_s = max(int(.04 * self.iter_step), 1)
            n_fts = reduce(operator.mul, list(x.shape[1:]), 1)
            topk = .2 * torch.ones([x.shape[0]], device=self.device)
            sp_old =  n_fts * torch.ones_like(topk)
            adasp_redstep = 1.5
            adasp_minstep = 10.

        loss_best_save = loss_best.detach().clone()
        reduced_save = torch.ones_like(loss_best)           # ???
        n_fts = x.shape[-3] * x.shape[-2] * x.shape[-1]        
        u = torch.arange(x.shape[0], device=self.device)
        for i in range(self.iter_step):
            with torch.no_grad():
                x_adv = x_adv.detach()
                x_diff = x_adv - x_adv_old
                x_adv_old = x_adv.clone()

                a = 0.75 if i > 0 else 1.0
                if norm == 'Linf':
                    x_adv_ = x_adv + step_size * torch.sign(grad)
                    x_adv_ = torch.min(torch.max(x_adv_, x - self.epsilon), x + self.epsilon)
                    x_adv_ = torch.clamp(x_adv_, 0.0, 1.0)
                    x_adv_ = torch.min(torch.max(x_adv + (x_adv_ - x_adv) * a + x_diff * (1 - a), x - self.epsilon), x + self.epsilon)
                    x_adv = torch.clamp(x_adv_, 0.0, 1.0)
                elif norm == 'L2':
                    x_adv_ = x_adv + step_size * self.normalize(grad, norm)
                    x_adv_ = torch.clamp(x + self.normalize(x_adv_ - x, norm) * torch.min(self.epsilon * torch.ones_like(x).detach(),
                        self.lp_norm(x_adv_ - x)), 0.0, 1.0)
                    x_adv_ = x_adv + (x_adv_ - x_adv) * a + x_diff * (1 - a)
                    x_adv_ = torch.clamp(x + self.normalize(x_adv_ - x, norm) * torch.min(self.epsilon * torch.ones_like(x).detach(),
                        self.lp_norm(x_adv_ - x)), 0.0, 1.0)
                elif norm == 'L1':
                    grad_topk = grad.abs().view(x.shape[0], -1).sort(-1)[0]
                    topk_curr = torch.clamp((1. - topk) * n_fts, min=0, max=n_fts - 1).long()
                    grad_topk = grad_topk[u, topk_curr].view(-1, *[1]*(len(x.shape) - 1))
                    sparsegrad = grad * (grad.abs() >= grad_topk).float()
                    x_adv_ = x_adv + step_size * sparsegrad.sign() / (
                        sparsegrad.sign().abs().view(x.shape[0], -1).sum(dim=-1).view(
                        -1, *[1]*(len(x.shape) - 1)) + 1e-10)
                    
                    delta_u = x_adv_ - x
                    delta_p = self.L1_projection(x, delta_u, self.epsilon)
                    x_adv_ = x + delta_u + delta_p
            x_adv.requires_grad_()
            # eot_iter numbers of iteration
            with torch.enable_grad():
                logits = self.model(x_adv)
                loss = criterion(logits, y)
                loss_sum = loss.sum()
            grad = torch.autograd.grad(loss_sum, [x_adv])[0].detach()

            pred = logits.detach().max(1)[1] == y
            acc = torch.min(acc, pred)
            idx = (pred == 0).nonzero().squeeze()
            x_best_adv[idx] = x_adv[idx]


            ### check step size
            with torch.no_grad():
                loss_ = loss.detach().clone()
                loss_steps[i] = loss_
                idx = (loss_ > loss_best).nonzero().squeeze()
                x_best[idx] = x_adv[idx].clone()
                grad_best[idx] = grad[idx].clone()
                loss_best[idx] = loss_[idx]
                count += 1

                if count == iter_s:
                    if norm in ['Linf', 'L2']:
                        oscillation = self.check_oscillation(loss_steps, i, self.iter_step2, loss_best, k3=0.75)
                        reduce_no_impr = (1. - reduced_save) * (loss_best_save >= loss_best).float()
                        oscillation = torch.max(oscillation, reduce_no_impr)
                        reduced_save = oscillation.clone()
                        loss_best_save = loss_best.clone()

                        if oscillation.sum() > 0:
                            osc_idx = (oscillation > 0).nonzero().squeeze()
                            step_size[osc_idx] /= 2.0
                            x_adv[osc_idx] = x_best[osc_idx].clone()
                            grad[osc_idx] = grad_best[osc_idx].clone()

                        iter_s = max(iter_s - self.iter_decr, self.iter_min)
                    
                    elif norm == 'L1':
                        sp_curr = self.L0_norm(x_best - x)
                        fl_redtopk = (sp_curr / sp_old) < .95
                        topk = sp_curr / n_fts / 1.5
                        step_size[fl_redtopk] = alpha * self.epsilon
                        step_size[~fl_redtopk] /= adasp_redstep
                        step_size.clamp_(alpha * self.epsilon / adasp_minstep, alpha * self.epsilon)
                        sp_old = sp_curr.clone()
                    
                        x_adv[fl_redtopk] = x_best[fl_redtopk].clone()
                        grad[fl_redtopk] = grad_best[fl_redtopk].clone()
                    count = 0
        return (x_best, acc, loss_best, x_best_adv)

    def eval(self, x, y, norm, loss):
        self.setup(x)
        x = x.detach().clone()
        pred = self.model(x).max(1)[1]
        y = y.detach().clone()
        x_adv = x.detach().clone()
        acc = pred == y

        for i in range(self.restarts):
            idx = acc.nonzero().squeeze()
            if len(idx.shape) == 0:
                idx = idx.unsqueeze(0)
            if idx.numel() != 0:
                x_ = x[idx].clone()
                y_ = y[idx].clone()
                res_curr = self.apgd_attack(x_, y_, norm, loss)
                best_curr, acc_curr, loss_curr, adv_curr = res_curr
                idx_curr = (acc_curr == 0).nonzero().squeeze()
                acc[idx[idx_curr]] = 0
                x_adv[idx[idx_curr]] = adv_curr[idx_curr].clone()
        return x_adv

