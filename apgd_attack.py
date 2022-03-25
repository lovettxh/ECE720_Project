import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.iter_min = max(int(0.06 * self.n_iter), 1)


    def normalize(self, x):
        t = x.abs().view(x.shape[0], -1).max(1)[0]
        return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

    def check_oscillation(self, x, j, k, y5, k3=0.75):
        t = torch.zeros(x.shape[1]).to(self.device)
        for counter5 in range(k):
          t += (x[j - counter5] > x[j - counter5 - 1]).float()

    def apgd_attack(self,x,y):



        loss_steps = torch.zeros([self.iter_step, x.shape[0]]).to(self.device)
        # t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
        # x_adv = x + self.eps * torch.ones_like(x
        #         ).detach() * self.normalize(t)
        x_adv = x.detach() + 0.001 * torch.randn(x.shape).cuda().detach()
        x_adv = x_adv.clamp(0.0, 1.0)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        
        # loss ce
        criterion = nn.CrossEntropyLoss(reduction='none')
        x_adv.requires_grad_()

        # eot_iter numbers of iteration
        with torch.enable_grad():
            logits = self.model(x_adv)
            loss = criterion(logits, y)
        grad = torch.autograd.grad(loss, [x_adv])[0].detach()
        acc = logits.detach().max(1)[1] == y
        grad_best = grad.clone()
        loss_best = loss.detach().clone()
        alpha = 2
        step_size = alpha * self.epsilon * torch.ones([x.shape[0], *([1]*self.dims)]).to(self.device).detach()  # ???
        x_adv_old = x_adv.clone()
        count = 0
        iter_s = self.iter_step2 
        loss_best_save = loss_best.clone()
        reduced_save = torch.ones_like(loss_best)           # ???

        for i in range(self.iter_step):
            with torch.no_grad():
                x_adv = x_adv.detach()
                x_diff = x_adv - x_adv_old
                x_adv_old = x_adv.clone()

                a = 0.75 if i > 0 else 1.0

                x_adv_ = x_adv + step_size * torch.sign(grad)
                x_adv_ = torch.min(torch.max(x_adv_, x - self.eps), x + self.eps)
                x_adv_ = torch.clamp(x_adv_, 0.0, 1.0)
                x_adv_ = torch.min(torch.max(x_adv + (x_adv_ - x_adv) * a + x_diff * (1 - a), x - self.eps), x + self.eps)
                x_adv = torch.clamp(x_adv_, 0.0, 1.0)
            
            x_adv.requires_grad_()
            # eot_iter numbers of iteration
            with torch.enable_grad():
                logits = self.model(x_adv)
                loss = criterion(logits, y)
            grad = torch.autograd.grad(loss, [x_adv])[0].detach()

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

                if count == self.iter_s:
                    oscillation = self.check_oscillation(loss_steps, i, self.iter_s, loss_best, k3=self.thr_decr)
                    reduce_no_impr = (1. - reduced_save) * (loss_best_save >= loss_best).float()
                    oscillation = torch.max(oscillation, reduce_no_impr)
                    reduced_save = oscillation.clone()
                    loss_best_save = loss_best.clone()

                    if oscillation.sum() > 0:
                        osc_idx = (oscillation > 0).nonzero().squeeze()
                        step_size[osc_idx] /= 2
                        x_adv[osc_idx] = x_best[osc_idx].clone()
                        grad[osc_idx] = grad_best[osc_idx].clone()

                    iter_s = max(iter_s - self.iter_decr, self.iter_min)
                    count = 0

        return (x_best, acc, loss_best, x_best_adv)

    def eval(self, x, y):
        self.setup(x)
        x = x.detach().clone().float().to(self.device)
        pred = self.model(x).max(1)[1]
        y = y.detach().clone().float().to(self.device)
        x_adv = x.detach().clone()
        acc = pred == y

        torch.random.manual_seed(self.seed)
        torch.cuda.random.manual_seed(self.seed)

        for i in range(self.restarts):
            idx = acc.nonzero().squeeze()
            if len(idx.shape) == 0:
                idx = idx.unsqueeze(0)
            if idx.numel() != 0:
                x_ = x[idx].clone()
                y_ = y[idx].clone()
                res_curr = self.n(x_, y_)
                best_curr, acc_curr, loss_curr, adv_curr = res_curr
                idx_curr = (acc_curr == 0).nonzero().squeeze()
                acc[idx[idx_curr]] = 0
                x_adv[idx[idx_curr]] = adv_curr[idx_curr].clone()
        return x_adv

