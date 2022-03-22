import torch
import torch.nn as nn
import torch.nn.functional as F

class APGD_attack():
    def __init__(self, model, device, iter_step, epsilon=0.3):
        self.model = model
        self.device = device
        self.iter_step = iter_step
        self.epsilon = epsilon
    def setup(self, x):
        self.dims = len(list(x.shape[1:]))
        self.iter_step2 = max(int(0.22 * self.iter_step),1)

    def normalize(self, x):
        t = x.abs().view(x.shape[0], -1).max(1)[0]
        return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

    def apgd_attack(self,x,y):
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
                