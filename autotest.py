from re import X
from imagenet_c import corrupt
import torch
import numpy as np
from apgd_attack import APGD_attack
from common_attack import Common_attack
class autotest():
    def __init__(self, model, mode, device, eps):
        self.model = model
        self.mode = mode
        self.device = device
        self.eps = eps
    def run_test(self, x, y, batch_size = 250):
        
        apgd = APGD_attack(self.model, self.device, 10, self.eps)
        c_a = Common_attack(2, [0,1,2,3,4])
        robust_flags = torch.zeros(x.shape[0], dtype=torch.bool, device=self.device)
        y_adv = torch.empty_like(y)
        batch_num = int(np.ceil(x.shape[0] / batch_size))
        for idx in range(batch_num):
            start_idx = batch_size * idx
            end_idx = min((idx + 1) * batch_size, x.shape[0])

            x_ = x[start_idx:end_idx, :].clone().to(self.device)
            y_ = y[start_idx:end_idx].clone().to(self.device)
            output = self.model(x_).max(dim=1)[1]
            y_adv[start_idx:end_idx] = output
            correct_batch = y_.eq(output)
            robust_flags[start_idx:end_idx] = correct_batch.detach()

        robust_accuracy = torch.sum(robust_flags).item() / x.shape[0] 
        print('initial accuracy: {:.2%}'.format(robust_accuracy))

        num_robust = torch.sum(robust_flags).item()
        if num_robust == 0:
            return
        batch_num = int(np.ceil(num_robust / batch_size))
        robust_lin_idcs = torch.nonzero(robust_flags, as_tuple=False)
        if num_robust > 1:
            robust_lin_idcs.squeeze_()
        
        for idx in range(batch_num):
            start_idx = batch_size * idx
            end_idx = min((idx + 1) * batch_size, num_robust)
            batch_datapoint_idcs = robust_lin_idcs[start_idx:end_idx]
            if len(batch_datapoint_idcs.shape) > 1:
                batch_datapoint_idcs.squeeze_(-1)
            x_ = x[batch_datapoint_idcs, :].clone().to(self.device)
            y_ = y[batch_datapoint_idcs].clone().to(self.device)
            if len(x_.shape) == 3:
                x_.unsqueeze_(dim=0)
            corrupt_image = c_a.eval(x_, 1)
            output = self.model(corrupt_image).max(dim=1)[1]
            false_batch = ~y_.eq(output).to(robust_flags.device)
            non_robust_lin_idcs = batch_datapoint_idcs[false_batch]
            robust_flags[non_robust_lin_idcs] = False
            num_non_robust_batch = torch.sum(false_batch)
            print('{} - {}/{} - {} out of {} successfully perturbed'.format('common', idx + 1, batch_num, num_non_robust_batch, x_.shape[0]))
        num_robust = torch.sum(robust_flags).item()
        if num_robust == 0:
            return
        batch_num = int(np.ceil(num_robust / batch_size))
        robust_lin_idcs = torch.nonzero(robust_flags, as_tuple=False)
        if num_robust > 1:
            robust_lin_idcs.squeeze_()

        for idx in range(batch_num):
            start_idx = batch_size * idx
            end_idx = min((idx + 1) * batch_size, num_robust)
            batch_datapoint_idcs = robust_lin_idcs[start_idx:end_idx]
            if len(batch_datapoint_idcs.shape) > 1:
                batch_datapoint_idcs.squeeze_(-1)
            x_ = x[batch_datapoint_idcs, :].clone().to(self.device)
            y_ = y[batch_datapoint_idcs].clone().to(self.device)
            if len(x_.shape) == 3:
                x_.unsqueeze_(dim=0)
            adv_curr = apgd.eval(x_, y_)
            output = self.model(adv_curr).max(dim=1)[1]
            false_batch = ~y_.eq(output).to(robust_flags.device)
            non_robust_lin_idcs = batch_datapoint_idcs[false_batch]
            robust_flags[non_robust_lin_idcs] = False
            num_non_robust_batch = torch.sum(false_batch)
            print('{} - {}/{} - {} out of {} successfully perturbed'.format('apgd', idx + 1, batch_num, num_non_robust_batch, x_.shape[0]))
        robust_accuracy = torch.sum(robust_flags).item() / x.shape[0]
        print(('robust accuracy after {}: {:.2%} '.format('apgd', robust_accuracy)))