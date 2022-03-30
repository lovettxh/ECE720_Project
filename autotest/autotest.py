from re import X
import torch
import numpy as np
from .common_attack import Common_attack
from .apgd_attack import APGD_attack
from .fab_pt import FAB_Attack_PT
from .attack import eval_adv_test_whitebox
class autotest():
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.steps = 50
    def run_normal_mode(self, x, y, batch_size, mode1, mode2, epsilon = 0.003):
        apgd = APGD_attack(self.model, self.device, self.steps, epsilon)
        fab = FAB_Attack_PT(self.model, n_restarts=5, n_iter=self.steps, eps=epsilon, device=self.device)
        c_a = Common_attack(1)
        robust_flags = torch.zeros(x.shape[0], dtype=torch.bool, device=self.device)
        y_adv = torch.empty_like(y)
        batch_num = int(np.ceil(x.shape[0] / batch_size))
        # narutal acc
        for idx in range(batch_num):
            start_idx = batch_size * idx
            end_idx = min((idx + 1) * batch_size, x.shape[0])
            x_ = x[start_idx:end_idx, :].clone().to(self.device)
            y_ = y[start_idx:end_idx].clone().to(self.device)
            output = self.model(x_).max(dim=1)[1]
            y_adv[start_idx:end_idx] = output
            correct_batch = y_.eq(output)
            robust_flags[start_idx:end_idx] = correct_batch.detach()
        init_accuracy = torch.sum(robust_flags).item() / x.shape[0] 
        print('initial accuracy: {:.2%}'.format(init_accuracy))
        num_robust = torch.sum(robust_flags).item()
        if num_robust == 0:
            return
        batch_num = int(np.ceil(num_robust / batch_size))
        robust_lin_idcs = torch.nonzero(robust_flags, as_tuple=False)
        if num_robust > 1:
            robust_lin_idcs.squeeze_()
        batch_num_save = batch_num
        robust_lin_idcs_save = robust_lin_idcs.detach().clone()
        robust_flags_save = robust_flags.detach().clone()
        num_robust_save = num_robust
        # common mode 1&2
        for m in [mode1, mode2]:
            batch_num = batch_num_save
            num_robust = num_robust_save
            robust_lin_idcs = robust_lin_idcs_save.detach().clone()
            robust_flags = robust_flags_save.detach().clone()
            for i in m:
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
                    corrupt_image = torch.tensor(c_a.eval(x_.cpu().detach().numpy(), i)).type(torch.FloatTensor).to(self.device)
                    output = self.model(corrupt_image).max(dim=1)[1]
                    false_batch = ~y_.eq(output).to(robust_flags.device)
                    non_robust_lin_idcs = batch_datapoint_idcs[false_batch]
                    robust_flags[non_robust_lin_idcs] = False
                    num_non_robust_batch = torch.sum(false_batch)
                    print('{} - {}/{} - {} out of {} successfully perturbed'.format('common-'+str(i), idx + 1, batch_num, num_non_robust_batch, x_.shape[0]))
                # update num
                num_robust = torch.sum(robust_flags).item()
                robust_accuracy = torch.sum(robust_flags).item() / x.shape[0]
                print(('robust accuracy after {}: {:.2%} '.format('common-'+str(i), robust_accuracy)))
                if num_robust == 0:
                    return
                batch_num = int(np.ceil(num_robust / batch_size))
                robust_lin_idcs = torch.nonzero(robust_flags, as_tuple=False)
                if num_robust > 1:
                    robust_lin_idcs.squeeze_()
                # --
            robust_accuracy = torch.sum(robust_flags).item() / x.shape[0]
            print(('robust accuracy after {}: {:.2%} '.format('common-mode-1', robust_accuracy)))

            # apgd
            norm = 'Linf'
            loss = 'ce'
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
                adv_curr = apgd.eval(x_, y_, norm, loss)
                output = self.model(adv_curr).max(dim=1)[1]
                false_batch = ~y_.eq(output).to(robust_flags.device)
                non_robust_lin_idcs = batch_datapoint_idcs[false_batch]
                robust_flags[non_robust_lin_idcs] = False
                num_non_robust_batch = torch.sum(false_batch)
                print('{} - {}/{} - {} out of {} successfully perturbed'.format('apgd-'+loss+'-'+norm, idx + 1, batch_num, num_non_robust_batch, x_.shape[0]))
            if m == mode1:
                robust_accuracy_mode1 = torch.sum(robust_flags).item() / x.shape[0]
            else:
                robust_accuracy_mode2 = torch.sum(robust_flags).item() / x.shape[0]
        print(('robust accuracy for {}: {:.2%} '.format('natural-mode1', robust_accuracy_mode1)))
        print(('robust score for {}: {:.2%} '.format('natural-mode1', robust_accuracy_mode1/init_accuracy)))
        print(('robust accuracy for {}: {:.2%} '.format('natural-mode2', robust_accuracy_mode2)))
        print(('robust score for {}: {:.2%} '.format('natural-mode2', robust_accuracy_mode2/init_accuracy)))


    def run_adv_mode(self, x, y, batch_size, adv_num, epsilon = 0.03):
        apgd = APGD_attack(self.model, self.device, self.steps, epsilon)
        fab = FAB_Attack_PT(self.model, n_restarts=5, n_iter=self.steps, eps=epsilon, device=self.device)
        c_a = Common_attack(1)
        robust_flags = torch.zeros(x.shape[0], dtype=torch.bool, device=self.device)
        y_adv = torch.empty_like(y)
        batch_num = int(np.ceil(x.shape[0] / batch_size))
        
        # natural acc
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
            return 0 
        batch_num = int(np.ceil(num_robust / batch_size))
        robust_lin_idcs = torch.nonzero(robust_flags, as_tuple=False)
        if num_robust > 1:
            robust_lin_idcs.squeeze_()
        
        #apgd
        norm = 'Linf' 
        for loss in ['ce', 'dlr']:
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
                adv_curr = apgd.eval(x_, y_, norm, loss)
                output = self.model(adv_curr).max(dim=1)[1]
                false_batch = ~y_.eq(output).to(robust_flags.device)
                non_robust_lin_idcs = batch_datapoint_idcs[false_batch]
                robust_flags[non_robust_lin_idcs] = False
                num_non_robust_batch = torch.sum(false_batch)
                print('{} - {}/{} - {} out of {} successfully perturbed'.format('apgd-'+loss+'-'+norm, idx + 1, batch_num, num_non_robust_batch, x_.shape[0]))
            num_robust = torch.sum(robust_flags).item()
            robust_accuracy = torch.sum(robust_flags).item() / x.shape[0]
            print(('robust accuracy after {}: {:.2%} '.format('apgd-'+loss+'-'+norm, robust_accuracy)))
            if num_robust == 0:
                return 0 
            batch_num = int(np.ceil(num_robust / batch_size))
            robust_lin_idcs = torch.nonzero(robust_flags, as_tuple=False)
            if num_robust > 1:
                robust_lin_idcs.squeeze_()
        # fab
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
            adv_curr = fab.perturb(x_, y_)
            output = self.model(adv_curr).max(dim=1)[1]
            false_batch = ~y_.eq(output).to(robust_flags.device)
            non_robust_lin_idcs = batch_datapoint_idcs[false_batch]
            robust_flags[non_robust_lin_idcs] = False
            num_non_robust_batch = torch.sum(false_batch)
            print('{} - {}/{} - {} out of {} successfully perturbed'.format('fab-Linf', idx + 1, batch_num, num_non_robust_batch, x_.shape[0]))
        robust_accuracy_save = torch.sum(robust_flags).item() / x.shape[0]
        print(('robust accuracy after {}: {:.2%} '.format('fab', robust_accuracy_save)))

        # update num
        num_robust = torch.sum(robust_flags).item()
        if num_robust == 0:
            return 0
        batch_num = int(np.ceil(num_robust / batch_size))
        robust_lin_idcs = torch.nonzero(robust_flags, as_tuple=False)
        if num_robust > 1:
            robust_lin_idcs.squeeze_()
        
        for n in adv_num:
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
              corrupt_image = torch.tensor(c_a.eval(x_.cpu().detach().numpy(), n)).type(torch.FloatTensor).to(self.device)
              output = self.model(corrupt_image).max(dim=1)[1]
              false_batch = ~y_.eq(output).to(robust_flags.device)
              non_robust_lin_idcs = batch_datapoint_idcs[false_batch]
              robust_flags[non_robust_lin_idcs] = False
              num_non_robust_batch = torch.sum(false_batch)
              print('{} - {}/{} - {} out of {} successfully perturbed'.format('common-'+str(n), idx + 1, batch_num, num_non_robust_batch, x_.shape[0]))
          num_robust = torch.sum(robust_flags).item()
          if num_robust == 0:
              return 0
          batch_num = int(np.ceil(num_robust / batch_size))
          robust_lin_idcs = torch.nonzero(robust_flags, as_tuple=False)
          if num_robust > 1:
              robust_lin_idcs.squeeze_()
        robust_accuracy = torch.sum(robust_flags).item() / x.shape[0]
        print('robust accuracy : {:.2%} '.format(robust_accuracy))
        return robust_accuracy

    def run_test(self, testloader, x, y, mode, epsilon = 0.003 , batch_size = 200):
        com_num1 = [0,4,8,14,16]
        com_num2 = [1,5,9,10,15]
        adv_num = [0,1,2,3,4]
        #com_num = com_num[::-1]
        if mode == 'natural':
            self.run_normal_mode(x, y, batch_size, com_num1,com_num2, epsilon)
        elif mode == 'adv':
            adv_acc = eval_adv_test_whitebox(self.model, self.device, testloader, epsilon, self.steps, 0.003)
            origin_acc = (x.shape[0] - adv_acc) / x.shape[0]
            robust_acc = self.run_adv_mode(x, y, batch_size, adv_num, epsilon)
            print('robust score: {:.2%}'.format(robust_acc/origin_acc))