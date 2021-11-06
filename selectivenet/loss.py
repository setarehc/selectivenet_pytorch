import torch

import numpy as np

class SelectiveLoss(torch.nn.Module):
    def __init__(self, loss_func, coverage:float, alpha:float=0.5, lm:float=32.0):
        """
        Args:
            loss_func: base loss function. the shape of loss_func(x, target) shoud be (B). 
                       e.g.) torch.nn.CrossEntropyLoss(reduction=none) : classification
            coverage: target coverage.
            lm: Lagrange multiplier for coverage constraint. original experiment's value is 32. 
        """
        super(SelectiveLoss, self).__init__()
        assert 0.0 < coverage <= 1.0
        assert 0.0 < lm
        assert 0.0 < alpha <= 1.0

        self.loss_func = loss_func
        self.coverage = coverage
        self.lm = lm
        self.alpha = alpha

    def forward(self, prediction_out, selection_out, auxiliary_out, target, threshold=0.5, mode='train'):
        """
        Args:
            prediction_out: (B,num_classes)
            selection_out:  (B, 1)
        """
        # compute emprical coverage (=phi^)
        emprical_coverage = selection_out.mean() 
        
        # compute emprical risk (=r^)
        emprical_risk = (self.loss_func(prediction_out, target)*selection_out.view(-1)).mean()
        emprical_risk = emprical_risk / emprical_coverage

        # compute penalty (=psi)
        coverage = torch.tensor([self.coverage], dtype=torch.float32, requires_grad=True, device='cuda')
        penalty = torch.max(coverage-emprical_coverage, torch.tensor([0.0], dtype=torch.float32, requires_grad=True, device='cuda'))**2
        penalty *= self.lm

        # compute selective loss (=L(f,g))
        selective_loss = emprical_risk + penalty
        
        # compute standard cross entropy loss
        ce_loss = torch.nn.CrossEntropyLoss()(auxiliary_out, target)
        
        # total loss
        loss_pytorch = self.alpha * selective_loss + (1.0 - self.alpha) * ce_loss
        
        # compute coverage based on source implementation
        selective_head_coverage = self.get_coverage(selection_out, threshold)

        # compute selective accuracy based on source implementation
        selective_head_selective_acc = self.get_selective_acc(prediction_out, selection_out, target)

        # compute accuracy based on source implementation
        classification_head_acc = self.get_accuracy(auxiliary_out, target)
        
        # compute selective loss (=selective_head_loss) based on source implementation
        selective_head_loss = self.get_selective_loss(prediction_out, selection_out, target)

        # compute cross entropy loss (=classification_head_loss) based on source implementation
        classification_head_loss = torch.nn.CrossEntropyLoss()(auxiliary_out, target)

        # compute loss
        loss = self.alpha * selective_head_loss + (1.0 - self.alpha) * classification_head_loss

        # empirical selective risk with rejection for test model
        if mode == 'test':
            test_selective_risk = self.get_selective_risk(prediction_out, selection_out, target, threshold) 

        # loss information dict 
        pref = ''
        if mode == 'validation':
            pref = 'val_'
        loss_dict={}
        loss_dict['{}emprical_coverage'.format(pref)] = emprical_coverage.detach().cpu().item()
        loss_dict['{}emprical_risk'.format(pref)] = emprical_risk.detach().cpu().item()
        loss_dict['{}penalty'.format(pref)] = penalty.detach().cpu().item()
        loss_dict['{}selective_loss'.format(pref)] = selective_loss.detach().cpu().item()
        loss_dict['{}ce_loss'.format(pref)] = ce_loss.detach().cpu().item()
        loss_dict['{}loss_pytorch'.format(pref)] = loss_pytorch
        loss_dict['{}selective_head_coverage'.format(pref)] = selective_head_coverage.detach().cpu().item() #coverage
        loss_dict['{}selective_head_selective_acc'.format(pref)] = selective_head_selective_acc.detach().cpu().item() #selective_accurcy
        loss_dict['{}classification_head_acc'.format(pref)] = classification_head_acc.detach().cpu().item() #calassification_accuracy
        loss_dict['{}selective_head_loss'.format(pref)] = selective_head_loss.detach().cpu().item() #selective_loss
        loss_dict['{}classification_head_loss'.format(pref)] = classification_head_loss.detach().cpu().item() #ce_loss
        loss_dict['{}loss'.format(pref)] = loss
        if mode == 'test':
            loss_dict['test_selective_risk'] = test_selective_risk.detach().cpu().item()

        return loss_dict

    # based on source implementation
    def get_selective_acc(self, prediction_out, selection_out, target):
        """
        Equivalent to selective_acc function of source implementation
        Args:
            prediction_out: (B,num_classes)
            selection_out:  (B, 1)
        """
        g = (selection_out.squeeze(-1) > 0.5).float()
        num = torch.dot(g, (torch.argmax(prediction_out, dim=-1) == target).float())
        return num / torch.sum(g)

    # based on source implementation
    def get_coverage(self, selection_out, threshold):
        """
        Equivalent to coverage function of source implementation
        Args:
            selection_out:  (B, 1)
        """
        g = (selection_out.squeeze(-1) >= threshold).float()
        return torch.mean(g)

    # based on source implementation
    def get_accuracy(self, auxiliary_out, target): #TODO: Check implementation with Lili
        """
        Equivalent to "accuracy" in Tensorflow
        Args:
            selection_out:  (B, 1)
        """ 
        num = torch.sum((torch.argmax(auxiliary_out, dim=-1) == target).float())
        return num / len(auxiliary_out)
    
    # based on source implementation
    def get_selective_loss(self, prediction_out, selection_out, target):
        """
        Equivalent to selective_loss function of source implementation
        Args:
            prediction_out: (B,num_classes)
            selection_out:  (B, 1)
        """
        ce = self.loss_func(prediction_out, target)
        empirical_risk_variant = torch.mean(ce * selection_out.view(-1))
        empirical_coverage = selection_out.mean() 
        penalty = torch.max(self.coverage - empirical_coverage, torch.tensor([0.0], dtype=torch.float32, requires_grad=True, device='cuda'))**2
        loss = empirical_risk_variant + self.lm * penalty
        return loss

    # selective risk in test mode
    def get_selective_risk(self, prediction_out, selection_out, target, threshold):
        g = (selection_out.squeeze(-1) >= threshold).float()
        empirical_coverage_rjc = torch.mean(g)
        empirical_risk_rjc = torch.mean(self.loss_func(prediction_out, target) * g.view(-1))
        empirical_risk_rjc /= empirical_coverage_rjc
        return empirical_risk_rjc