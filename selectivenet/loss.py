import torch
from torch.nn.modules.loss import _Loss

import numpy as np

class NLLLoss(torch.nn.Module):
    """
    Negative log likelihood loss
    """
    def __init__(self, distribution_type, reduction='mean'):
        super(NLLLoss, self).__init__()
        self.distribution_type = distribution_type
        self.reduction = reduction

    def gaussian_nll(self, mu, sigma, labels):
        likelihood = torch.distributions.Normal(mu, sigma)
        nll = - likelihood.log_prob(labels.unsqueeze(-1))
        if self.reduction == 'none':
            return nll.sum(dim=1)
        elif self.reduction == 'mean':
            return nll.sum(dim=1).mean()
        else:
            raise Exception("Reductio type incorrect!")
    
    def laplace_nll(self, mu, b, labels):
        likelihood = torch.distributions.Laplace(mu, b)
        nll = - likelihood.log_prob(labels.unsqueeze(-1))
        if self.reduction == 'none':
            return nll.sum(dim=1)
        elif self.reduction == 'mean':
            return nll.sum(dim=1).mean()
        else:
            raise Exception("Reductio type incorrect!")

    def forward(self, prediction, target):
        mu, uncertainty = prediction
        if self.distribution_type == 'Gaussian':
            nll_loss = self.gaussian_nll(mu, uncertainty, target)
        elif self.distribution_type == 'Laplace':
            nll_loss = self.laplace_nll(mu, uncertainty, target)
        else:
            raise Exception("Distribution type incorrect!")
        return nll_loss

class SelectiveLoss(torch.nn.Module):
    def __init__(self, loss_func, coverage:float, alpha:float=0.5, lm:float=32.0, regression=False, prob_mode=False):
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
        self.regression = regression
        self.prob_mode = prob_mode

    def forward(self, prediction_out, selection_out, auxiliary_out, target, threshold=0.5, mode='train'):
        """
        Args:
            prediction_out: (B,num_classes)
            selection_out:  (B, 1)
        """
        #import pdb; pdb.set_trace()
        
        if self.regression and not self.prob_mode:
                prediction_out = prediction_out.view(-1)
        
        # compute emprical coverage (=phi^)
        emprical_coverage = selection_out.mean() 
        
        # compute emprical risk (=r^)
        emprical_risk = (self.loss_func(prediction_out, target)*selection_out.view(-1)).mean()
        emprical_risk = emprical_risk / emprical_coverage

        # compute penulty (=psi)
        coverage = torch.tensor([self.coverage], dtype=torch.float32, requires_grad=True, device='cuda')
        penulty = torch.max(coverage-emprical_coverage, torch.tensor([0.0], dtype=torch.float32, requires_grad=True, device='cuda'))**2
        penulty *= self.lm

        # compute selective loss (=L(f,g))
        selective_loss = emprical_risk + penulty
        
        # compute tf accuracy
        classification_head_acc = self.get_accuracy(auxiliary_out, target, self.prob_mode)

        if self.regression and not self.prob_mode:
                auxiliary_out = auxiliary_out.view(-1)

        # compute standard cross entropy loss
        ce_loss = self.loss_func(auxiliary_out, target).mean()
        
        # total loss
        loss_pytorch = self.alpha * selective_loss + (1.0 - self.alpha) * ce_loss
        
        # compute tf coverage
        selective_head_coverage = self.get_coverage(selection_out, threshold)

        # compute tf selective accuracy 
        selective_head_selective_acc = self.get_selective_acc(prediction_out, selection_out, target, self.prob_mode)

        
        # compute tf selective loss (=selective_head_loss)
        selective_head_loss = self.get_selective_loss(prediction_out, selection_out, target)

        # compute tf cross entropy loss (=classification_head_loss)
        classification_head_loss = self.loss_func(auxiliary_out, target).mean()

        # compute loss
        loss = self.alpha * selective_head_loss + (1.0 - self.alpha) * classification_head_loss

        # empirical selective risk with rejection for test model
        if mode != 'train':
            test_selective_risk = self.get_selective_risk(prediction_out, selection_out, target, threshold)
            test_selective_loss = self.get_filtered_loss(prediction_out, selection_out, target, threshold)
            if self.regression and self.prob_mode:
                test_diff = self.get_diff(prediction_out[0], selection_out, target, threshold)
                test_risk  = self.get_risk(prediction_out[0], selection_out, target, threshold)
            elif self.regression:
                test_diff = self.get_diff(prediction_out, selection_out, target, threshold)

        # loss information dict 
        pref = ''
        if mode == 'validation':
            pref = 'val_'
        loss_dict={}
        loss_dict['{}emprical_coverage'.format(pref)] = emprical_coverage.detach().cpu().item()
        loss_dict['{}emprical_risk'.format(pref)] = emprical_risk.detach().cpu().item()
        loss_dict['{}penulty'.format(pref)] = penulty.detach().cpu().item()
        loss_dict['{}selective_loss'.format(pref)] = selective_loss.detach().cpu().item()
        loss_dict['{}ce_loss'.format(pref)] = ce_loss.detach().cpu().item()
        loss_dict['{}loss_pytorch'.format(pref)] = loss_pytorch
        loss_dict['{}selective_head_coverage'.format(pref)] = selective_head_coverage.detach().cpu().item() #coverage
        loss_dict['{}selective_head_selective_acc'.format(pref)] = selective_head_selective_acc.detach().cpu().item() #selective_accurcy
        loss_dict['{}classification_head_acc'.format(pref)] = classification_head_acc.detach().cpu().item() #calassification_accuracy
        loss_dict['{}selective_head_loss'.format(pref)] = selective_head_loss.detach().cpu().item() #selective_loss
        loss_dict['{}classification_head_loss'.format(pref)] = classification_head_loss.detach().cpu().item() #ce_loss
        loss_dict['{}loss'.format(pref)] = loss
        if mode != 'train':
            loss_dict['{}test_selective_risk'.format(pref)] = test_selective_risk.detach().cpu().item()
            loss_dict['{}test_selective_loss'.format(pref)] = test_selective_loss.detach().cpu().item()
            if self.regression:
                loss_dict['{}test_diff'.format(pref)] = test_diff.detach().cpu().item()
                if self.prob_mode:
                    loss_dict['{}test_risk'.format(pref)] = test_risk.detach().cpu().item()

        return loss_dict

    def get_selective_acc(self, prediction_out, selection_out, target, prob_mode):
        """
        Equivalent to selective_acc function of source implementation
        Args:
            prediction_out: (B,num_classes)
            selection_out:  (B, 1)
        """
        if prob_mode:
            return torch.tensor([0.0], dtype=torch.float32, requires_grad=True, device='cuda')
        g = (selection_out.squeeze(-1) > 0.5).float()
        num = torch.dot(g, (torch.argmax(prediction_out, dim=-1) == target).float())
        return num / torch.sum(g)

    # Tensorflow
    def get_coverage(self, selection_out, threshold):
        """
        Equivalent to coverage function of source implementation
        Args:
            selection_out:  (B, 1)
        """
        g = (selection_out.squeeze(-1) >= threshold).float()
        return torch.mean(g)

    # Tensorflow
    def get_accuracy(self, auxiliary_out, target, prob_mode): #TODO: Check implementation with Lili
        if prob_mode:
            return torch.tensor([0.0], dtype=torch.float32, requires_grad=True, device='cuda')
        """
        Equivalent to "accuracy" in Tensorflow
        Args:
            selection_out:  (B, 1)
        """ 
        num = torch.sum((torch.argmax(auxiliary_out, dim=-1) == target).float())
        return num / len(auxiliary_out)
    
    # Tensorflow
    def get_selective_loss(self, prediction_out, selection_out, target):
        """
        Equivalent to selective_loss function of source implementation
        Args:
            prediction_out: (B,num_classes)
            selection_out:  (B, 1)
        """
        ce = self.loss_func(prediction_out, target)
        empirical_risk_variant = (ce * selection_out.view(-1)).mean()
        empirical_coverage = selection_out.mean() 
        penalty = torch.max(self.coverage - empirical_coverage, torch.tensor([0.0], dtype=torch.float32, requires_grad=True, device='cuda'))**2
        loss = empirical_risk_variant + self.lm * penalty
        return loss

    # selective risk in test/validation mode
    def get_selective_risk(self, prediction_out, selection_out, target, threshold):
        g = (selection_out.squeeze(-1) >= threshold).float()
        empirical_coverage_rjc = torch.mean(g)
        empirical_risk_rjc = (self.loss_func(prediction_out, target) * g.view(-1)).mean()
        empirical_risk_rjc /= empirical_coverage_rjc
        return empirical_risk_rjc
    
    # prediction loss in test/validation mode
    def get_filtered_loss(self, prediction_out, selection_out, target, threshold):
        g = (selection_out.squeeze(-1) >= threshold).float()
        loss_rjc = (self.loss_func(prediction_out, target) * g.view(-1)).mean()
        return loss_rjc
    
    # difference between prediction and ground-truth for test mode
    def get_diff(self, prediction_out, selection_out, target, threshold):
        g = (selection_out.squeeze(-1) >= threshold).float()
        diff = (((prediction_out .view(-1) - target)**2) * g.view(-1)).mean()
        #diff = (torch.abs(prediction_out.view(-1) - target) * g.view(-1)).mean()
        return diff
    
    def get_risk(self, prediction_out, selection_out, target, threshold):
        g = (selection_out.squeeze(-1) >= threshold).float()
        cov = torch.mean(g)
        diff = (((prediction_out.view(-1) - target)**2) * g.view(-1)).mean()
        #diff = (torch.abs(prediction_out.view(-1) - target) * g.view(-1)).mean()
        diff /= cov
        return diff