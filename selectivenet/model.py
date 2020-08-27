import torch

class ProbabilisticSelectiveNet(torch.nn.Module):
    """
    SelectiveNet for regression with probabilistic uncertainty measure.
    """
    def __init__(self, input_dim:int, dim_features:int, init_weights=True, div_by_ten=False):
        """
        Args
            input_dim: dimension of input vector.
            dim_featues: dimension of features from body block.
        """
        super(ProbabilisticSelectiveNet, self).__init__()
        self.input_dim = input_dim
        self.dim_features = dim_features
        self.div_by_ten = div_by_ten

        # main body block
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.dim_features),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm1d(self.dim_features),
        )

        # represented as f() in the original paper
        self.mean_predictor = torch.nn.Sequential(
            torch.nn.Linear(self.dim_features, 1)
        )
        
        self.std_predictor = torch.nn.Sequential(
            torch.nn.Linear(self.dim_features, 1),
            torch.nn.Softplus()
        )

        # represented as g() in the original paper
        self.pre_selector = torch.nn.Sequential(
            torch.nn.Linear(self.dim_features, 16),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm1d(16) 
        )

        self.post_selector = torch.nn.Sequential(
            torch.nn.Linear(16, 1),
            torch.nn.Sigmoid()
        )

        # represented as h() in the original paper
        self.aux_mean_predictor = torch.nn.Sequential(
            torch.nn.Linear(self.dim_features, 1)
        )

        self.aux_std_predictor = torch.nn.Sequential(
            torch.nn.Linear(self.dim_features, 1),
            torch.nn.Softplus()
        )

        # initialize weights of heads
        if init_weights:
            self._initialize_weights(self.feature_extractor)
            self._initialize_weights(self.mean_predictor)
            self._initialize_weights(self.std_predictor)
            self._initialize_weights(self.pre_selector)
            self._initialize_weights(self.post_selector)
            self._initialize_weights(self.aux_mean_predictor)
            self._initialize_weights(self.aux_std_predictor)
        

    def forward(self, x):
        x = self.feature_extractor(x.float())
        x = x.view(x.size(0), -1)
        
        mean = self.mean_predictor(x)
        std = self.std_predictor(x)
        #std = torch.ones_like(std)

        selection_out= self.pre_selector(x)
        if self.div_by_ten:
            selection_out /= 10.0
        selection_out = self.post_selector(selection_out)
    
        aux_mean = self.aux_mean_predictor(x)
        aux_std = self.aux_std_predictor(x)
        #aux_std = torch.ones_like(aux_std)

        return (mean, std), selection_out, (aux_mean, aux_std)

    def _initialize_weights(self, module):
        for m in module.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm1d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)


class SelectiveNetRegression(torch.nn.Module):
    """
    SelectiveNet for regression with rejection option.
    """
    def __init__(self, input_dim:int, dim_features:int, init_weights=True, div_by_ten=False):
        """
        Args
            input_dim: dimension of input vector.
            dim_featues: dimension of features from body block.
        """
        super(SelectiveNetRegression, self).__init__()
        self.input_dim = input_dim
        self.dim_features = dim_features
        self.div_by_ten = div_by_ten

        # main body block
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.dim_features),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm1d(self.dim_features),
        )

        # represented as f() in the original paper
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(self.dim_features, 1)
        )

        # represented as g() in the original paper
        self.pre_selector = torch.nn.Sequential(
            torch.nn.Linear(self.dim_features, 16),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm1d(16) 
        )

        self.post_selector = torch.nn.Sequential(
            torch.nn.Linear(16, 1),
            torch.nn.Sigmoid()
        )

        # represented as h() in the original paper
        self.aux_predictor = torch.nn.Sequential(
            torch.nn.Linear(self.dim_features, 1)
        )

        # initialize weights of heads
        if init_weights:
            self._initialize_weights(self.feature_extractor)
            self._initialize_weights(self.predictor)
            self._initialize_weights(self.pre_selector)
            self._initialize_weights(self.post_selector)
            self._initialize_weights(self.aux_predictor)
        

    def forward(self, x):
        x = self.feature_extractor(x.float())
        x = x.view(x.size(0), -1)
        
        prediction_out = self.predictor(x)

        selection_out= self.pre_selector(x)
        if self.div_by_ten:
            selection_out /= 10.0
        selection_out = self.post_selector(selection_out)

        #selection_out = torch.ones_like(selection_out)
    
        auxiliary_out  = self.aux_predictor(x)

        return prediction_out, selection_out, auxiliary_out

    def _initialize_weights(self, module):
        for m in module.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm1d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)


class SelectiveNet(torch.nn.Module):
    """
    SelectiveNet for classification with rejection option.
    In the experiments of original papaer, variant of VGG-16 is used as body block for feature extraction.  
    """
    def __init__(self, features, dim_features:int, num_classes:int, init_weights=True, div_by_ten=False):
        """
        Args
            features: feature extractor network (called body block in the paper).
            dim_featues: dimension of features from body block.  
            num_classes: number of classification class.
        """
        super(SelectiveNet, self).__init__()
        self.features = features
        self.dim_features = dim_features
        self.num_classes = num_classes
        self.div_by_ten = div_by_ten
        
        # represented as f() in the original paper
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.dim_features, self.num_classes)
        )

        # represented as g() in the original paper
        self.pre_selector = torch.nn.Sequential(
            torch.nn.Linear(self.dim_features, self.dim_features),
            torch.nn.ReLU(True),
            torch.nn.BatchNorm1d(self.dim_features) 
        )

        self.post_selector = torch.nn.Sequential(
            torch.nn.Linear(self.dim_features, 1),
            torch.nn.Sigmoid()
        )
        

        # represented as h() in the original paper
        self.aux_classifier = torch.nn.Sequential(
            torch.nn.Linear(self.dim_features, self.num_classes)
        )

        # initialize weights of heads
        if init_weights:
            self._initialize_weights(self.classifier)
            self._initialize_weights(self.pre_selector)
            self._initialize_weights(self.post_selector)
            self._initialize_weights(self.aux_classifier)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        
        prediction_out = self.classifier(x)

        selection_out= self.pre_selector(x)
        if self.div_by_ten:
            selection_out /= 10.0
        selection_out = self.post_selector(selection_out)
    
        auxiliary_out  = self.aux_classifier(x)

        return prediction_out, selection_out, auxiliary_out

    def _initialize_weights(self, module):
        for m in module.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm1d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
    import os
    import sys

    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
    sys.path.append(base)

    from selectivenet.vgg_variant import vgg16_variant

    features = vgg16_variant(32,0.3).cuda()
    model = SelectiveNet(features,512,10).cuda()