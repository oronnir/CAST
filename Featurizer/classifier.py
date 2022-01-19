import collections
import torch


class Classifier(torch.nn.Module):
    def __init__(self, num_classes, feature_planes):
        assert isinstance(num_classes, int) or isinstance(num_classes, collections.abc.Iterable)
        super(Classifier, self).__init__()
        self.feature_planes = feature_planes
        if isinstance(num_classes, int):
            num_classes = (num_classes,)
        self.num_classes = num_classes
        self.features = None  # Need to be initialized by a child class.
        self.fcs = torch.nn.ModuleList([torch.nn.Linear(feature_planes, c) for c in num_classes])

        self.outputs = []

    def forward(self, x, classifier_index=0):
        if isinstance(x, list):
            sizes = [y.size()[0] for y in x]
            x = torch.cat(x, 0)
            out = self.features(x)
            out = out.view(-1, self.feature_planes)
            outs = torch.split(out, sizes, dim=0)
            return [self.fcs[i](out) for i, out in enumerate(outs)]
        else:
            assert classifier_index < len(self.fcs)
            out = self.features(x)
            out = out.view(-1, self.feature_planes)
            return self.fcs[classifier_index](out)

    def _forward_hook(self, module, input, output):
        self.outputs.append(output)
    
    def get_outputs(self, x, layer_names):
        for layer_name in layer_names:
            layer = self.find_layer_by_name(layer_name)
            layer.register_forward_hook(self.forward_hook)
            
        self.outputs = []
        self.forward(x)
        return self.outputs

    def find_layer_by_name(self, name):
        paths = name.split('.')
        current = self
        for p in paths:
            children = current.named_children()
            current = children[p]
        return current

    def make_feature_extractor(self):
        for param in self.features.parameters():
            param.requires_grad = False

    def load_feature_extractor_state(self, state_dict):
        filtered_dict = {}
        features_state_dict = self.features.state_dict()
        for name in features_state_dict.keys():
            name = 'features.' + name
            if name in state_dict:
                filtered_dict[name] = state_dict[name]
            else:
                print("Warning: {} is missing".format(name))
                
        self.load_state_dict(filtered_dict, strict=False)
        
        
class SimplisticClassifier(Classifier):
    def __init__(self, num_classes, feature_planes):
        super(SimplisticClassifier, self).__init__(num_classes, feature_planes)
        self.features = lambda x: x
