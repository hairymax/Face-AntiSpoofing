from collections import OrderedDict
import torch
import torch.nn.functional as F

from src.NN import MultiFTNet, MiniFASNetV2SE
    
class AntiSpoofPretrained:
    def __init__(self, cnf):
        self.device = cnf.device
        self.input_size = cnf.input_size
        self.kernel_size = cnf.kernel_size
        self.num_classes = cnf.num_classes
        self.model = MiniFASNetV2SE(conv6_kernel=self.kernel_size, 
                                    num_classes=self.num_classes).to(self.device)
        self.model_path = cnf.model_path
        
        state_dict = torch.load(self.model_path, map_location=self.device)
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            if not 'FTGenerator' in key:
                name_key = key.replace('module.model.', '', 1)
                new_state_dict[name_key] = value
        self.model.load_state_dict(new_state_dict)
        
    def predict(self, img):
        img = img.to(self.device)
        self.model.eval()
        with torch.no_grad():
            result = self.model.forward(img)
            result = F.softmax(result, -1).cpu().numpy()
        return result
    
class AntiSpoofPretrainedFT(AntiSpoofPretrained):
    def __init__(self, cnf):
        self.device = cnf.device
        self.input_size = cnf.input_size
        self.kernel_size = cnf.kernel_size
        self.num_classes = cnf.num_classes
        self.model = MultiFTNet(conv6_kernel=self.kernel_size, 
                                num_classes=self.num_classes).to(self.device)
        self.model_path = cnf.model_path
        
        state_dict = torch.load(self.model_path, map_location=self.device)
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            name_key = key.replace('module.', '', 1)
            new_state_dict[name_key] = value
        self.model.load_state_dict(new_state_dict)