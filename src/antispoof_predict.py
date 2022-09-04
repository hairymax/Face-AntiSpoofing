import torch
import torch.nn.functional as F
import torchvision.transforms as T
import os
from src.NN import MultiFTNet
from src.dataset_loader import SquarePad


# def parse_model_name(name):
#     model_type = name.split('_')[0]
#     input_size = int(name.split('.pth')[0].split('_')[-1])
#     return model_type, input_size  
    
class AntiSpoofPredict():
    def __init__(self, cnf):
        super(AntiSpoofPredict, self).__init__()
        self.device = cnf.device
        self.input_size = cnf.input_size
        self.kernel_size = cnf.kernel_size
        self.num_classes = cnf.num_classes
        self.model = MultiFTNet(conv6_kernel=self.kernel_size, 
                                num_classes=self.num_classes).to(self.device)
        self.model_path = cnf.model_path
        
        state_dict = torch.load(self.model_path, map_location=self.device)
        first_layer_name = iter(state_dict).__next__()
        if first_layer_name.find('module') >= 0:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for key, value in state_dict.items():
                name_key = key.replace('module.', '', 1)
                new_state_dict[name_key] = value
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(state_dict)

    def predict(self, img):
        # test_transform = T.Compose([
        #     T.ToPILImage(),
        #     SquarePad(),
        #     T.Resize(size=self.input_size),
        #     T.ToTensor()
        # ])
        # img = test_transform(img)
        # if len(img.size()) < 4:
        #     img = img.unsqueeze(0)
        img = img.to(self.device)
        self.model.eval()
        with torch.no_grad():
            result = self.model.forward(img)
            result = F.softmax(result, -1).cpu().numpy()
        return result