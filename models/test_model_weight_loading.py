import torch
import sys
sys.path.insert(1, '../models/')
sys.path.insert(2, '../data/')

import full_image_colorization, fusion_module


# model = full_image_colorization.FullImageColorization()
# model.load_state_dict(torch.load("/Users/zhangenhao/Desktop/UW/2021SP/CSE576/576_Project_Colorization/checkpoints/siggraph_retrained/latest_net_G.pth"))
# print(model)

model = fusion_module.FusionModule()
# print(model.state_dict())
full_image_state_dict = torch.load("/Users/zhangenhao/Desktop/UW/2021SP/CSE576/576_Project_Colorization/checkpoints/siggraph_retrained/latest_net_G.pth")
instance_state_dict = torch.load("/Users/zhangenhao/Desktop/UW/2021SP/CSE576/576_Project_Colorization/checkpoints/siggraph_retrained/latest_net_G.pth")
# Load full-image network
model.load_state_dict(instance_state_dict, strict=False)
fusion_state_dict = model.state_dict()
# Load instance network
for name, param in instance_state_dict.items():
    if isinstance(param, torch.nn.parameter.Parameter):
        # backwards compatibility for serialized parameters
        param = param.data
    fusion_state_dict["instance_model." + name].copy_(param)
print(model.state_dict())