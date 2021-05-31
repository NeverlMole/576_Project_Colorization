import torch
import sys
sys.path.insert(1, '../models/')
sys.path.insert(2, '../data/')

import full_image_colorization


model = full_image_colorization.FullImageColorization()
model.load_state_dict(torch.load("/Users/zhangenhao/Desktop/UW/2021SP/CSE576/576_Project_Colorization/checkpoints/siggraph_retrained/latest_net_G.pth"))
print(model)