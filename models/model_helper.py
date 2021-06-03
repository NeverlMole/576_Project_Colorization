import os
import torch
import full_image_colorization, fusion_module, instance_colorization

model_data_path = '../model_data/'

def get_model(file_name):
    '''
    This function should return an torch.nn model indicated by file_name. All
    the models should be stored in a folder. If the file_name already exists
    in the folder, the function should retrieve the model and return. If the
    file_name does not exists, the function should create a new model with
    struction indicated by the file_name.

    The file_name should start with its type:
        - Start with 'I', means the instance network.
        - Start with 'H', means the full image network.
        - Start with 'F', means the fusion of instance and full image network.
    '''
    model_path = model_data_path + file_name + '.pth'
    # Initialize model
    if file_name[0] == 'I':
        model = full_image_colorization.FullImageColorization()
    elif file_name[0] == 'H':
        model = full_image_colorization.FullImageColorization()
    elif file_name[0] == 'F':
        model = fusion_module.FusionModule()

    # Load model state dict if exists
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        return model, True
    return model, False

def save_model(model, file_name):
    '''
    The function should store the model into a file named file_name.
    '''
    # Save the model (state_dict) for inference
    model_path = model_data_path + file_name + '.pth'
    torch.save(model.state_dict(), model_path)

def get_fusion_model_for_training(full_image_filename, instance_filename):
    model = fusion_module.FusionModule()

    full_image_model_path = model_data_path + full_image_filename + '.pth'
    instance_model_path = model_data_path + instance_filename + '.pth'

    full_image_state_dict = torch.load(full_image_model_path)
    instance_state_dict = torch.load(instance_model_path)
    # Load full-image network
    model.load_state_dict(full_image_state_dict, strict=False)
    fusion_state_dict = model.state_dict()
    # Load instance network
    for name, param in instance_state_dict.items():
        if isinstance(param, torch.nn.parameter.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        fusion_state_dict["instance_model." + name].copy_(param)
    return model
