import torch

def save_model(model, path):
    '''
    '''
    torch.save(model.state_dict(), path)

def load_model(model_class, path, num_classes):
    '''
    '''
    model = model_class(num_classes)
    model.load_state_dict(torch.load(path))
    return model
