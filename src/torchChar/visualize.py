import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm.autonotebook import tqdm

def visualize_neuron(model, layer_name):
    model.register_output_hook(layer_name)
    model.eval()
    for param in model.parameters():
        param.requires_grad=False

    try:
        layer = getattr(model, layer_name)
        if layer_name.startswith("conv"):
            n_out = getattr(layer, "out_channels")
        else:
            n_out = getattr(layer, "out_features")
    except AttributeError as ex:
        print(ex)
        return

    filter_imgs = []
    for filter_i in tqdm(range(n_out)):
        filter_img = visualize_filter(model, layer_name, filter_i)
        filter_imgs.append(filter_img)

    model.hooks[layer_name].remove()
    return filter_imgs

def visualize_filter(model, layer_name, filter_idx):
    X = torch.FloatTensor(1,64,64).random_(0,255)
    X.requires_grad=True
    optimizer = optim.Adam([X], lr=20)        
    for _ in range(20):
        optimizer.zero_grad()
        model(X)
        act = F.relu(model.layer_outputs[layer_name])
        mean_acts = act.mean(axis=[2,3])        
        loss = -mean_acts[0, filter_idx]    
        loss.backward()
        optimizer.step()
    final_X = X.detach().clone()
    return final_X