import os
import torch
import argparse
import importlib
from torch.backends import cudnn
cudnn.enabled = True
from configs.model_config import get_stage1_config
from tool.infer_fun import create_pseudo_mask

if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--weights", default='checkpoints/stage1_checkpoint_trained_on_luad.pth', type=str)
    # parser.add_argument("--network", default="network.resnet38_cls", type=str)
    # parser.add_argument("--dataroot", default="datasets/LUAD-HistoSeg", type=str)
    # parser.add_argument("--dataset", default="luad", type=str)
    # parser.add_argument("--num_workers", default=2, type=int)
    # parser.add_argument("--n_class", default=4, type=int)
    # args = parser.parse_args()
    
    args = get_stage1_config()
    print(args)
    
    if args.dataset == 'luad':
        palette = [0]*15
        palette[0:3] = [205,51,51] # Class 0: => Indian Red / Rust
        palette[3:6] = [0,255,0] # Class 1: => Green
        palette[6:9] = [65,105,225] # Class 2: => Blue
        palette[9:12] = [255,165,0] # Class 3: => Orange
        palette[12:15] = [255, 255, 255] # Class 4: => White
    elif args.dataset == 'bcss':
        palette = [0]*15
        palette[0:3] = [255, 0, 0] # Class 0: => pure Red
        palette[3:6] = [0,255,0] # Class 1: => pure Green
        palette[6:9] = [0,0,255] # Class 2: => pure Blue
        palette[9:12] = [153, 0, 255] # Class 3: => pure Purple
        palette[12:15] = [255, 255, 255] # Class 4: => pure White
        
    PMpath = os.path.join(args.dataroot,'train_PM')
    if not os.path.exists(PMpath):
        os.mkdir(PMpath)
    
    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = getattr(importlib.import_module("network.resnet38_cls"), 'Net_CAM')(n_class=args.n_class)
    weights_dict = torch.load(args.weights, map_location=device, weights_only=False)
    model.load_state_dict(weights_dict, strict=False)
    model.eval()
    model = model.to(device)
    
    ##
    fm = 'b4_5'
    savepath = os.path.join(PMpath,'PM_'+fm)
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    create_pseudo_mask(model, args.dataroot, fm, savepath, args.n_class, palette, args.dataset)
    
    ##
    fm = 'b5_2'
    savepath = os.path.join(PMpath,'PM_'+fm)
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    create_pseudo_mask(model, args.dataroot, fm, savepath, args.n_class, palette, args.dataset)
    
    ##
    fm = 'bn7'
    savepath = os.path.join(PMpath,'PM_'+fm)
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    create_pseudo_mask(model, args.dataroot, fm, savepath, args.n_class, palette, args.dataset)