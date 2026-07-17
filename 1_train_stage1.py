import os
import numpy as np
from configs.model_config import get_stage1_config
# import argparse
import importlib
# from visdom import Visdom
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


import torch
import torch.nn.functional as F
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from tool import pyutils, torchutils
from tool.GenDataset import Stage1_TrainDataset
from tool.infer_fun import infer
cudnn.enabled = True

def compute_acc(pred_labels, gt_labels):
    pred_correct_count = 0
    for pred_label in pred_labels:
        if pred_label in gt_labels:
            pred_correct_count += 1
    union = len(gt_labels) + len(pred_labels) - pred_correct_count
    acc = round(pred_correct_count/union, 4)
    return acc

def train_phase(args):
    # viz = Visdom(env=args.env_name)
    # 1. Setup the writer instead of Visdom (log_dir replaces the env_name concept)
    log_dir = os.path.join("runs", args.env_name)
    writer = SummaryWriter(log_dir=log_dir)
    
    # 1. Define the runtime device context dynamically
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = getattr(importlib.import_module(args.network), 'Net')(args.init_gama, n_class=args.n_class)
    print(vars(args))
    transform_train = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                  transforms.RandomVerticalFlip(p=0.5),
                                  transforms.ToTensor()]) 
    train_dataset = Stage1_TrainDataset(data_path=args.trainroot,transform=transform_train, dataset=args.dataset)
    train_data_loader = DataLoader(train_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.num_workers,
                                    pin_memory=False,
                                    drop_last=True)
    max_step = (len(train_dataset) // args.batch_size) * args.max_epoches
    param_groups = model.get_parameter_groups()
    optimizer = torchutils.PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wt_dec},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0}
    ], lr=args.lr, weight_decay=args.wt_dec, max_step=max_step)
    # if args.weights[-7:] == '.params':
    #     assert args.network == "network.resnet38_cls"
    #     import network.resnet38d
    #     weights_dict = network.resnet38d.convert_mxnet_to_torch(args.weights)
    #     model.load_state_dict(weights_dict, strict=False)
    # elif args.weights[-4:] == '.pth':
    if args.weights[-4:] == '.pth':
        # weights_dict = torch.load(args.weights)
        weights_dict = torch.load( 
          args.weights, 
          map_location="cpu",
          weights_only=False 
          )
        model.load_state_dict(weights_dict, strict=False)
    else:
        print('random init')
        
    # model = model.cuda()
    # 2. Transfer the model to the target device
    model = model.to(torch.device)
    
    avg_meter = pyutils.AverageMeter(
            'loss',
            'avg_ep_EM',
            'avg_ep_acc')
    timer = pyutils.Timer("Session started: ")
    for ep in range(args.max_epoches):
        model.train()
        args.ep_index = ep
        ep_count = 0
        ep_EM = 0
        ep_acc = 0
        for iter, (filename, data, label) in enumerate(train_data_loader):
            img = data
            # label = label.cuda(non_blocking=True)
            # Note: non_blocking=True is safely ignored by PyTorch if it falls back to CPU
            label = label.to(device, non_blocking=True)
            img_device = img.to(device)
            if ep > 2:
                enable_PDA = 1
            else:
                enable_PDA = 0
            x, feature, y = model(img_device, enable_PDA)
            prob = y.cpu().data.numpy()
            gt = label.cpu().data.numpy()
            for num, one in enumerate(prob):
                ep_count += 1
                pass_cls = np.where(one > 0.5)[0]
                true_cls = np.where(gt[num] == 1)[0]
                if np.array_equal(pass_cls, true_cls) == True:  # exact match
                    ep_EM += 1
                acc = compute_acc(pass_cls, true_cls)
                ep_acc += acc
            avg_ep_EM = round(ep_EM/ep_count, 4)
            avg_ep_acc = round(ep_acc/ep_count, 4)
            loss = F.multilabel_soft_margin_loss(x, label)
            avg_meter.add({'loss':loss.item(),
                            'avg_ep_EM':avg_ep_EM,
                            'avg_ep_acc':avg_ep_acc,
                           })
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Conditional cache cleanup (only runs if a GPU is actively attached)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            # torch.cuda.empty_cache()
            if (optimizer.global_step)%20 == 0 and (optimizer.global_step)!=0:
                timer.update_progress(optimizer.global_step / max_step)

                print('Epoch:%2d' % (ep),
                      'Iter:%5d/%5d' % (optimizer.global_step, max_step),
                      'Loss:%.4f' % (avg_meter.get('loss')),
                      'avg_ep_EM:%.4f' % (avg_meter.get('avg_ep_EM')),
                      'avg_ep_acc:%.4f' % (avg_meter.get('avg_ep_acc')),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']), 
                      'Fin:%s' % (timer.str_est_finish()),
                      flush=True)
                # viz.line([avg_meter.pop('loss')],[optimizer.global_step],win='loss',update='append',opts=dict(title='loss'))
                # viz.line([avg_meter.pop('avg_ep_EM')],[optimizer.global_step],win='Acc_exact',update='append',opts=dict(title='Acc_exact'))
                # viz.line([avg_meter.pop('avg_ep_acc')],[optimizer.global_step],win='Acc',update='append',opts=dict(title='Acc'))
                # 2. Log metrics (Use scalar instead of line)
                # Note: Using .get() instead of .pop() is usually safer so you don't delete 
                # the values from your metric dictionary mid-loop.
                writer.add_scalar('Loss/train', avg_meter.get('loss'), optimizer.global_step)
                writer.add_scalar('Accuracy/exact', avg_meter.get('avg_ep_EM'), optimizer.global_step)
                writer.add_scalar('Accuracy/standard', avg_meter.get('avg_ep_acc'), optimizer.global_step)
        if model.gama > 0.65:
            model.gama = model.gama*0.98
        print('Gama of progressive dropout attention is: ',model.gama)
    writer.close()
    torch.save(model.state_dict(), os.path.join(args.save_folder, 'stage1_checkpoint_trained_on_'+args.dataset+'.pth'))

def test_phase(args):
    model = getattr(importlib.import_module(args.network), 'Net_CAM')(n_class=args.n_class)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.cuda()
    model = model.to(device)
    
    args.weights = os.path.join(args.save_folder, 'stage1_checkpoint_trained_on_'+args.dataset+'.pth')
    # weights_dict = torch.load(args.weights)
    weights_dict = torch.load(args.weights, 
                                  map_location="cpu", 
                                  weights_only=False 
                                  ) 
    model.load_state_dict(weights_dict, strict=False)
    model.eval()
    # print(args.testroot)
    
    # score = infer(model, args.testroot, args.n_class)
    # print(score)
    
    # 2. Wrap the infer function with a description
    with tqdm(total=1, desc="Running Inference", bar_format="{desc}: {elapsed}") as pbar:
        score = infer(model, args.testroot, args.n_class)
        pbar.update(1)
        
    print(score)
    
    torch.save(model.state_dict(), os.path.join(args.save_folder, 'stage1_checkpoint_trained_on_'+args.dataset+'.pth'))

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--batch_size", default=20, type=int)
    # parser.add_argument("--max_epoches", default=20, type=int)
    # parser.add_argument("--network", default="network.resnet38_cls", type=str)
    # parser.add_argument("--lr", default=0.01, type=float)
    # # parser.add_argument("--num_workers", default=10, type=int) # changed num_workers is 2 because colab dataloader can handle only 2 workers 
    # parser.add_argument("--num_workers", default=2, type=int)
    # parser.add_argument("--wt_dec", default=5e-4, type=float)
    # parser.add_argument("--session_name", default="Stage 1", type=str)
    # parser.add_argument("--env_name", default="PDA", type=str)
    # parser.add_argument("--model_name", default='PDA', type=str)
    # parser.add_argument("--n_class", default=4, type=int)

    # parser.add_argument("--weights", default='init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.pth')
    # # parser.add_argument("--weights", default='init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.params', type=str)
    # parser.add_argument("--trainroot", default='datasets/LUAD-HistoSeg/train300/', type=str)
    # parser.add_argument("--testroot", default='datasets/LUAD-HistoSeg/test/', type=str)
    # parser.add_argument("--save_folder", default='checkpoints/',  type=str)
    # parser.add_argument("--init_gama", default=1, type=float)
    # parser.add_argument("--dataset", default='bcss', type=str)
    # args = parser.parse_args()
    
    # Automatically parses all arguments, builds Path objects, and sets up folders
    args = get_stage1_config()

    # You can access your arguments clean and safe now:
    print(f"Running session: {args.session_name} with network: {args.network}")
    print(f"training Data directory: {args.trainroot}")
    print(f"testing Data directory: {args.testroot}")

    # train_phase(args)
    test_phase(args)
    
    
"""
 How to see the charts:
Open your terminal, navigate to your project directory, and launch the dashboard by running:

BASH
tensorboard --logdir=logs
or
python -m tensorboard.main --logdir=logs

It will give you a local URL link (usually http://localhost:6006/). Open that in your web browser, and you will see dynamic line charts updating in real-time as your code runs.  
"""