from configs.model_config import get_stage2_config
import os
import numpy as np
from tqdm import tqdm
import torch
from tool.GenDataset import make_data_loader
from network.sync_batchnorm.replicate import patch_replication_callback
from network.deeplab import *
from tool.loss import SegmentationLosses
from tool.lr_scheduler import LR_Scheduler
from tool.saver import Saver
from tool.summaries import TensorboardSummary
from tool.metrics import Evaluator

class Trainer(object):
    def __init__(self, args):
        self.args = args
        # Define device based on availability and arguments
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
     
        self.saver = Saver(args)
        self.summary = TensorboardSummary('logs')
        self.writer = self.summary.create_summary()
        kwargs = {'num_workers': args.workers, 'pin_memory': False}
        self.train_loader, self.val_loader, self.test_loader = make_data_loader(args, **kwargs)
        self.nclass = args.n_class
        
        model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)
        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)
       
        # self.criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.criterion = SegmentationLosses(weight=None, cuda=self.device.type == "cuda").build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer
        self.evaluator = Evaluator(self.nclass)
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.train_loader))

        ##  Create ResNet38 and load the weights of Stage 1.
        import importlib
        model_stage1 = getattr(importlib.import_module('network.resnet38_cls'), 'Net_CAM')(n_class=4)
        resume_stage1 = 'checkpoints/stage1_checkpoint_trained_on_'+str(args.dataset)+'.pth'
        
        # Map the weights to the correct device (stops it from crashing on CPU-only machines)
        weights_dict = torch.load(resume_stage1, map_location=self.device)
        model_stage1.load_state_dict(weights_dict)
        
        # self.model_stage1 = model_stage1.cuda()
        self.model_stage1 = model_stage1.to(self.device)
        self.model_stage1.eval()

       # Using cuda/cpu dynamically for main model
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            self.model = self.model.cuda()
        else:
            self.model = self.model.to(self.device)
            
        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            
            # checkpoint = torch.load(args.resume)
            checkpoint = torch.load(
              args.resume,
              map_location=self.device,
              weights_only=False
              )
              
            if args.cuda:
                W = checkpoint['state_dict']
                if not args.ft:
                    del W['decoder.last_conv.8.weight']
                    del W['decoder.last_conv.8.bias']
                self.model.module.load_state_dict(W, strict=False)
            else:
                W = checkpoint['state_dict']
                if not args.ft:
                    # Clean the keys if the saved checkpoint used DataParallel but we are loading on CPU
                    W = {k.replace('module.', ''): v for k, v in W.items()}
                    if 'decoder.last_conv.8.weight' in W: del W['decoder.last_conv.8.weight']
                    if 'decoder.last_conv.8.bias' in W: del W['decoder.last_conv.8.bias']
                    
                self.model.load_state_dict(W, strict=False)
                
            if args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' ".format(args.resume))

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            image, target, target_a, target_b = sample['image'], sample['label'], sample['label_a'], sample['label_b']
            # if self.args.cuda:
            #     image, target, target_a, target_b = image.cuda(), target.cuda(), target_a.cuda(), target_b.cuda()
            image, target, target_a, target_b = image.to(self.device), target.to(self.device), target_a.to(self.device), target_b.to(self.device)

            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            
            # one = torch.ones((output.shape[0],1,224,224)).cuda()
            one = torch.ones((output.shape[0],1,224,224), device=self.device)
            
            output = torch.cat([output,(100 * one * (target==4).unsqueeze(dim = 1))],dim = 1)

            loss_o = self.criterion(output, target)
            loss_a = self.criterion(output, target_a)
            loss_b = self.criterion(output, target_b)
            loss = 0.6*loss_o + 0.2*loss_a + 0.2*loss_b

            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        
        for i, sample in enumerate(tbar):
            image, target = sample[0]['image'], sample[0]['label']
            # if self.args.cuda:
            #     image, target = image.cuda(), target.cuda()
            image, target = image.to(self.device), target.to(self.device)
            
            with torch.no_grad():
                output = self.model(image)
                
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            ## cls 4 is exclude
            pred[target==4]=4
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        ious = self.evaluator.Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)
        print('IoUs: ', ious)

        if mIoU > self.best_pred:
            self.best_pred = mIoU
            # Safely extract state_dict whether DataParallel is active or not
            state_dict = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
            
            self.saver.save_checkpoint({
                'state_dict':state_dict,
                'optimizer': self.optimizer.state_dict()
            }, 'stage2_checkpoint_trained_on_v2_'+self.args.dataset+'.pth')
   
    def load_the_best_checkpoint(self):
        checkpoint = torch.load('checkpoints/stage2_checkpoint_trained_on_'+self.args.dataset+'.pth', map_location=self.device)
        
        # self.model.module.load_state_dict(checkpoint['state_dict'], strict=False)
        # Check if DataParallel is being used or if we are on a standard CPU/GPU model
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            # If the checkpoint was saved with DataParallel, we clean the keys for the raw model
            state_dict = checkpoint['state_dict']
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict, strict=False)
   
    def test(self, epoch, Is_GM):
        self.load_the_best_checkpoint()
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.test_loader, desc='\r')
        test_loss = 0.0
        
        for i, sample in enumerate(tbar):
            image, target = sample[0]['image'], sample[0]['label']
            # if self.args.cuda:
            #     image, target = image.cuda(), target.cuda()
            image, target = image.to(self.device), target.to(self.device)
            
            with torch.no_grad():
                output = self.model(image)
                if Is_GM:
                    # output = self.model(image)
                    _,y_cls = self.model_stage1.forward_cam(image)
                    y_cls = y_cls.cpu().data
                    pred_cls = (y_cls > 0.1)
                    
            pred = output.data.cpu().numpy()
            if Is_GM:
                pred = pred*(pred_cls.unsqueeze(dim=2).unsqueeze(dim=3).numpy())
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            ## cls 4 is exclude
            pred[target==4]=4
            self.evaluator.add_batch(target, pred)

        # Fast testing
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        ious = self.evaluator.Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        
        print('Test:')
        print('[numImages: %5d]' % (i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)
        print('IoUs: ', ious)

def main():
    # parser = argparse.ArgumentParser(description="WSSS Stage2")
    # parser.add_argument('--backbone', type=str, default='resnet', choices=['resnet', 'xception', 'drn', 'mobilenet'])
    # parser.add_argument('--out-stride', type=int, default=16)
    # parser.add_argument('--Is_GM', type=bool, default=True, help='Enable the Gate mechanism in test phase')
    # parser.add_argument('--dataroot', type=str, default='datasets/BCSS-WSSS/')
    # parser.add_argument('--dataset', type=str, default='bcss')
    # parser.add_argument('--savepath', type=str, default='checkpoints/')
    # parser.add_argument('--workers', type=int, default=2, metavar='N')
    # parser.add_argument('--sync-bn', type=bool, default=None)
    # parser.add_argument('--freeze-bn', type=bool, default=False)
    # parser.add_argument('--loss-type', type=str, default='ce', choices=['ce', 'focal'])
    # parser.add_argument('--n_class', type=int, default=4)
    # # training hyper params
    # parser.add_argument('--epochs', type=int, default=30, metavar='N')
    # parser.add_argument('--batch-size', type=int, default=16, metavar='N')
    # # optimizer params
    # parser.add_argument('--lr', type=float, default=0.01, metavar='LR')
    # parser.add_argument('--lr-scheduler', type=str, default='poly',choices=['poly', 'step', 'cos'])
    # parser.add_argument('--momentum', type=float, default=0.9, metavar='M')
    # parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='M')
    # parser.add_argument('--nesterov', action='store_true', default=False )
    # # cuda, seed and logging
    # parser.add_argument('--no-cuda', action='store_true', default=False)
    # parser.add_argument('--gpu-ids', type=str, default='0')
    # parser.add_argument('--seed', type=int, default=1, metavar='S')
    # # checking point
    # parser.add_argument('--resume', type=str, default='init_weights/deeplab-resnet.pth.tar')
    # parser.add_argument('--checkname', type=str, default='deeplab-resnet')
    # parser.add_argument('--ft', action='store_true', default=False)
    # parser.add_argument('--eval-interval', type=int, default=1)
    # args = parser.parse_args()
    args = get_stage2_config()
    
    # Configure runtime device settings (Graceful CPU fallback if CUDA is unavailable)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')
    else:
        # Fallback tracking if CUDA is not initialized
        args.gpu_ids = []
        
    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False
            
    # Log arguments for traceability
    print("Final run configurations:", args)

    # Initialize Trainer
    trainer = Trainer(args)
    # for epoch in range(trainer.args.epochs):
    #     trainer.training(epoch)
    #     trainer.validation(epoch)
    
    # trainer.test(epoch, args.Is_GM)
    # trainer.writer.close()
    
    if args.test_only:
        print("--- [TEST ONLY MODE] ---")
        print(f"Skipping training loop. Directly evaluating with model checkpoint: {args.resume}")
        # Execute testing immediately
        # We pass 0 as the mock epoch index to meet standard evaluation function contracts.
        trainer.test(0, args.Is_GM)
    else:
        print("--- [TRAINING AND EVALUATION MODE] ---")
        for epoch in range(trainer.args.epochs):
            trainer.training(epoch)
            trainer.validation(epoch)
            
        # Run final test phase on completion of epochs
        trainer.test(epoch, args.Is_GM)
        
    trainer.writer.close()

if __name__ == "__main__":
   main()