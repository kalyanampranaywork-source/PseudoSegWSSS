# import importlib as imp
# import pdb
import contextlib

from tqdm import tqdm

import numpy as np
import torch
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
from tool import pyutils, iouutils
from PIL import Image
import torch.nn.functional as F
import os.path
import cv2
from tool import infer_utils
from tool.GenDataset import Stage1_InferDataset
from torchvision import transforms
from tool.gradcam import GradCam
def CVImageToPIL(img):
    img = img[:,:,::-1]
    img = Image.fromarray(np.uint8(img))
    return img
def PILImageToCV(img):
    img = np.asarray(img)
    img = img[:,:,::-1]
    return img

def fuse_mask_and_img(mask, img):
    mask = PILImageToCV(mask)
    img = PILImageToCV(img)
    Combine = cv2.addWeighted(mask,0.3,img,0.7,0)
    return Combine

def infer(model, dataroot, n_class):
    # pdb.set_trace()
    model.eval()
    n_gpus = torch.cuda.device_count()
    model_replicas = torch.nn.parallel.replicate(model, list(range(n_gpus)))
    cam_list = []
    gt_list = []    
    bg_list = []
    transform = transforms.Compose([transforms.ToTensor()]) 
    infer_dataset = Stage1_InferDataset(data_path=os.path.join(dataroot,'img'),transform=transform)
    # infer_data_loader = DataLoader(infer_dataset,
    #                             shuffle=False,
    #                             num_workers=8,
    #                             pin_memory=False)
    infer_data_loader = DataLoader(infer_dataset,
                                shuffle=False,
                                num_workers=2,
                                pin_memory=False)
    
    progress_bar = tqdm(
        enumerate(infer_data_loader),
        total=len(infer_data_loader),
        desc="Stage-1 Inference",
        unit="image",
        dynamic_ncols=True,
        leave=True,
    )
    
    # for iter, (img_name, img_list) in enumerate(infer_data_loader):
    for iter, (img_name, img_list) in progress_bar:
        img_name = img_name[0]; 

        # img_path = os.path.join(os.path.join(dataroot,'img'),img_name+'.png')
        # img_path = os.path.join(dataroot,img_name+'.png')
        img_path = os.path.join(os.path.join(dataroot,'img'),os.path.basename(img_name)+'.png') 
        
        orig_img = np.asarray(Image.open(img_path))
        orig_img_size = orig_img.shape[:2]

        def _work(i, img, thr=0.15):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            with torch.no_grad():
                # with torch.cuda.device(i%n_gpus):
                if torch.cuda.is_available() and n_gpus > 0:
                    device_context = torch.cuda.device(i % n_gpus)
                    active_model = model_replicas[i % n_gpus]
                else:
                    # Safe fallback if running on CPU or n_gpus is evaluated as 0
                    device_context = contextlib.nullcontext() 
                    active_model = model_replicas[0] if (
                        'model_replicas' in locals() and len(model_replicas) > 0
                    ) else model

                with device_context:
                    # cam, y = model_replicas[i%n_gpus].forward_cam(img.to(device))
                    cam, y = active_model.forward_cam(img.to(device))
                    
                    # print("Image:", img_name)
                    # print("Raw prediction:", y.cpu().numpy())

                    y = y.cpu().detach().numpy().tolist()[0]
                    label = torch.tensor([1.0 if j >thr else 0.0 for j in y])
                    # cam = F.upsample(cam, orig_img_size, mode='bilinear', align_corners=False)[0]
                    cam = F.interpolate(cam, size=orig_img_size, mode='bilinear', align_corners=False)[0]
                    cam = cam.cpu().numpy() * label.clone().view(4, 1, 1).numpy()
                    return cam, label

        # thread_pool = pyutils.BatchThreader(_work, list(enumerate(img_list.unsqueeze(0))),
                                            # batch_size=12, prefetch_size=0, processes=8)
        thread_pool = pyutils.BatchThreader(_work, list(enumerate(img_list.unsqueeze(0))),
                                            batch_size=12, prefetch_size=0, processes=2)
        cam_pred = thread_pool.pop_results()
        cams = [pair[0] for pair in cam_pred]
        label = [pair[1] for pair in cam_pred][0]
        sum_cam = np.sum(cams, axis=0)
        # norm_cam = (sum_cam-np.min(sum_cam)) / (np.max(sum_cam)-np.min(sum_cam))
        cam_min = np.min(sum_cam)
        cam_max = np.max(sum_cam)

        if cam_max > cam_min:
            norm_cam = (sum_cam - cam_min) / (cam_max - cam_min)
        else:
            norm_cam = np.zeros_like(sum_cam)

        # cam --> segmap
        cam_dict = infer_utils.cam_npy_to_cam_dict(norm_cam, label)
        # print("Image:", img_path)
        cam_score, bg_score = infer_utils.dict2npy(cam_dict, label, orig_img, None)
        seg_map = infer_utils.cam_npy_to_label_map(cam_score)
        
        progress_bar.set_postfix(
            Image=os.path.basename(img_name),
            Processed=f"{iter + 1}/{len(infer_data_loader)}",
        )
        
        if iter%100==0:
            print(iter)
            
        cam_list.append(seg_map)
        # gt_map_path = os.path.join(os.path.join(dataroot,'mask'), img_name + '.png') 
        gt_map_path = os.path.join(os.path.join(dataroot,'mask'), os.path.basename(img_name) + '.png') 
        
        gt_map = np.array(Image.open(gt_map_path))
        gt_list.append(gt_map)
    return iouutils.scores(gt_list, cam_list, n_class=n_class)

      
def create_pseudo_mask(model, dataroot, fm, savepath, n_class, palette, dataset):
    # print(model)
    print("create_pseudo_mask() called")
    print("Save path:", savepath)
    if fm=='b4_3':
        ffmm = model.b4_3
    elif fm=='b4_5':
        ffmm = model.b4_5
    elif fm=='b5_2':
        ffmm = model.b5_2
    elif fm=='b6':
        ffmm = model.b6
    elif fm=='bn7':
        ffmm = model.bn7
    else:
        print('error')
        return
    print(dataset)
    transform = transforms.Compose([transforms.ToTensor()]) 
    infer_dataset = Stage1_InferDataset(data_path=os.path.join(dataroot,'train5'),transform=transform)
    
    print("Dataset size:", len(infer_dataset))

    infer_data_loader = DataLoader(infer_dataset,
                                shuffle=False,
                                num_workers=2,
                                pin_memory=False)
    
    for iter, (img_name, img_list) in enumerate(infer_data_loader):      
        img_name = img_name[0]
        img_path = os.path.join(os.path.join(dataroot,'train5'),os.path.basename(img_name)+'.png')
        orig_img = np.asarray(Image.open(img_path))
        grad_cam = GradCam(model=model, feature_module=ffmm, \
                target_layer_names=["1"], use_cuda=True)
        cam = []
        for i in range(n_class):
            target_category = i
            grayscale_cam, _ = grad_cam(img_list, target_category)
            cam.append(grayscale_cam)
        norm_cam = np.array(cam)
        _range = np.max(norm_cam) - np.min(norm_cam)
        norm_cam = (norm_cam - np.min(norm_cam))/_range
        ##  Extract the image-level label from the filename
        ##  LUAD-HistoSeg   : 'Image-name-of-BCSS'+'+index'+'[abcd]'.png
        ##  BCSS-WSSS       : 'patient_ID'+'_x-axis'+'_y-axis'+'[a b c d]'.png
        label_str = img_name.split(']')[0].split('[')[-1]
        if dataset == 'luad':
            label = torch.Tensor([int(label_str[0]),int(label_str[2]),int(label_str[4]),int(label_str[6])])
        elif dataset == 'bcss':
            label = torch.Tensor([int(label_str[0]),int(label_str[1]),int(label_str[2]),int(label_str[3])])

        cam_dict = infer_utils.cam_npy_to_cam_dict(norm_cam, label)
        cam_score, bg_score = infer_utils.dict2npy(cam_dict, label, orig_img, None) #此处加入了背景，做修改
        ##  "bg_score" is the white area generated by "cv2.threshold".
        ##  Since lungs are the main organ of the respiratory system. There are a lot of alveoli (some air sacs) serving for exchanging the oxygen and carbon dioxide, which forms some white background in WSIs.
        ##  For LUAD-HistoSeg, we uses it in the pseudo-annotation generation phase to avoid some meaningless areas to participate in the training phase of stage2.
        if dataset == 'luad':
            bgcam_score = np.concatenate((cam_score, bg_score), axis=0)
        ##  Since the white background of images of breast cancer is meaningful (e.g. fat, etc), we do not use it for the training set of BCSS-WSSS.
        elif dataset == 'bcss':
            bg_score = np.zeros((1,224,224))
            bgcam_score = np.concatenate((cam_score, bg_score), axis=0)
        seg_map = infer_utils.cam_npy_to_label_map(bgcam_score) 
        visualimg  = Image.fromarray(seg_map.astype(np.uint8), "P")
        visualimg.putpalette(palette)
        visualimg.save(os.path.join(savepath, os.path.basename(img_name)+'.png'), format='PNG')

        if iter%100==0:           
            print(iter)
            


def create_pseudo_mask_v2(
    model,
    dataroot,
    fm,
    savepath,
    n_class,
    palette,
    dataset,
    logger=None,
):
    """
    Generate pseudo masks from a selected feature map.

    Parameters
    ----------
    model : torch.nn.Module

    dataroot : str

    fm : str

    savepath : str

    n_class : int

    palette : list

    dataset : str

    logger : logging.Logger, optional
    """

    # ------------------------------------------------------------
    # Select Feature Map
    # ------------------------------------------------------------
    
    if fm == "b4_3":
        ffmm = model.b4_3
    elif fm == "b4_5":
        ffmm = model.b4_5
    elif fm == "b5_2":
        ffmm = model.b5_2
    elif fm == "b6":
        ffmm = model.b6
    elif fm == "bn7":
        ffmm = model.bn7
    else:
        raise ValueError(f"Unknown feature map: {fm}")

    os.makedirs(savepath, exist_ok=True)

    feature_modules = {
        "b4_3": model.b4_3,
        "b4_5": model.b4_5,
        "b5_2": model.b5_2,
        "b6": model.b6,
        "bn7": model.bn7,
    }

    if fm not in feature_modules:
        raise ValueError(f"Unknown feature map: {fm}")

    ffmm = feature_modules[fm]

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    infer_dataset = Stage1_InferDataset(
        data_path=os.path.join(dataroot, "train5"),
        transform=transform,
    )

    infer_loader = DataLoader(
        infer_dataset,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
    )

    if logger is not None:

        logger.info("=" * 80)
        logger.info(f"Generating pseudo masks from feature map : ({fm})")
        logger.info("=" * 80)
        logger.info(f"Dataset          : {dataset}")
        logger.info(f"Feature Map      : {fm}")
        logger.info(f"Images           : {len(infer_dataset)}")
        logger.info(f"Output Directory : {savepath}")

    # ------------------------------------------------------------
    # Generate Masks
    # ------------------------------------------------------------

    progress = tqdm(
        infer_loader,
        desc=f"{fm}",
        unit="image",
        dynamic_ncols=True,
    )
    for iteration, (img_name, img_list) in enumerate(progress):
    # for _, (img_name, img_list) in enumerate(progress):

        img_name = img_name[0]

        img_path = os.path.join(
            dataroot,
            "train5",
            os.path.basename(img_name) + ".png",
        )

        orig_img = np.asarray(
            Image.open(img_path)
        )

        grad_cam = GradCam(
            model=model,
            feature_module=ffmm,
            target_layer_names=["1"],
            use_cuda=torch.cuda.is_available(),
        )

        cam = []

        for class_id in range(n_class):

            grayscale_cam, _ = grad_cam(
                img_list,
                class_id,
            )

            cam.append(grayscale_cam)

        norm_cam = np.array(cam)

        cam_min = np.min(norm_cam)
        cam_max = np.max(norm_cam)

        if cam_max > cam_min:

            norm_cam = (norm_cam - cam_min) / (
                cam_max - cam_min
            )

        else:

            norm_cam = np.zeros_like(norm_cam)

        # --------------------------------------------------------
        # Parse Image-Level Label
        # --------------------------------------------------------

        label_str = img_name.split("]")[0].split("[")[-1]

        if dataset == "luad":

            label = torch.tensor(
                [
                    int(label_str[0]),
                    int(label_str[2]),
                    int(label_str[4]),
                    int(label_str[6]),
                ]
            )

        else:

            label = torch.tensor(
                [
                    int(label_str[0]),
                    int(label_str[1]),
                    int(label_str[2]),
                    int(label_str[3]),
                ]
            )

        cam_dict = infer_utils.cam_npy_to_cam_dict(
            norm_cam,
            label,
        )

        cam_score, bg_score = infer_utils.dict2npy(
            cam_dict,
            label,
            orig_img,
            None,
        )

        if dataset == "luad":

            bgcam_score = np.concatenate(
                (
                    cam_score,
                    bg_score,
                ),
                axis=0,
            )

        elif dataset == "bcss":

            bg_score = np.zeros((1, 224, 224))

            bgcam_score = np.concatenate(
                (
                    cam_score,
                    bg_score,
                ),
                axis=0,
            )

        seg_map = infer_utils.cam_npy_to_label_map(
            bgcam_score
        )

        visual = Image.fromarray(
            seg_map.astype(np.uint8),
            "P",
        )

        visual.putpalette(
            palette
        )

        visual.save(
            os.path.join(
                savepath,
                os.path.basename(img_name) + ".png",
            ),
            format="PNG",
        )

        progress.set_postfix(
            Image=os.path.basename(img_name),
        )
        
        if logger is not None and (iteration + 1) % 100 == 0:

            logger.info(
                f"[{fm}] Processed {iteration + 1}/{len(infer_dataset)} images"
            )

    progress.close()

    if logger is not None:

        logger.info(
            f"Completed pseudo-mask generation ({fm})"
        )

        logger.info(
            f"Saved masks to : {savepath}"
        )

        logger.info("")
        
        

from tool.gradcam import GradCAMv2


def create_pseudo_mask_v3(model, layer_name, dataroot, savepath, n_class, palette, dataset, device, logger=None):
    os.makedirs(savepath, exist_ok=True)
    
    # print("=== All modules in model ===")
    # for name, module in model.named_modules():
    #     print(name, "->", type(module).__name__)

    # One GradCAM extractor per layer -- hooks are attached once here
    cam_extractor = GradCAMv2(model, layer_name, device)

    transform = transforms.Compose([transforms.ToTensor()])
    infer_dataset = Stage1_InferDataset(data_path=os.path.join(dataroot, 'train5'), transform=transform)
    infer_loader = DataLoader(infer_dataset, shuffle=False, num_workers=2, pin_memory=False)
    
    if logger is not None:

        logger.info("=" * 80)
        logger.info(f"Generating pseudo masks from feature map : ({layer_name})")
        logger.info("=" * 80)
        logger.info(f"Dataset          : {dataset}")
        logger.info(f"Feature Map      : {layer_name}")
        logger.info(f"Images           : {len(infer_dataset)}")
        logger.info(f"Output Directory : {savepath}")
        
    progress = tqdm(
        infer_loader,
        desc=f"{layer_name}",
        unit="image",
        dynamic_ncols=True,
    )
    
    for iter, (img_name, img_tensor) in enumerate(progress):
    # for iter, (img_name, img_tensor) in enumerate(infer_loader):
        img_name = img_name[0]
        img_path = os.path.join(dataroot, 'train5', os.path.basename(img_name) + '.png')
        orig_img = np.asarray(Image.open(img_path))

        # Get one heatmap per class from THIS layer
        cams = []
        for class_idx in range(n_class):
            cam = cam_extractor(img_tensor, class_idx)   # (224, 224), values in [0, 1]
            # print(f"[{layer_name}] class={class_idx} mean={cam.mean():.6f} std={cam.std():.6f} shape={cam.shape}")
            
            cams.append(cam)
        norm_cam = np.stack(cams, axis=0)   # shape (n_class, 224, 224)

        # ---- everything below is UNCHANGED from your original code ----
        label_str = img_name.split(']')[0].split('[')[-1]
        if dataset == 'luad':
            label = torch.Tensor([int(label_str[0]), int(label_str[2]), int(label_str[4]), int(label_str[6])])
        elif dataset == 'bcss':
            label = torch.Tensor([int(label_str[0]), int(label_str[1]), int(label_str[2]), int(label_str[3])])

        cam_dict = infer_utils.cam_npy_to_cam_dict(norm_cam, label)
        cam_score, bg_score = infer_utils.dict2npy(cam_dict, label, orig_img, None)

        if dataset == 'luad':
            bgcam_score = np.concatenate((cam_score, bg_score), axis=0)
        elif dataset == 'bcss':
            bg_score = np.zeros((1, 224, 224))
            bgcam_score = np.concatenate((cam_score, bg_score), axis=0)

        seg_map = infer_utils.cam_npy_to_label_map(bgcam_score)
        visualimg = Image.fromarray(seg_map.astype(np.uint8), "P")
        visualimg.putpalette(palette)
        visualimg.save(os.path.join(savepath, os.path.basename(img_name) + '.png'), format='PNG')
        
        progress.set_postfix(
            Image=os.path.basename(img_name),
        )
        
        if logger is not None and (iter + 1) % 100 == 0:
            logger.info(
                f"[{layer_name}] Processed {iter + 1}/{len(infer_dataset)} images"
            )

        # if iter % 100 == 0:
        #     print(iter)
        progress.close()

    if logger is not None:

        logger.info(
            f"Completed pseudo-mask generation ({layer_name})"
        )

        logger.info(
            f"Saved masks to : {savepath}"
        )

        logger.info("")