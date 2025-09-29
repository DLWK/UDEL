
import os
import random
import numpy as np
import cv2
from tqdm import tqdm
import torch
from skimage.measure import label, regionprops, find_contours
from sklearn.utils import shuffle
from metrics import precision, recall, F2, dice_score, jac_score
# from medpy.metric.binary import jc, dc, hd, hd95, recall, specificity, precision
from sklearn.metrics import accuracy_score, confusion_matrix
from medpy import metric
from torch import nn

def compute_ece(preds, targets, n_bins=10):
    """
    计算预期校准误差(ECE)
    Args:
        preds: 模型预测概率 [N, H, W] (经过sigmoid后的值)
        targets: 真实标签 [N, H, W] (值为0或1)
        n_bins: 分箱数量
    Returns:
        ece: 预期校准误差
        bin_accuracies: 各箱准确率
        bin_confidences: 各箱置信度
    """
    # 展平预测和标签
    preds_flat = preds.flatten()
    targets_flat = targets.flatten()
    preds_flat=torch.from_numpy(preds_flat)
    targets_flat=torch.from_numpy( targets_flat)

    
    # 确保数据类型正确
    preds_flat = preds_flat.float()
    targets_flat = targets_flat.float()
    
    # 分箱边界
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    bin_accuracies = []
    bin_confidences = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # 获取当前箱内的样本
        in_bin = (preds_flat >= bin_lower) & (preds_flat < bin_upper)
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin.item() > 0:
            # 计算箱内准确率和平均置信度
            accuracy_in_bin = targets_flat[in_bin].float().mean()
            avg_confidence_in_bin = preds_flat[in_bin].mean()
            
            bin_accuracies.append(accuracy_in_bin.item())
            bin_confidences.append(avg_confidence_in_bin.item())
            
            # 累加ECE
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        else:
            bin_accuracies.append(0)
            bin_confidences.append(0)
    
    return ece.item()



























def calculate_metric_percase(pred, gt):
    # pred[pred > 0.5] = 1
    # pred[pred != 1] = 0
    
    # gt[gt > 0] = 1

    pred = pred>0.5
    gt =  gt>0.5

    try:
        # dice = metric.binary.dc(pred, gt)
        # jc = metric.binary.jc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        # hd = metric.binary.hd(pred, gt) 
      
    except:
        hd95 = 0 
    return hd95

""" Seeding the randomness. """
def seeding(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

""" Create a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

""" Shuffle the dataset. """
def shuffling(x, y, l):
    x, y, l = shuffle(x, y, l, random_state=42)
    return x, y, l

def shuffling4(x, y, l,j):
    x, y, l,j = shuffle(x, y, l, j, random_state=42)
    return x, y, l,j


""" Shuffle the dataset. """
def shuffling1(x, y):
    x, y= shuffle(x, y, random_state=42)
    return x, y

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def print_and_save(file_path, data_str):
    print(data_str)
    with open(file_path, "a") as file:
        file.write(data_str)
        file.write("\n")

def label_dictionary():
    label_dict = {}
    label_dict["polyp"] = ["one", "multiple", "small", "medium", "large"]
    return label_dict

""" Convert a mask to border image """
def mask_to_border(mask):
    h, w = mask.shape
    border = np.zeros((h, w))

    contours = find_contours(mask, 128)
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y] = 255

    return border

""" Mask to bounding boxes """
def mask_to_bbox(mask):
    bboxes = []

    mask = mask_to_border(mask)
    lbl = label(mask)
    props = regionprops(lbl)
    for prop in props:
        x1 = prop.bbox[1]
        y1 = prop.bbox[0]

        x2 = prop.bbox[3]
        y2 = prop.bbox[2]

        bboxes.append([x1, y1, x2, y2])

    return bboxes

def calculate_metrics(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    y_pred = y_pred > 0.5
    y_pred = y_pred.reshape(-1)
    y_pred = y_pred.astype(np.uint8)

    y_true = y_true > 0.5
    y_true = y_true.reshape(-1)
    y_true = y_true.astype(np.uint8)

    ## Score
    score_jaccard = jac_score(y_true, y_pred)
    score_f1 = dice_score(y_true, y_pred)
    score_recall = recall(y_true, y_pred)
    score_precision = precision(y_true, y_pred)
    score_fbeta = F2(y_true, y_pred)
    score_hd95 = calculate_metric_percase(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)
    score_ECE = compute_ece(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc, score_fbeta,score_hd95,score_ECE]   
# 
