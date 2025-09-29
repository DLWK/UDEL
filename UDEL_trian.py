

import random
import time
import datetime
from turtle import shape
import numpy as np
import albumentations as A
import cv2
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from text2embed import Text2Embed
from utils import seeding, create_dir, print_and_save, shuffling1, epoch_time, calculate_metrics, label_dictionary, mask_to_bbox
# from model import TGAPolypSeg
####

from models.unet import UNet
from models.UNet_2Plus import  UNet_2Plus
from models.cenet import CE_Net
from models.BaseNet import CPFNet 
from  MSNet.msnet import  MSNet
from UACAmodel.lib.UACANet import  UACANet

from  models.UDELNet.ResNet_models import  Generator, FCDiscriminator
from  models.UDELNet.VMUNet import VMUNet
####
from metrics import DiceLoss, DiceBCELoss, MultiClassBCE
from sklearn.model_selection import train_test_split
from PraNet.PraNet_Res2Net import PraNet
import os

from loss import make_confidence_label, uncertainty_aware_structure_loss
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# def load_names(path, file_path):
#     f = open(file_path, "r")
#     data = f.read().split("\n")[:-1]
#     images = [os.path.join(path,"images", name) + ".jpg" for name in data]
#     masks = [os.path.join(path,"masks", name) + ".jpg" for name in data]
#     return images, masks

# def load_data(path):
#     train_names_path = f"{path}/train.txt"
#     valid_names_path = f"{path}/val.txt"

#     train_x, train_y = load_names(path, train_names_path)
#     valid_x, valid_y = load_names(path, valid_names_path)

#     label_dict = label_dictionary()
#     train_label = len(train_x) * [label_dict["polyp"]]
#     valid_label = len(valid_x) * [label_dict["polyp"]]

#     return (train_x, train_y, train_label), (valid_x, valid_y, valid_label)

def load_data(path, split=0.1):
    images = sorted(glob(os.path.join(path, "images/*")))
    masks = sorted(glob(os.path.join(path, "masks/*")))
    edges = sorted(glob(os.path.join(path, "point/*")))

    total_size = len(images)
    valid_size = int(split * total_size)
    test_size = int(split * total_size)
  
   

    train_x, valid_x = train_test_split(images, test_size=valid_size, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=valid_size, random_state=42)
   
    train_x, test_x = train_test_split(train_x, test_size=test_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=test_size, random_state=42)
  
   
    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)




class DATASET(Dataset):
    def __init__(self, images_path,  masks_path, size, transform=None):
        super().__init__()

        self.images_path = images_path
     
        self.masks_path = masks_path
        self.transform = transform
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        """ Reading Image & Mask """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)

        """ Applying Data Augmentation """
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        """ Image """
        image = cv2.resize(image, size)
        image = np.transpose(image, (2, 0, 1))
        image = image/255.0

        """ Mask """
        mask = cv2.resize(mask, size)
        mask_copy = mask
        mask = np.expand_dims(mask, axis=0)
        mask = mask/255.0

        # """ Mask to Textual information """
        # num_polyps, polyp_sizes = self.mask_to_text(mask_copy)

        # """ Label """
        # label = []
        # words = self.labels_path[index]
        # for word in words:
        #     word_embed = self.embed.to_embed(word)[0]
        #     label.append(word_embed)
        # label = np.array(label)

        return image ,mask

    def __len__(self):
        return self.n_samples

def train(model1,model2, loader, optimizer1, optimizer2,loss_fn, device):
    model1.train()
    model2.train()

    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    for i, (x,y) in enumerate(loader):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)
       
     

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        # p0,p1,p2,p = model(x)
        # Obtain initial and reference predictions from the generator
        init_pred, ref_pred = model1(x)
       
        # Pass the generator predictions and ground truth through discriminator
        post_init = torch.sigmoid(init_pred.detach())
        post_ref = torch.sigmoid(ref_pred.detach())

        confi_init_label = make_confidence_label(gts=y, pred=post_init)
        confi_ref_label = make_confidence_label(gts=y, pred=post_ref)

        # Concatenate image with prediction as input 
        post_init = torch.cat((post_init, x), dim=1)
        post_ref = torch.cat((post_ref, x), dim=1)

        # Predict the confidence map
        confi_init_pred = model2.forward(post_init)
        confi_ref_pred = model2.forward(post_ref)

    # Compute cross-entropy loss 
        confi_loss_pred_init = CE(torch.sigmoid(confi_init_pred), confi_init_label)
        confi_loss_pred_ref = CE(torch.sigmoid(confi_ref_pred), confi_ref_label)
        OCE_loss = 0.5 * (confi_loss_pred_init + confi_loss_pred_ref)

        # Backpropagate the loss through UDELNet
        OCE_loss.backward()
        optimizer2.step()

       

        # Compute structure loss 
        struct_loss1 = uncertainty_aware_structure_loss(pred=init_pred, mask=y, confi_map=confi_init_pred.detach(), epoch=epoch)
        struct_loss2 = uncertainty_aware_structure_loss(pred=ref_pred, mask=y, confi_map=confi_ref_pred.detach(), epoch=epoch)
        COD_loss = 0.5 * (struct_loss1 + struct_loss2)

        # Backpropagate loss 
        COD_loss.backward()
        optimizer1.step()

        loss=OCE_loss+COD_loss

        # loss = loss_fn[0](p0, y)+loss_fn[0](p1, y)+loss_fn[0](p2, y)+loss_fn[0](p, y)
        
        epoch_loss += loss.item()

        """ Calculate the metrics """
        batch_jac = []
        batch_f1 = []
        batch_recall = []
        batch_precision = []

        for yt, yp in zip(y, ref_pred):
            score = calculate_metrics(yt, yp)
            batch_jac.append(score[0])
            batch_f1.append(score[1])
            batch_recall.append(score[2])
            batch_precision.append(score[3])

        epoch_jac += np.mean(batch_jac)
        epoch_f1 += np.mean(batch_f1)
        epoch_recall += np.mean(batch_recall)
        epoch_precision += np.mean(batch_precision)

    epoch_loss = epoch_loss/len(loader)
    epoch_jac = epoch_jac/len(loader)
    epoch_f1 = epoch_f1/len(loader)
    epoch_recall = epoch_recall/len(loader)
    epoch_precision = epoch_precision/len(loader)

    return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision]

def evaluate(model1,model2, loader, optimizer1, optimizer2,loss_fn, device):
    model1.eval()
    model2.eval()

    epoch_loss = 0
    epoch_loss = 0.0
    epoch_jac = 0.0
    epoch_f1 = 0.0
    epoch_recall = 0.0
    epoch_precision = 0.0

    with torch.no_grad():
        for i, (x,y) in enumerate(loader):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
           
            # p0,p1,p2,p= model(x)
            # loss = loss_fn[0](p0, y)+loss_fn[0](p1, y)+loss_fn[0](p2, y)+loss_fn[0](p, y)
            # 
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            # p0,p1,p2,p = model(x)
            # Obtain initial and reference predictions from the generator
            init_pred, ref_pred = model1(x)

            # Pass the generator predictions and ground truth through discriminator
            post_init = torch.sigmoid(init_pred.detach())
            post_ref = torch.sigmoid(ref_pred.detach())

            confi_init_label = make_confidence_label(gts=y, pred=post_init)
            confi_ref_label = make_confidence_label(gts=y, pred=post_ref)

            # Concatenate image with prediction as input 
            post_init = torch.cat((post_init, x), dim=1)
            post_ref = torch.cat((post_ref, x), dim=1)

            # Predict the confidence map
            confi_init_pred = model2.forward(post_init)
            confi_ref_pred = model2.forward(post_ref)

        # Compute cross-entropy loss 
            confi_loss_pred_init = CE(torch.sigmoid(confi_init_pred), confi_init_label)
            confi_loss_pred_ref = CE(torch.sigmoid(confi_ref_pred), confi_ref_label)
            OCE_loss = 0.5 * (confi_loss_pred_init + confi_loss_pred_ref)

           

        

            # Compute structure loss 
            struct_loss1 = uncertainty_aware_structure_loss(pred=init_pred, mask=y, confi_map=confi_init_pred.detach(), epoch=epoch)
            struct_loss2 = uncertainty_aware_structure_loss(pred=ref_pred, mask=y, confi_map=confi_ref_pred.detach(), epoch=epoch)
            COD_loss = 0.5 * (struct_loss1 + struct_loss2)

           

            loss=OCE_loss+COD_loss
            epoch_loss += loss.item()

            """ Calculate the metrics """
            batch_jac = []
            batch_f1 = []
            batch_recall = []
            batch_precision = []

            for yt, yp in zip(y, ref_pred):
                score = calculate_metrics(yt, yp)
                batch_jac.append(score[0])
                batch_f1.append(score[1])
                batch_recall.append(score[2])
                batch_precision.append(score[3])

            epoch_jac += np.mean(batch_jac)
            epoch_f1 += np.mean(batch_f1)
            epoch_recall += np.mean(batch_recall)
            epoch_precision += np.mean(batch_precision)

        epoch_loss = epoch_loss/len(loader)
        epoch_jac = epoch_jac/len(loader)
        epoch_f1 = epoch_f1/len(loader)
        epoch_recall = epoch_recall/len(loader)
        epoch_precision = epoch_precision/len(loader)

        return epoch_loss, [epoch_jac, epoch_f1, epoch_recall, epoch_precision]

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Directories """
    create_dir("psemyfiles")

    """ Training logfile """
    train_log_path = "psemyfiles/train_log.txt"
    if os.path.exists(train_log_path):
        print("Log file exists")
    else:
        train_log = open("psemyfiles/train_log.txt", "w")
        train_log.write("\n")
        train_log.close()

    """ Record Date & Time """
    datetime_object = str(datetime.datetime.now())
    print_and_save(train_log_path, datetime_object)
    print("")

    """ Hyperparameters """
    image_size = 256
    size = (image_size, image_size)
    batch_size = 8
    num_epochs = 500
    lr = 1e-4
    early_stopping_patience = 50
    checkpoint_path = "weight/CVC-ClinicDB/UDELNet/checkpoint.pth"
    path = "CVC-ClinicDB/CVC-ClinicDB"
    # path = "BKAI"
    # path = "Kvasir-SEG"
    # path = "sessile"

    data_str = f"Image Size: {size}\nBatch Size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\n"
    data_str += f"Early Stopping Patience: {early_stopping_patience}\n"
    print_and_save(train_log_path, data_str)

    """ Data augmentation: Transforms """
    transform =  A.Compose([
        A.Rotate(limit=35, p=0.3),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32)
    ])

    """ Dataset """
    (train_x, train_y), (valid_x, valid_y),(test_x, test_y) = load_data(path)
    train_x, train_y = shuffling1(train_x, train_y)
    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}- test: {len(test_x)}\n"
    print_and_save(train_log_path, data_str)

    """ Dataset and loader """
    train_dataset = DATASET(train_x,  train_y, (image_size, image_size), transform=transform)
    valid_dataset = DATASET(valid_x,  valid_y, (image_size, image_size), transform=None)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    """ Model """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = PraNet()
    # model = UNet(n_channels=3, n_classes=1)
    # model = UNet_2Plus(in_channels=3, n_classes=1)
    # model = CE_Net(num_classes=1, num_channels=3)
    # model = CPFNet()
    # model = MSNet()
    # model =  UACANet()
    model1 = Generator(32).to(device)
    # model2 = FCDiscriminator().to(device)
    model2 = VMUNet(4,1).to(device)

    optimizer1 = torch.optim.Adam(model1.parameters(), lr=lr)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=lr)

    scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1, 'min', patience=5, verbose=True)
    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer2, 'min', patience=5, verbose=True)
    loss_fn = [DiceBCELoss(), nn.BCEWithLogitsLoss(), nn.CrossEntropyLoss()]
    CE = torch.nn.BCELoss()
    loss_name = "BCE Dice Loss"

    data_str = f"Optimizer: Adam\nLoss: {loss_name}\n"
    print_and_save(train_log_path, data_str)

    """ Training the model """
    best_valid_metrics = 0.0
    early_stopping_count = 0

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, train_metrics = train(model1, model2,train_loader, optimizer1,optimizer2, loss_fn, device)
        valid_loss, valid_metrics = evaluate(model1,model2, valid_loader,optimizer1,optimizer2, loss_fn, device)
        scheduler2.step(valid_loss)

        if valid_metrics[1] > best_valid_metrics:
            data_str = f"Valid F1 improved from {best_valid_metrics:2.4f} to {valid_metrics[1]:2.4f}. Saving checkpoint: {checkpoint_path}"
            print_and_save(train_log_path, data_str)

            best_valid_metrics = valid_metrics[1]
            torch.save(model1.state_dict(), checkpoint_path)
            early_stopping_count = 0

        elif valid_metrics[1] < best_valid_metrics:
            early_stopping_count += 1

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n"
        data_str += f"\tTrain Loss: {train_loss:.4f} - Jaccard: {train_metrics[0]:.4f} - F1: {train_metrics[1]:.4f} - Recall: {train_metrics[2]:.4f} - Precision: {train_metrics[3]:.4f}\n"
        data_str += f"\t Val. Loss: {valid_loss:.4f} - Jaccard: {valid_metrics[0]:.4f} - F1: {valid_metrics[1]:.4f} - Recall: {valid_metrics[2]:.4f} - Precision: {valid_metrics[3]:.4f}\n"
        print_and_save(train_log_path, data_str)

        if early_stopping_count == early_stopping_patience:
            data_str = f"Early stopping: validation loss stops improving from last {early_stopping_patience} continously.\n"
            print_and_save(train_log_path, data_str)
            break
