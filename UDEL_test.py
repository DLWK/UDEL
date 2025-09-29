
import os, time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import imageio
import torch
from model import TGAPolypSeg
from utils import create_dir, seeding
from utils import calculate_metrics
from trainfree import load_data
from text2embed import Text2Embed
####

from models.unet import UNet
from models.UNet_2Plus import  UNet_2Plus
from models.BaseNet import CPFNet 
from  MSNet.msnet import  MSNet
from UACAmodel.lib.UACANet import  UACANet

####
from EAGmodels.EAGmodel import EAGNet
from PraNet.PraNet_Res2Net import PraNet
from  models.UDELNet.ResNet_models import  Generator, FCDiscriminator
def process_mask(y_pred):
    y_pred = y_pred[0].cpu().numpy()
    y_pred = np.squeeze(y_pred, axis=0)
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.int32)
    y_pred = y_pred * 255
    y_pred = np.array(y_pred, dtype=np.uint8)
    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=2)
    return y_pred

def print_score(metrics_score):
    jaccard = metrics_score[0]/len(test_x)
    f1 = metrics_score[1]/len(test_x)
    recall = metrics_score[2]/len(test_x)
    precision = metrics_score[3]/len(test_x)
    acc = metrics_score[4]/len(test_x)
    f2 = metrics_score[5]/len(test_x)

    print(f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f} - F2: {f2:1.4f}")

def evaluate(model, save_path, test_x, test_y, size):
    metrics_score_1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []
   

    for i, (x, y,z) in tqdm(enumerate(zip(test_x, test_y,test_z)), total=len(test_x)):
        name = y.split("/")[-1].split(".")[0]

        """ Image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, size)
        save_img = image
        image = np.transpose(image, (2, 0, 1))
        image = image/255.0
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        image = image.to(device)


        """ Mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, size)
        save_mask = mask
       
        save_mask = np.expand_dims(save_mask, axis=-1)
     
        save_mask = np.concatenate([save_mask, save_mask, save_mask], axis=2)
        mask = np.expand_dims(mask, axis=0)
        mask = mask/255.0
        mask = np.expand_dims(mask, axis=0)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)
        mask = mask.to(device)

        with torch.no_grad():
            """ FPS calculation """
            start_time = time.time()
            p1, p= model(image)

            p = torch.sigmoid(p)

             ######概率图保存######
            p2 = torch.sigmoid(p)
            p2 = np.array(p2.data.cpu()[0])[0]
            p2 = p2*255
           ######################
        
            end_time = time.time() - start_time
            time_taken.append(end_time)

            """ Evaluation metrics """
            score_1 = calculate_metrics(mask, p)
            metrics_score_1 = list(map(add, metrics_score_1, score_1))
           
            p = process_mask(p)
           
           

        """ Save the image - mask - pred """
        line = np.ones((size[0], 10, 3)) * 255
        cat_images = np.concatenate([save_img, line, save_mask, line, p], axis=1)

        save_image_name = f"{name}"
        
       
        
        # p2 = np.argmax(p2)
        # p3 = np.argmax(p3)
        #
        # if p2 == 0:
        #     save_image_name += "-one_polyp"
        # else:
        #     save_image_name += "-multiple_polyp"
        #
        # if p3 == 0: save_image_name += "-small"
        # if p3 == 1: save_image_name += "-medium"
        # if p3 == 2: save_image_name += "-large"
        cv2.imwrite(f"{save_path}/image/{name}.jpg", save_img)
        cv2.imwrite(f"{save_path}/all/{name}.jpg", cat_images)
        cv2.imwrite(f"{save_path}/mask/{name}.jpg", save_mask)
        cv2.imwrite(f"{save_path}/pred/{name}.jpg", p)
        cv2.imwrite(f"{save_path}/prop/{name}.jpg", p2)

    print_score(metrics_score_1)
   
    mean_time_taken = np.mean(time_taken)
    mean_fps = 1/mean_time_taken
    print("Mean FPS: ", mean_fps)






if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = PraNet()
    # model =UNet(n_channels=3, n_classes=1)
    # model = UNet_2Plus(in_channels=3, n_classes=1)
    # model = CPFNet()
    # model = MSNet()
    # model =  UACANet()
    model = Generator(32).to(device)
    model = model.to(device)
    checkpoint_path = "weight/Kvasir-SEG/UDELNet/checkpoint.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    """ Test dataset """
    path = "sessile"
    # path ="CVC-ClinicDB/CVC-ClinicDB"
    # path = "BKAI"
    path = "Kvasir-SEG"
    # path = "ISIC2018"
    (train_x, train_y,train_z), (valid_x, valid_y,valid_z), (test_x, test_y,test_z) = load_data(path)
   
    # embed = Text2Embed()
    save_path = f"test_result/Kvasir-SEG/UDELNet"
    # save_path = f"test_result/sessile/UACANet"
    size = (256, 256)
    create_dir(f"{save_path}/image")
    create_dir(f"{save_path}/all")
    create_dir(f"{save_path}/mask")
    create_dir(f"{save_path}/pred")
    create_dir(f"{save_path}/prop")
    evaluate(model, save_path, test_x, test_y, size)
