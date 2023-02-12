import os 
import datetime
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as trv
from PIL import Image
import random
import utils
import logging
import argparse

def main(opt):
    VOC_DIR = opt.data_folder
    BATCH_SIZE = opt.batch_size
    NUM_CLASSES = opt.num_classes
    NUM_EPOCHS = opt.epochs
    Learning_Rate = opt.lr
    Weight_Decay = opt.Weight_decay
    
    log_name = datetime.datetime.strftime(datetime.datetime.now(), "%m%d%H%M") + ".log"
    log_path = os.path.join("logs", log_name)
    logging.basicConfig(level=logging.DEBUG, filename=log_path, filemode='w', format="")
    weights_name = datetime.datetime.strftime(datetime.datetime.now(), "%m%d%H%M") + ".pth"
    save_path = os.path.join("weights", weights_name)
    
    if ('Darwin' in os.uname()) and ("arm64" in os.uname()):
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    Train_dataset = utils.VOC_dataset(VOC_DIR)
    Valid_dataset = utils.VOC_dataset(VOC_DIR, train=False)

    Train_loader = torch.utils.data.DataLoader(Train_dataset, BATCH_SIZE, shuffle=True, 
                                               drop_last=True, num_workers=4)
    Valid_loader = torch.utils.data.DataLoader(Valid_dataset, BATCH_SIZE, shuffle=False, 
                                               drop_last=True, num_workers=4)

    model = utils.ResNet18_FCN(NUM_CLASSES)

    def loss_fun(inputs, targets):
        return F.cross_entropy(inputs, targets, reduction="none").mean(1).mean(1)

    optimizer = torch.optim.SGD(model.parameters(), lr=Learning_Rate, weight_decay=Weight_Decay)

    model.to(device);

    for epoch in range(NUM_EPOCHS):

        print(f"Epoch: {epoch+1} ")
        logging.info(f"Epoch: {epoch+1} ")

        model.train()
        train_loss = 0.0
        sample_num = 0.0
        for inputs, labels in Train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = loss_fun(outputs, labels).sum()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            sample_num += labels.shape[0]

        train_loss = train_loss / sample_num
        print(f'train Loss: {train_loss:.4f}')
        logging.info(f'train Loss: {train_loss:.4f}')

        model.eval()
        valid_loss = 0.0
        sample_num = 0.0
        for inputs, labels in Valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = loss_fun(outputs, labels).sum()
                valid_loss += loss.item()
                sample_num += labels.shape[0]

        valid_loss = valid_loss / sample_num
        print(f'valid Loss: {valid_loss:.4f}')
        logging.info(f'valid Loss: {valid_loss:.4f}')
    
    model.to("cpu");
    torch.save(model.state_dict(), save_path)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default="dataset/VOCdevkit/VOC2012/", help="dataset path")
    parser.add_argument('--batch_size', type=int, default=16, help="batch size")
    parser.add_argument('--num_classes', type=int, default=21, help="number of all classes")
    parser.add_argument('--epochs', type=int, default=30, help='Interval between each validation')
    parser.add_argument('--saved_model', default='', help="epochs")
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--Weight_decay', type=float, default=1e-3, help='weight decay')
    
    opt = parser.parse_args()
    main(opt)
