import os
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from torch.util_functions.data import DataLoader
from accelerate import Accelerator
from itertools import groupby

import configs
import util_functions
from data_convert import HRDataset
from model_setup import CRNN


def train_one_epoch(loader, model, optimizer, criterion, device, phase):
    accelerator = Accelerator(mixed_precision="fp16")
    if phase == 'train':
        model.train()
    else:
        model.eval()
    #General model training code
    loop = tqdm(loader)
    total_loss = 0
    correct = 0
    total = 0

    if phase == 'train':
            for batch_idx, (inputs, labels) in enumerate(loop):
                batch_size = inputs.shape[0]
                inputs = inputs.to(device)

                y_pred = model(inputs)
                y_pred = y_pred.permute(1, 0, 2)

                input_lengths = torch.IntTensor(batch_size).fill_(37)
                target_lengths = torch.IntTensor([len(t) for t in labels])

                loss = criterion(y_pred.cpu(), labels,
                                input_lengths, target_lengths)
                total_loss += loss.detach().numpy()

                _, max_index = torch.max(y_pred.cpu(), dim=2)

                #CRNN specific training error measurement 
                for i in range(batch_size):
                    #Get most probable words as integers
                    pred_raw = list(max_index[:, i].numpy())
                    #Convert predicted raw predictions and observed labels into tensors
                    prediction = torch.IntTensor(
                        [c for c, _ in groupby(pred_raw) if c != 0])
                    real = torch.IntTensor(
                        [c for c, _ in groupby(labels[i]) if c != 0])
                    #Compare
                    if len(prediction) == len(real) and torch.all(prediction.eq(real)):
                        correct += 1
                    total += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            ratio = correct / total
    if phase == 'valid':
        if total_loss < best_vloss:
            best_vloss = total_loss
            model_path = 'best_model.ph'
            print(f'New best model found. Saving it as {model_path}. Loss={best_vloss:.3f}')
            unwrapped_model = accelerator.unwrap_model(model)
            accelerator.save(unwrapped_model.state_dict(), model_path)
    print('TEST correct: ', correct, '/', total, ' P:', ratio)
    print("Avg CTC loss:", total_loss/batch_idx)


def main():
    train_data = util_functions.get_dataset(configs.train_csv)
    valid_data = util_functions.get_dataset(configs.valid_csv)
    test_data = util_functions.get_dataset(configs.test_csv)

    train_dataset = HRDataset(train_data, util_functions.encode, mode='train')
    valid_dataset = HRDataset(valid_data, util_functions.encode, mode='valid')
    test_dataset = HRDataset(test_data, util_functions.encode, mode='test')

    train_loader = DataLoader(
        train_dataset, batch_size=configs.batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(
        valid_dataset, batch_size=configs.batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(
        test_dataset, batch_size=configs.batch_size, shuffle=False, pin_memory=True)

    input_size = 64
    hidden_size = 256
    output_size = configs.vocab_size + 1
    num_layers = 2
    accelerator = Accelerator(mixed_precision="fp16")

    model = CRNN(input_size, hidden_size, output_size, num_layers)

    model.to(configs.device)

    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=configs.learning_rate)

    best_vloss = 1000000

    for epoch in range(configs.NUM_EPOCHS):
        #Training
        train_one_epoch(train_loader, model, optimizer, criterion, configs.device, phase = "train")
        #Evaluation
        train_one_epoch(valid_loader, model, optimizer, criterion, configs.device, phase = "valid")



if __name__ == "__main__":
    main()