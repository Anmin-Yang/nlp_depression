"""
Train Nerual Networks
Select model type by --model
"""

import torch
import numpy as np
import time
import copy
import os
import random
import pickle
os.environ['CUDA_VISIBLE_DEVICES']="1"
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from textcnn import TextCnn,TextCnn_BN
from dpcnn import DPCnn
from textrnn import TextRnn, TextRnn_att
import argparse
######### read data and model
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--model', default = 'textcnn', help='model type')
parser.add_argument('--fold_num', type = int, default = 0, help='model type')
parser.add_argument('--all_data', type = int, default = 0, help='model type')
parser.add_argument('--cutsupp', type = int, default = 0, help='model type')
args = parser.parse_args()
# f = open('max_words.txt', 'r')
# max_words = int(f.readlines()[0])
if(args.cutsupp == 0):
    if(args.all_data == 0):
        max_words = 244
    else:
        max_words = 305
else:
    if(args.all_data == 0):
        max_words = 113
    else:
        max_words = 102
if(args.model == 'textcnn'):
    print('Using TextCNN')
    model = TextCnn()
elif(args.model == 'textcnn_bn'):
    print('Using TextCNN_BN')
    model = TextCnn_BN()
elif(args.model == 'textrnn'):
    print('Using TextRNN')
    model = TextRnn(max_words = max_words)
elif(args.model == 'textrnn_att'):
    print('Using TextRNN_Attention')
    model = TextRnn_att()
elif(args.model == 'dpcnn'):
    print('Using DPCNN')
    model = DPCnn()
num_params = 0
for param in model.parameters():
    num_params += param.numel()
print('Model Params: ', num_params)

######### modify model
# truncate model to eliminate the embedding layer
# truncated_model = nn.Sequential(*list(model.children())[1:])
truncated_model = model

# read word vectors
data_path = '/nfs/s2/userhome/xuyitao/NLP/data/data/word2vec_matrix/'
if(args.cutsupp == 0):
    if(args.all_data == 0):
        print('Reading Junior Data...')
        normal_matrix = np.load(os.path.join(data_path,'normal_junior_shuffle.npy'),allow_pickle=True)
        depressed_matrix = np.load(os.path.join(data_path,'depressed_junior_shuffle.npy'),allow_pickle=True)
    else:
        print('Reading All Age Data...')
        normal_matrix = np.load(os.path.join(data_path,'normal_all_time_shuffle.npy'),allow_pickle=True)
        depressed_matrix = np.load(os.path.join(data_path,'depressed_all_time_shuffle.npy'),allow_pickle=True)
else:
    if(args.all_data == 0):
        print('Reading Junior Data with cut and supplement...')
        normal_matrix = np.load(os.path.join(data_path,'normal_junior_cutsupp_shuffle.npy'),allow_pickle=True)
        depressed_matrix = np.load(os.path.join(data_path,'depressed_junior_cutsupp_shuffle.npy'),allow_pickle=True)
    else:
        print('Reading All Age Data with cut and supplement...')
        normal_matrix = np.load(os.path.join(data_path,'normal_all_time_cutsupp_shuffle.npy'),allow_pickle=True)
        depressed_matrix = np.load(os.path.join(data_path,'depressed_all_time_cutsupp_shuffle.npy'),allow_pickle=True)
 # 3-D structure for word matrix, (word_length, feature_length, num of texts)


# match data size
size_to_match = min(depressed_matrix.shape[2],normal_matrix.shape[2])
size_to_match = int(size_to_match * 1.5)
normal_matrix_match = normal_matrix.transpose()
# print(normal_matrix_match.shape)
# exit()
# np.random.shuffle(normal_matrix_match)
normal_matrix_matched = normal_matrix_match[:size_to_match,:,:].transpose()
normal_matrix_left = normal_matrix_match[size_to_match:,:,:].transpose()
print('Original Normal Matrix Shape: ', normal_matrix.shape)
print('Original Depressed Matrix Shape: ', depressed_matrix.shape)
print('Cut Normal Matrix Shape: ', normal_matrix_matched.shape)
print('Left Normal Matrix Shape: ', normal_matrix_left.shape)
print('=======================================================')

# split data
all_idx_normal = list(range(normal_matrix_matched.shape[2]))
one_fold = round(normal_matrix_matched.shape[2]*0.2)
valid_idx_normal_start = one_fold * args.fold_num
valid_idx_normal_end = valid_idx_normal_start + one_fold
if(valid_idx_normal_end >= normal_matrix_matched.shape[2]):
    valid_idx_normal_end = normal_matrix_matched.shape[2]
valid_idx_normal = all_idx_normal[valid_idx_normal_start : valid_idx_normal_end]
train_idx_normal = [x for x in all_idx_normal if x not in valid_idx_normal]
print('Normal Sample Train number: ', len(train_idx_normal))
print('Normal Sample Val number: ', len(valid_idx_normal))

all_idx_depress = list(range(depressed_matrix.shape[2]))
one_fold = round(depressed_matrix.shape[2]*0.2)
valid_idx_depress_start = one_fold * args.fold_num
valid_idx_depress_end = valid_idx_depress_start + one_fold
if(valid_idx_depress_end >= depressed_matrix.shape[2]):
    valid_idx_depress_end = depressed_matrix.shape[2]
valid_idx_depress = all_idx_depress[valid_idx_depress_start : valid_idx_depress_end]
train_idx_depress = [x for x in all_idx_depress if x not in valid_idx_depress]
print('Depress Sample Train number: ', len(train_idx_depress))
print('Depress Sample Val number: ', len(valid_idx_depress))

if(len(valid_idx_normal) > len(valid_idx_depress)):
    valid_idx_normal = valid_idx_normal[:len(valid_idx_depress)]
    print('Cut Normal Sample Val Set to ', len(valid_idx_normal))
print('Building Dataset...')

## prepare dataset
train_dataset = []
for i in train_idx_depress:
    temp_data = depressed_matrix[:,:,i]
    temp_data = temp_data.astype('float64')
    temp_data = torch.from_numpy(temp_data)
    train_dataset.append((temp_data,1))

for i in train_idx_normal:
    temp_data = normal_matrix_matched[:,:,i]
    temp_data = temp_data.astype('float64')
    temp_data = torch.from_numpy(temp_data)
    train_dataset.append((temp_data,0))

val_dataset = []
for i in valid_idx_depress:
    temp_data = depressed_matrix[:,:,i]
    temp_data = temp_data.astype('float64')
    temp_data = torch.from_numpy(temp_data)
    val_dataset.append((temp_data,1))

for i in valid_idx_normal:
    temp_data = normal_matrix_matched[:,:,i]
    temp_data = temp_data.astype('float64')
    temp_data = torch.from_numpy(temp_data)
    val_dataset.append((temp_data,0))

print('Train set length: ', len(train_dataset))
print('Val set length: ', len(val_dataset))
train_positive_sample_num = sum([x[1] for x in train_dataset])
train_negetive_sample_num = len(train_dataset) - train_positive_sample_num
val_positive_sample_num = sum([x[1] for x in val_dataset])
val_negative_sample_num = len(val_dataset) - val_positive_sample_num
print('Depress sample number in Train set: ', train_positive_sample_num)
print('Normal sample number in Train set: ', train_negetive_sample_num)
print('Depress sample number in Val set: ', val_positive_sample_num)
print('Normal sample number in Val set: ', val_negative_sample_num)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True, drop_last = True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 10, shuffle = False)

######### rightness
def rightness(predictions, labels):
    pred = torch.max(predictions.data, 1)[1]  
    rights = pred.eq(labels.data.view_as(pred)).sum()  
    return rights, len(labels)  


########## train network
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(truncated_model.parameters(), lr = 0.01, momentum=0.9, weight_decay = 5e-4)
optimizer = optim.AdamW(truncated_model.parameters(), lr = 0.005, weight_decay = 1e-4)
num_epochs = 60
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [num_epochs // 3, num_epochs // 3 * 2], gamma=0.1)
record = []


truncated_model.train(True)
truncated_model = truncated_model.cuda()
best_model = truncated_model
best_r = 0.0
best_acc = 0.0
train_loss = []
val_loss = []
softmax = nn.Softmax()
for epoch in range(num_epochs):
    truncated_model.train()
    #optimizer = exp_lr_scheduler(optimizer, epoch)
    train_rights = []  
    train_losses = []
    for batch_idx, (data_big, target) in enumerate(train_loader):   
        if(epoch == 0 and batch_idx == 0):
            print('Sample Data ID: ', int(data_big[0,0,-1]))
            print('Sample Data Shape and Label Shape: ', data_big.shape, target.shape)
        # exit()
        data = data_big[:,:,:-1].cuda().float()
        target = target.cuda().long()  
        # print(type(data))
        output = truncated_model(data) 
        # print(data.shape, output.shape, target.shape)
        loss = criterion(output, target)  
        optimizer.zero_grad()  
        loss.backward() 
        optimizer.step()  
        right = rightness(output, target)  
        train_rights.append(right)  

        train_losses.append(loss.data.cpu().numpy())

    train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))

     
    truncated_model.eval()  
    test_loss = 0
    correct = 0
    vals = []
    conf_dict_normal = {}
    conf_dict_depress = {}
    conf_dict_normal_false = {}
    conf_dict_depress_false = {}
     
    for data_big, target in val_loader:
        data = data_big[:,:,:-1].cuda().float()
        target = target.cuda().long()
        output = truncated_model(data)  
        for i in range(data.size(0)):
            text_id = int(data_big[i, 0, -1])
            output_softmax = softmax(output[i].cpu())
            # pred = torch.max(output_softmax)
            # target_number = target.item()
            if(output_softmax[0] > output_softmax[1] and target[i] == 0):
                conf_dict_normal[text_id] = output_softmax[0].item()
            elif(output_softmax[0] < output_softmax[1] and target[i] == 1):
                conf_dict_depress[text_id] = output_softmax[1].item()

            if(output_softmax[0] < output_softmax[1] and target[i] == 0):
                conf_dict_normal_false[text_id] = output_softmax[1].item()
            elif(output_softmax[0] > output_softmax[1] and target[i] == 1):
                conf_dict_depress_false[text_id] = output_softmax[0].item()

        val = rightness(output, target)  
        vals.append(val)  

     
    val_r = (sum([tup[0] for tup in vals]), sum([tup[1] for tup in vals]))
    val_ratio = 1.0*val_r[0]/val_r[1]

    if val_ratio > best_r:
        best_r = val_ratio
        best_model = copy.deepcopy(truncated_model)
     
    now_val_acc = 100. * val_r[0].cpu().numpy()/val_r[1]
    if(now_val_acc > best_acc):
        best_acc = now_val_acc
        normal_conf_sorted = sorted(conf_dict_normal.items(), key = lambda x:x[1], reverse = True)
        depress_conf_sorted = sorted(conf_dict_depress.items(), key = lambda x:x[1], reverse = True)
        normal_conf_false_sorted = sorted(conf_dict_normal_false.items(), key = lambda x:x[1], reverse = True)
        depress_conf_false_sorted = sorted(conf_dict_depress_false.items(), key = lambda x:x[1], reverse = True)
        conf_dict = {}
        conf_dict['normal'] = normal_conf_sorted[:20]
        conf_dict['depress'] = depress_conf_sorted[:20]
        conf_dict['normal_false'] = normal_conf_false_sorted[:20]
        conf_dict['depress_false'] = depress_conf_false_sorted[:20]
        # print('Conf Dict: ', conf_dict)
        if(args.cutsupp == 0):
            if(args.all_data == 0):
                with open(f'result/{args.model}_{args.fold_num}.txt', 'w') as f:
                    f.write(str(now_val_acc) + '\t' + f'{epoch}')
                with open(f'conf_record/{args.model}_{args.fold_num}.pkl', 'wb') as f:
                    pickle.dump(conf_dict, f, pickle.HIGHEST_PROTOCOL)
            else:
                with open(f'result/{args.model}_{args.fold_num}_all_data.txt', 'w') as f:
                    f.write(str(now_val_acc) + '\t' + f'{epoch}')
                with open(f'conf_record/{args.model}_{args.fold_num}_all_data.pkl', 'wb') as f:
                    pickle.dump(conf_dict, f, pickle.HIGHEST_PROTOCOL)
        else:
            if(args.all_data == 0):
                with open(f'result/{args.model}_{args.fold_num}_cutsupp.txt', 'w') as f:
                    f.write(str(now_val_acc) + '\t' + f'{epoch}')
                with open(f'conf_record/{args.model}_{args.fold_num}_cutsupp.pkl', 'wb') as f:
                    pickle.dump(conf_dict, f, pickle.HIGHEST_PROTOCOL)
            else:
                with open(f'result/{args.model}_{args.fold_num}_cutsupp_all_data.txt', 'w') as f:
                    f.write(str(now_val_acc) + '\t' + f'{epoch}')
                with open(f'conf_record/{args.model}_{args.fold_num}_cutsupp_all_data.pkl', 'wb') as f:
                    pickle.dump(conf_dict, f, pickle.HIGHEST_PROTOCOL)
    if(epoch % 1 == 0):
        print('Epoch: {} \tLoss: {:.6f}\tTrain Acc: {:.2f}%, Val Acc: {:.2f}%'.format(
            epoch, np.mean(train_losses), 100. * train_r[0].cpu().numpy() / train_r[1], 100. * val_r[0].cpu().numpy()/val_r[1]))
    record.append([np.mean(train_losses), train_r[0].cpu().numpy() / train_r[1], val_r[0].cpu().numpy()/val_r[1]])
    scheduler.step()

# print results of model training
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
plt.rcParams['savefig.dpi'] = 300 
plt.rcParams['figure.dpi'] = 300 
train_acc_all = [x[1] for x in record]
train_idx = list(range(len(train_acc_all)))
val_acc_all = [x[2] for x in record]
val_idx = list(range(len(val_acc_all)))

if(args.all_data == 0):
    plt.plot(train_idx, train_acc_all)
    plt.savefig(f'result_pic/{args.model}_{args.fold_num}_train.png')
    plt.clf()
    plt.plot(val_idx, val_acc_all)
    plt.savefig(f'result_pic/{args.model}_{args.fold_num}_val.png')
else:
    plt.plot(train_idx, train_acc_all)
    plt.savefig(f'result_pic/{args.model}_{args.fold_num}_train_all_data.png')
    plt.clf()
    plt.plot(val_idx, val_acc_all)
    plt.savefig(f'result_pic/{args.model}_{args.fold_num}_val_all_data.png')
