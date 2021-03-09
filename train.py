from __future__ import print_function, division
import torch
import matplotlib as mpl
mpl.use('TkAgg')
import argparse,os
import pandas as pd
import cv2
import numpy as np
import random
import math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pdb import set_trace

from model.model import AlexNet, MaximumMeanDiscrepancy, SimilarityEmbedding 


from Loadtemporal_BinaryMask_train_3modality import Spoofing_train, Normaliztion, Resize, CenterCrop, ToTensor, RandomHorizontalFlip, Cutout, RandomErasing
from Loadtemporal_valtest_3modality import Spoofing_valtest, Resize_val, CenterCrop_val, Normaliztion_valtest, ToTensor_valtest


import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import copy
import pdb

from utils import AvgrageMeter, accuracy, performances



# Dataset root
image_dir = '/home/Disk1T/hxy/CASIA-CeFA/CASIA-CeFA/phase1/'         

train_list = '/home/Disk1T/hxy/CASIA-CeFA/CASIA-CeFA/phase1/4@1_train.txt'
val_list = '/home/Disk1T/hxy/CASIA-CeFA/CASIA-CeFA/phase1/4@1_dev_res.txt'

   
# train_list = '/wrk/yuzitong/DONOTREMOVE/CVPRW2020/4@1_train.txt'
# val_list = '/wrk/yuzitong/DONOTREMOVE/CVPRW2020/4@1_dev_res.txt'

#train_list = '/wrk/yuzitong/DONOTREMOVE/CVPRW2020/4@2_train.txt'
#val_list = '/wrk/yuzitong/DONOTREMOVE/CVPRW2020/4@2_dev_res.txt'

#train_list = '/wrk/yuzitong/DONOTREMOVE/CVPRW2020/4@3_train.txt'
#val_list = '/wrk/yuzitong/DONOTREMOVE/CVPRW2020/4@3_dev_res.txt'



# main function
def train_parent():
    # GPU  & log file  -->   if use DataParallel, please comment this command
    #os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % (args.gpu)
    save_epoch = 2
    isExists = os.path.exists(args.log)
    if not isExists:
        os.makedirs(args.log)
    log_file = open(args.log+'/'+ args.log+'_log.txt', 'w')
    
    echo_batches = args.echo_batches

    print("Oulu-NPU, P1:\n ")

    log_file.write('Oulu-NPU, P1:\n ')
    log_file.flush()

    # load the network, load the pre-trained model in UCF101?
    finetune = args.finetune
    if finetune==True:
        print('finetune!\n')

    else:
        print('train from scratch!\n')
        log_file.write('train from scratch!\n')
        log_file.flush()
         
		 
        #model = CDCN_3modality2( basic_conv=Conv2d_cd, theta=0.7)
		# model = CDCN_3modality2( basic_conv=Conv2d_cd, theta=args.theta)
        model = AlexNet()

        model = model.cuda()


        lr = args.lr
        optimizer = optim.SGD(model.parameters(), lr=0.0001, weight_decay=0.0005,momentum=0.9)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    
    print(model) 
    
    
    # criterion_absolute_loss = nn.MSELoss().cuda()
    # criterion_contrastive_loss = Contrast_depth_loss().cuda() 
    CEloss = nn.CrossEntropyLoss().cuda()


    ACER_save = 1.0
    
    for epoch in range(args.epochs):  # loop over the dataset multiple times

    	train_accu = []
    	val_accu = []
        # scheduler.step()
        if (epoch + 1) % args.step_size == 0:
            lr *= args.gamma

        
        # loss_absolute = AvgrageMeter()
        # loss_contra =  AvgrageMeter()
        train_loss_CE = AvgrageMeter()
        val_loss_CE = AvgrageMeter()
        #top5 = utils.AvgrageMeter()
        
        
        ###########################################
        '''                train             '''
        ###########################################
        model.train()
        
        # load random 16-frame clip data every epoch
        train_data = Spoofing_train(train_list, image_dir, transform=transforms.Compose([RandomErasing(), RandomHorizontalFlip(), Resize(256), CenterCrop(224), ToTensor(), Cutout(), Normaliztion()]))
        dataloader_train = DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=0)

        for i, sample_batched in enumerate(dataloader_train):
            # get the inputs
            inputs, binary_mask, spoof_label = sample_batched['image_x'].cuda(), sample_batched['binary_mask'].cuda(), sample_batched['spoofing_label'].cuda() 
            inputs_ir, inputs_depth = sample_batched['image_ir'].cuda(), sample_batched['image_depth'].cuda()
            
            
            optimizer.zero_grad()

            
            # forward + backward + optimize
            # map_x, embedding, x_Block1, x_Block2, x_Block3, x_input =  model(inputs, inputs_ir, inputs_depth)
            #map_x, embedding, x_Block1, x_Block2, x_Block3, x_input =  model(inputs, inputs_depth)

            output = model(inputs, inputs_depth, inputs_ir)
            #pdb.set_trace()
            # absolute_loss = criterion_absolute_loss(map_x, binary_mask)
            # contrastive_loss = criterion_contrastive_loss(map_x, binary_mask)
            # set_trace()
            # loss =  absolute_loss + contrastive_loss
            loss = CEloss(output, spoof_label.long().squeeze(1))

            loss.backward()
            
            optimizer.step()
            
            n = inputs.size(0)
            # loss_absolute.update(absolute_loss.data, n)
            # loss_contra.update(contrastive_loss.data, n)
            train_loss_CE.update(loss, n)
            # set_trace()
            train_accu.append(accuracy(output, spoof_label.long().squeeze(1))[0].item())
        

            if i % echo_batches == echo_batches-1:    # print every 50 mini-batches
                
                # visualization
                #FeatureMap2Heatmap(x_input, x_Block1, x_Block2, x_Block3, map_x)

                # log written
                # set_trace()
                print('epoch:%d, mini-batch:%3d, lr=%f, CE_loss= %.4f, accuracy= %.4f' % (epoch + 1, i + 1, lr,  train_loss_CE.avg, train_accu[i]))
        
            #break            
        
        # scheduler.step()  
        # whole epoch average
        print('epoch:%d, Train: CE_loss= %.4f, accuracy = %.4f' % (epoch + 1, train_loss_CE.avg, sum(train_accu)/len(train_accu)))
        log_file.write('epoch:%d, Train: CE_loss= %.4f, accuracy= %.4f' % (epoch + 1, train_loss_CE.avg, sum(train_accu)/len(train_accu)))
        log_file.flush()
           
    
            
        # epoch_test = 1
        # if epoch>10 and epoch % epoch_test == epoch_test-1:   
        #if epoch>-1 and epoch % epoch_test == epoch_test-1:  
        model.eval()
        
        with torch.no_grad():
            ###########################################
            '''                val             '''
            ###########################################
            # val for threshold
            val_data = Spoofing_valtest(val_list, image_dir, transform=transforms.Compose([Resize(256), CenterCrop(224), Normaliztion_valtest(), ToTensor_valtest()]))
            dataloader_val = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)
                        
            for i, sample_batched in enumerate(dataloader_val):
                # get the inputs
                # inputs = sample_batched['image_x'].cuda()
                # inputs_ir, inputs_depth = sample_batched['image_ir'].cuda(), sample_batched['image_depth'].cuda()
                # string_name, binary_mask = sample_batched['string_name'], sample_batched['binary_mask'].cuda()
    			
    			inputs, binary_mask, spoof_label = sample_batched['image_x'].cuda(), sample_batched['binary_mask'].cuda(), sample_batched['spoofing_label'].cuda() 
            	inputs_ir, inputs_depth = sample_batched['image_ir'].cuda(), sample_batched['image_depth'].cuda()
            	string_name = sample_batched['string_name']

                # optimizer.zero_grad()
                
                output = model(inputs, inputs_depth, inputs_ir)


                loss = CEloss(output, spoof_label.long().squeeze(1))

                n = inputs.size(0)
                val_loss_CE.update(loss, n)

                val_accu.append(accuracy(output, spoof_label.long().squeeze(1))[0].item())

        print('epoch:%d, Validation: CE_loss= %.4f' % (epoch + 1, val_loss_CE.avg))
        log_file.write('epoch:%d, Validation: CE_loss= %.4f, accuracy= %.4f' % (epoch + 1, val_loss_CE.avg, sum(train_accu)/len(train_accu)))

            #     map_score = 0.0
            #     for frame_t in range(inputs.shape[1]):
            #         map_x, embedding, x_Block1, x_Block2, x_Block3, x_input =  model(inputs[:,frame_t,:,:,:], inputs_ir[:,frame_t,:,:,:], inputs_depth[:,frame_t,:,:,:])
            #         #map_x, embedding, x_Block1, x_Block2, x_Block3, x_input =  model(inputs[:,frame_t,:,:,:], inputs_depth[:,frame_t,:,:,:])
            #         score_norm = torch.sum(map_x)/torch.sum(binary_mask[:,frame_t,:,:])
            #         map_score += score_norm
            #     map_score = map_score/inputs.shape[1]
                
            #     if map_score>1:
            #         map_score = 1.0

            #     map_score_list.append('{} {}\n'.format( string_name[0], map_score ))
                
            # map_score_val_filename = args.log+'/'+ args.log+ '_map_score_val_%d.txt'% (epoch + 1)
            # with open(map_score_val_filename, 'w') as file:
            #     file.writelines(map_score_list)                
                

            
        # save the model until the next improvement
        if (epoch+1) % save_epoch == 0:
            torch.save(model.state_dict(), args.log+'/'+args.log+'_%d.pkl' % (epoch + 1))


    print('Finished Training')
    log_file.close()
  


# main function
def train_student():
    # GPU  & log file  -->   if use DataParallel, please comment this command
    #os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % (args.gpu)

    isExists = os.path.exists(args.log)
    if not isExists:
        os.makedirs(args.log)
    log_file = open(args.log+'/'+ args.log+'_log.txt', 'w')
    
    echo_batches = args.echo_batches

    print("Oulu-NPU, P1:\n ")

    log_file.write('Oulu-NPU, P1:\n ')
    log_file.flush()

    # load the network, load the pre-trained model in UCF101?
    finetune = args.finetune
    if finetune==True:
        print('finetune!\n')

    else:
        print('train from scratch!\n')
        log_file.write('train from scratch!\n')
        log_file.flush()
         
         
        #model = CDCN_3modality2( basic_conv=Conv2d_cd, theta=0.7)
        # model = CDCN_3modality2( basic_conv=Conv2d_cd, theta=args.theta)
        model = AlexNet()

        model = model.cuda()


        lr = args.lr
        optimizer = optim.SGD(model.parameters(), lr=0.0001, weight_decay=0.0005,momentum=0.9)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    
    print(model) 
    
    
    # criterion_absolute_loss = nn.MSELoss().cuda()
    # criterion_contrastive_loss = Contrast_depth_loss().cuda() 
    CEloss = nn.CrossEntropyLoss().cuda()


    ACER_save = 1.0
    accu = []
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        # scheduler.step()
        if (epoch + 1) % args.step_size == 0:
            lr *= args.gamma

        
        # loss_absolute = AvgrageMeter()
        # loss_contra =  AvgrageMeter()
        loss_CE = AvgrageMeter()
        #top5 = utils.AvgrageMeter()
        
        
        ###########################################
        '''                train             '''
        ###########################################
        model.train()
        
        # load random 16-frame clip data every epoch
        train_data = Spoofing_train(train_list, image_dir, transform=transforms.Compose([RandomErasing(), RandomHorizontalFlip(),  ToTensor(), Cutout(), Normaliztion()]))
        dataloader_train = DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=0)

        for i, sample_batched in enumerate(dataloader_train):
            # get the inputs
            inputs, binary_mask, spoof_label = sample_batched['image_x'].cuda(), sample_batched['binary_mask'].cuda(), sample_batched['spoofing_label'].cuda() 
            inputs_ir, inputs_depth = sample_batched['image_ir'].cuda(), sample_batched['image_depth'].cuda()
            
            
            optimizer.zero_grad()

            
            # forward + backward + optimize
            # map_x, embedding, x_Block1, x_Block2, x_Block3, x_input =  model(inputs, inputs_ir, inputs_depth)
            #map_x, embedding, x_Block1, x_Block2, x_Block3, x_input =  model(inputs, inputs_depth)

            output = model(inputs, inputs_depth, inputs_ir)
            #pdb.set_trace()
            # absolute_loss = criterion_absolute_loss(map_x, binary_mask)
            # contrastive_loss = criterion_contrastive_loss(map_x, binary_mask)
            # set_trace()
            # loss =  absolute_loss + contrastive_loss
            loss = CEloss(output, spoof_label.long().squeeze(1))

            loss.backward()
            
            optimizer.step()
            
            n = inputs.size(0)
            # loss_absolute.update(absolute_loss.data, n)
            # loss_contra.update(contrastive_loss.data, n)
            loss_CE.update(loss, n)
            # set_trace()
            accu.append(accuracy(output, spoof_label.long().squeeze(1))[0].item())
        

            if i % echo_batches == echo_batches-1:    # print every 50 mini-batches
                
                # visualization
                #FeatureMap2Heatmap(x_input, x_Block1, x_Block2, x_Block3, map_x)

                # log written
                # set_trace()
                print('epoch:%d, mini-batch:%3d, lr=%f, CE_loss= %.4f, accuracy= %.4f' % (epoch + 1, i + 1, lr,  loss_CE.avg, accu[i]))
        
            #break            
        
        # scheduler.step()  
        # whole epoch average
        print('epoch:%d, Train: CE_loss= %.4f, accuracy= %.4f' % (epoch + 1, loss_CE.avg, sum(accu)/len(accu)))
        log_file.write('epoch:%d, Train: CE_loss= %.4f, accuracy= %.4f' % (epoch + 1, loss_CE.avg, sum(accu)/len(accu)))
        log_file.flush()
           
    
            
        # epoch_test = 1
        # if epoch>10 and epoch % epoch_test == epoch_test-1:   
        # #if epoch>-1 and epoch % epoch_test == epoch_test-1:  
        #     model.eval()
            
        #     with torch.no_grad():
        #         ###########################################
        #         '''                val             '''
        #         ###########################################
        #         # val for threshold
        #         val_data = Spoofing_valtest(val_list, image_dir, transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
        #         dataloader_val = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)
                
        #         map_score_list = []
                
        #         for i, sample_batched in enumerate(dataloader_val):
        #             # get the inputs
        #             inputs = sample_batched['image_x'].cuda()
        #             inputs_ir, inputs_depth = sample_batched['image_ir'].cuda(), sample_batched['image_depth'].cuda()
        #             string_name, binary_mask = sample_batched['string_name'], sample_batched['binary_mask'].cuda()
        
        #             optimizer.zero_grad()
                    
                    
        #             map_score = 0.0
        #             for frame_t in range(inputs.shape[1]):
        #                 map_x, embedding, x_Block1, x_Block2, x_Block3, x_input =  model(inputs[:,frame_t,:,:,:], inputs_ir[:,frame_t,:,:,:], inputs_depth[:,frame_t,:,:,:])
        #                 #map_x, embedding, x_Block1, x_Block2, x_Block3, x_input =  model(inputs[:,frame_t,:,:,:], inputs_depth[:,frame_t,:,:,:])
        #                 score_norm = torch.sum(map_x)/torch.sum(binary_mask[:,frame_t,:,:])
        #                 map_score += score_norm
        #             map_score = map_score/inputs.shape[1]
                    
        #             if map_score>1:
        #                 map_score = 1.0
    
        #             map_score_list.append('{} {}\n'.format( string_name[0], map_score ))
                    
        #         map_score_val_filename = args.log+'/'+ args.log+ '_map_score_val_%d.txt'% (epoch + 1)
        #         with open(map_score_val_filename, 'w') as file:
        #             file.writelines(map_score_list)                
                

            
            # save the model until the next improvement
        if epoch > 10 and epoch % epoch_test == epoch_test -1:
            torch.save(model.state_dict(), args.log+'/'+args.log+'_%d.pkl' % (epoch + 1))


    print('Finished Training')
    log_file.close()
  


 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--gpu', type=int, default=3, help='the gpu id used for predict')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')  #default=0.0001
    parser.add_argument('--batchsize', type=int, default=9, help='initial batchsize')  #default=9  
    parser.add_argument('--step_size', type=int, default=20, help='how many epochs lr decays once')  # 500  | DPC = 400
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--echo_batches', type=int, default=50, help='how many batches display once')  # 50
    parser.add_argument('--epochs', type=int, default=50, help='total training epochs')
    parser.add_argument('--log', type=str, default="CDCN_3modality2_P1", help='log and save model name')
    parser.add_argument('--finetune', action='store_true', default=False, help='whether finetune other models')
    parser.add_argument('--theta', type=float, default=0.7, help='hyper-parameters in CDCNpp')
	
    args = parser.parse_args()
    train_parent()
