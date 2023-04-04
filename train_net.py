import numpy as np
import torch
import torch.optim as optim

import time
import random
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
import cv2 as cv
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torchvision.transforms as transforms
import yaml
from config import *
from volleyball import *
from collective import *
from dataset import *
from gcn_model import *
from base_model import *
from utils import *
from trans import ViT
from group.models.head.st_plus_tr_cross_cluster import ST_plus_TR_cross_cluster
from easydict import EasyDict
def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
            
def adjust_lr(optimizer, new_lr):
    print('change learning rate:',new_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def train_net(cfg):
    """
    training gcn net
    """
    os.environ['CUDA_VISIBLE_DEVICES']=cfg.device_list
    
    # Show config parameters
    cfg.init_config()
    show_config(cfg)
    
    # Reading dataset
    training_set,validation_set=return_dataset(cfg)
    
    params = {
        'batch_size': cfg.batch_size,
        'shuffle': True,
        'num_workers': 0
    }
    training_loader=data.DataLoader(training_set,**params)
    
    params['batch_size']=cfg.test_batch_size
    validation_loader=data.DataLoader(validation_set,**params)
    
    # Set random seed
    np.random.seed(cfg.train_random_seed)
    torch.manual_seed(cfg.train_random_seed)
    random.seed(cfg.train_random_seed)

    # Set data position
    if cfg.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print('gpu')
        print(torch.cuda.get_device_properties(0))


    else:
        device = torch.device('cpu')
    
    # Build model and optimizer
    basenet_list={'volleyball':Basenet_volleyball, 'collective':Basenet_collective}
    gcnnet_list={'volleyball':GCNnet_volleyball, 'collective':GCNnet_collective}
    with open('config/cluster_tr/inv3_cluster_sttr_global_w3_b3.yaml') as f:#------------------------------
        config = yaml.load(f, Loader=yaml.FullLoader)#-----------------------------------------
    config = EasyDict(config)
    # pose_head = ST_plus_TR_cross_cluster(config.structure.pose_head)
    # pose_head = pose_head.to(device)
    #pose_head.load_state_dict(torch.load(cfg.pose_head_path))
    #print('Load model states from: ', cfg.pose_head_path)
    # optimizer2 = optim.Adam(filter(lambda p: p.requires_grad, pose_head.parameters()), lr=0.0001)
    # optimizer2 = optim.Adam(filter(lambda p: p.requires_grad, pose_head.parameters()), lr=cfg.train_learning_rate,weight_decay=cfg.weight_decay)

    if cfg.training_stage==1:
        Basenet=basenet_list[cfg.dataset_name]
        model=Basenet(cfg,config)#------------------------------------------
        # model.loadmodel(cfg.stage1_model_path)
    elif cfg.training_stage==2:
        GCNnet=gcnnet_list[cfg.dataset_name]
        model=GCNnet(cfg)
        # Load backbone
        model.loadmodel(cfg.stage1_model_path)
    else:
        assert(False)
    #print('a')
    # if cfg.use_multi_gpu:
    #     model=nn.DataParallel(model)
    model=model.to(device)
    #print('modeldevice')
    model.train()
    if cfg.set_bn_eval:
        model.apply(set_bn_eval)
    
    optimizer=optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=cfg.train_learning_rate,weight_decay=cfg.weight_decay)

    train_list={'volleyball':train_volleyball, 'collective':train_collective}
    test_list={'volleyball':test_volleyball, 'collective':test_collective}
    train=train_list[cfg.dataset_name]
    test=test_list[cfg.dataset_name]
    
    if cfg.test_before_train:
        test_info=test(validation_loader, model, device, 0, cfg)
        print(test_info)
    #print('start')
    # Training iteration
    best_result={'epoch':0, 'activities_acc':0}
    start_epoch=1
    for epoch in range(start_epoch, start_epoch+cfg.max_epoch):
        
        if epoch in cfg.lr_plan:
            adjust_lr(optimizer, cfg.lr_plan[epoch])
            # adjust_lr(optimizer2, cfg.lr2_plan[epoch])
        #print('epoch')
        # One epoch of forward and backward
        train_info=train(training_loader, model,device, optimizer, epoch, cfg)
        show_epoch_info('Train', cfg.log_path, train_info)

        # Test
        # if epoch % cfg.test_interval_epoch == 0:
        if epoch > cfg.test_interval_epoch:
            test_info=test(validation_loader, model, device, epoch, cfg)
            show_epoch_info('Test', cfg.log_path, test_info)

            if test_info['activities_acc']>best_result['activities_acc']:
                best_result=test_info
            print_log(cfg.log_path, 
                      'Best group activity accuracy: %.2f%% at epoch #%d.'%(best_result['activities_acc'], best_result['epoch']))
            
            # Save model
            if cfg.training_stage==2:
                state = {
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }

                filepath=cfg.result_path+'/stage%d_epoch%d_%.2f%%.pth'%(cfg.training_stage,epoch,test_info['activities_acc'])
                torch.save(state, filepath)
                print('model saved to:',filepath)   
            elif cfg.training_stage==1:
                if test_info['activities_acc'] == best_result['activities_acc']:
                    for m in model.modules():
                        if isinstance(m, Basenet):
                            filepath=cfg.result_path+'/Vstage%d_epoch%d_%.2f%%.pth'%(cfg.training_stage,epoch,test_info['activities_acc'])
                            m.savemodel(filepath)
    #                         print('model saved to:',filepath)
    #                 filepath2 = cfg.result_path + '/Tstage%d_epoch%d_%.2f%%.pth' % (cfg.training_stage, epoch, test_info['activities_acc'])
    #                 torch.save(pose_head.state_dict(),filepath2)
            else:
                assert False
    
   
def train_volleyball(data_loader, model,device, optimizer, epoch, cfg): #----------------------
    
    actions_meter=AverageMeter()
    activities_meter=AverageMeter()
    loss_meter=AverageMeter()
    aux_loss_meter=AverageMeter()
    epoch_timer=Timer()
    # a=[]
    # b=[]
    i=0
    for batch_data in data_loader:
        #print("A",batch_data[0].shape)
        #print("B",batch_data[1].shape)
        model.train()
        if cfg.set_bn_eval:
            model.apply(set_bn_eval)
    
        # prepare batch data
        batch_data=[b.to(device=device) for b in batch_data]
        batch_size=batch_data[0].shape[0]
        num_frames=batch_data[0].shape[1]

        actions_in=batch_data[2].reshape((batch_size,num_frames,cfg.num_boxes))
        activities_in=batch_data[3].reshape((batch_size,num_frames))

        actions_in=actions_in[:,0,:].reshape((batch_size*cfg.num_boxes,))
        activities_in=activities_in[:,0].reshape((batch_size,))
        # a.append(activities_in)

        # forward
        actions_scores,activities_scores,aux_loss=model((batch_data[0],batch_data[1],batch_data[4])) # train stage 1
        # actions_scores,activities_scores=model((batch_data[0],batch_data[1]))

        B = batch_data[0].shape[0]
        T = batch_data[0].shape[1]
        N=12
        # poses=batch_data[4]
        # poses = poses.reshape(B * T * N, -1)
        # pose_fc = nn.Sequential(nn.Linear(34, 1024), nn.Linear(1024, 256))  # -------------------------
        # pose_fc = pose_fc.to(device)
        # poses_features = pose_fc(poses).reshape(B, T, N, -1)
        # poses_token = poses_features.permute(2, 0, 1, 3).contiguous().reshape(N, B * T, -1).mean(0, keepdim=True)
        # actions_scores2, activities_scores2, aux_loss2 = pose_head(poses_features, poses_token)
        # actions_scores3=actions_scores2+actions_scores
        # activities_scores3=activities_scores2+activities_scores

        # b.append(activities_scores)

        # v = ViT(
        #     num_classes=8,
        #     dim=1024,
        #     depth=6,
        #     heads=16,
        #     mlp_dim=2048,
        #     dropout=0.1,
        #     emb_dropout=0.1,
        # )
        #v=v.to(device)
        #activities_scores=v(boxes_states)
        #print("actonns_scores",actions_scores.shape)

        # Predict actions
        if isinstance(actions_scores, list):     # train stage 1
            actions_scores = sum(actions_scores)   # train stage 1
        actions_weights=torch.tensor(cfg.actions_weights).to(device=device)
        actions_loss=F.cross_entropy(actions_scores,actions_in,weight=actions_weights)
        actions_labels=torch.argmax(actions_scores,dim=1)  #---------------------------
        actions_correct=torch.sum(torch.eq(actions_labels.int(),actions_in.int()).float())

        # Predict activities
        if isinstance(activities_scores, list):            # train stage 1
            activities_scores = sum(activities_scores)   # train stage 1
        activities_loss=F.cross_entropy(activities_scores,activities_in)
        activities_labels=torch.argmax(activities_scores,dim=1)
        activities_correct=torch.sum(torch.eq(activities_labels.int(),activities_in.int()).float())
        

        # Get accuracy
        actions_accuracy=actions_correct.item()/actions_scores.shape[0]
        activities_accuracy=activities_correct.item()/activities_scores.shape[0]

        actions_meter.update(actions_accuracy, actions_scores.shape[0])
        activities_meter.update(activities_accuracy, activities_scores.shape[0])

        # Total loss
        total_loss=activities_loss+cfg.actions_loss_weight*actions_loss
        loss_meter.update(total_loss.item(), batch_size)
        aux_loss_meter.update(aux_loss.item(), batch_size)   # train stage 1
        # Optim1
        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        aux_loss.backward()
        optimizer.step()
        # if(i%2==0):                             # train stage 1
        #     total_loss.backward()
        #     i=i+1
        # else:
        #     aux_loss.backward()
        #     i=i+1
        # optimizer.step()

        # Total loss2
        # total_loss2 = activities_loss2 + cfg.actions_loss_weight * actions_loss2
        # loss_meter.update(total_loss2.item(), batch_size)
        # # Optim2
        # optimizer2.zero_grad()
        # total_loss2.backward(retain_graph=True)
        # optimizer.step()
        # optimizer2.step()
    train_info={
        'time':epoch_timer.timeit(),
        'epoch':epoch,
        'loss':loss_meter.avg,
        'activities_acc':activities_meter.avg*100,
        'actions_acc':actions_meter.avg*100
    }
    # print(a.shape,b.shape)
    return train_info
        
    
def test_volleyball(data_loader, model, device, epoch, cfg):
    model.eval()
    
    actions_meter=AverageMeter()
    activities_meter=AverageMeter()
    loss_meter=AverageMeter()
    aux_loss_meter = AverageMeter()

    # a =torch.tensor([])
    # b = torch.tensor([])
    # a=a.to(device)
    # b=b.to(device)
    epoch_timer=Timer()
    with torch.no_grad():
        for batch_data_test in data_loader:
            # prepare batch data
            batch_data_test=[b.to(device=device) for b in batch_data_test]
            batch_size=batch_data_test[0].shape[0]
            num_frames=batch_data_test[0].shape[1]
            # print("0",batch_data_test[0].shape)
            # print("1",batch_data_test[1].shape)
            actions_in=batch_data_test[2].reshape((batch_size,num_frames,cfg.num_boxes))
            activities_in=batch_data_test[3].reshape((batch_size,num_frames))

            actions_in = actions_in[:, 0, :].reshape((batch_size * cfg.num_boxes,))
            activities_in = activities_in[:, 0].reshape((batch_size,))
            # a=torch.cat((a,activities_in),0)
            # print("activities_in",activities_in.shape)
            # print(activities_in)

            # forward
            # actions_scores,activities_scores=model((batch_data_test[0],batch_data_test[1]))
            actions_scores, activities_scores, aux_loss = model((batch_data_test[0], batch_data_test[1], batch_data_test[4]))    # train stage 1
            # actions_scores, activities_scores = model((batch_data_test[0], batch_data_test[1]))
            B = batch_data_test[0].shape[0]
            T = batch_data_test[0].shape[1]
            N = 12
            # poses = batch_data_test[4]
            # poses = poses.reshape(B * T * N, -1)
            # pose_fc = nn.Sequential(nn.Linear(34, 1024), nn.Linear(1024, 256))  # -------------------------
            # pose_fc = pose_fc.to(device)
            # poses_features = pose_fc(poses).reshape(B, T, N, -1)
            # poses_token = poses_features.permute(2, 0, 1, 3).contiguous().reshape(N, B * T, -1).mean(0, keepdim=True)
            #
            # actions_scores2, activities_scores2, aux_loss2 = pose_head(poses_features, poses_token)
            #
            # actions_scores3 = actions_scores2 + actions_scores
            # activities_scores3 = activities_scores2 + activities_scores

            # b=torch.cat((b,activities_scores),0)

            # print("target",activities_scores3.shape)
            # print(activities_scores3)
            #actions_scores,activities_scores,actions_scores3,activities_scores3=model((batch_data_test[0],batch_data_test[1],batch_data_test[4]))#-----------------
            #actions_scores,boxes_states=model((batch_data_test[0],batch_data_test[1]))
            #print("boxes_states",boxes_states.shape)
            # v = ViT(
            #     num_classes=8,
            #     dim=1024,
            #     depth=6,
            #     heads=16,
            #     mlp_dim=2048,
            #     dropout=0.1,
            #     emb_dropout=0.1,
            # )
            #v = v.to(device)
            #activities_scores = v(boxes_states)
            # Predict actions
            # actions_in=actions_in[:,0,:].reshape((batch_size*cfg.num_boxes,))
            # activities_in=activities_in[:,0].reshape((batch_size,))
            if isinstance(actions_scores, list):                          # train stage 1
                actions_scores = sum(actions_scores)                        # train stage 1
            actions_weights=torch.tensor(cfg.actions_weights).to(device=device)
            #print(actions_weights.shape)
            actions_loss=F.cross_entropy(actions_scores,actions_in,weight=actions_weights)  
            # actions_loss2=F.cross_entropy(actions_scores2,actions_in,weight=actions_weights)
            #actions_labels=torch.argmax(actions_scores3,dim=1)
            actions_labels=torch.argmax(actions_scores,dim=1)#----------------

            # Predict activities
            if isinstance(activities_scores, list):                        # train stage 1
                activities_scores = sum(activities_scores)                  # train stage 1
            activities_loss=F.cross_entropy(activities_scores,activities_in)
            # activities_loss2=F.cross_entropy(activities_scores2,activities_in)
            activities_labels=torch.argmax(activities_scores,dim=1)
            # activities_labels=torch.argmax(activities_scores3,dim=1)#------------------

            actions_correct=torch.sum(torch.eq(actions_labels.int(),actions_in.int()).float())
            activities_correct=torch.sum(torch.eq(activities_labels.int(),activities_in.int()).float())
            
            # Get accuracy
            actions_accuracy=actions_correct.item()/actions_scores.shape[0]
            activities_accuracy=activities_correct.item()/activities_scores.shape[0]

            actions_meter.update(actions_accuracy, actions_scores.shape[0])
            activities_meter.update(activities_accuracy, activities_scores.shape[0])

            # Total loss
            total_loss=activities_loss+cfg.actions_loss_weight*actions_loss
            loss_meter.update(total_loss.item(), batch_size)

            # total_loss2 = activities_loss2 + cfg.actions_loss_weight * actions_loss2
            # loss_meter.update(total_loss2.item(), batch_size)
            aux_loss_meter.update(aux_loss.item(), batch_size)                   # train stage 1

    test_info={
        'time':epoch_timer.timeit(),
        'epoch':epoch,
        'loss':loss_meter.avg,
        'activities_acc':activities_meter.avg*100,
        'actions_acc':actions_meter.avg*100
    }
    # print(a.shape,b.shape)
    # torch.save(a,"Tensora.pth")
    # torch.save(b,"Tensorb.pth")
    return test_info


def train_collective(data_loader, model, device, optimizer, epoch, cfg):
    
    actions_meter=AverageMeter()
    activities_meter=AverageMeter()
    loss_meter=AverageMeter()
    epoch_timer=Timer()
    for batch_data in data_loader:
        model.train()
        model.apply(set_bn_eval)
    
        # prepare batch data
        batch_data=[b.to(device=device) for b in batch_data]
        batch_size=batch_data[0].shape[0]
        num_frames=batch_data[0].shape[1]

        # forward
        actions_scores,activities_scores=model((batch_data[0],batch_data[1],batch_data[4]))
        
        actions_in=batch_data[2].reshape((batch_size,num_frames,cfg.num_boxes))
        activities_in=batch_data[3].reshape((batch_size,num_frames))
        bboxes_num=batch_data[4].reshape(batch_size,num_frames)

        actions_in_nopad=[]
        if cfg.training_stage==1:
            actions_in=actions_in.reshape((batch_size*num_frames,cfg.num_boxes,))
            bboxes_num=bboxes_num.reshape(batch_size*num_frames,)
            for bt in range(batch_size*num_frames):
                N=bboxes_num[bt]
                actions_in_nopad.append(actions_in[bt,:N])
        else:
            for b in range(batch_size):
                N=bboxes_num[b][0]
                actions_in_nopad.append(actions_in[b][0][:N])
        actions_in=torch.cat(actions_in_nopad,dim=0).reshape(-1,)  #ALL_N,
            
        if cfg.training_stage==1:
            activities_in=activities_in.reshape(-1,)
        else:
            activities_in=activities_in[:,0].reshape(batch_size,)
        
        # Predict actions
        actions_loss=F.cross_entropy(actions_scores,actions_in,weight=None)  
        actions_labels=torch.argmax(actions_scores,dim=1)  #B*T*N,
        actions_correct=torch.sum(torch.eq(actions_labels.int(),actions_in.int()).float())

        # Predict activities
        activities_loss=F.cross_entropy(activities_scores,activities_in)
        activities_labels=torch.argmax(activities_scores,dim=1)  #B*T,
        activities_correct=torch.sum(torch.eq(activities_labels.int(),activities_in.int()).float())
        
        
        # Get accuracy
        actions_accuracy=actions_correct.item()/actions_scores.shape[0]
        activities_accuracy=activities_correct.item()/activities_scores.shape[0]
        
        actions_meter.update(actions_accuracy, actions_scores.shape[0])
        activities_meter.update(activities_accuracy, activities_scores.shape[0])

        # Total loss
        total_loss=activities_loss+cfg.actions_loss_weight*actions_loss
        loss_meter.update(total_loss.item(), batch_size)

        # Optim
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
    train_info={
        'time':epoch_timer.timeit(),
        'epoch':epoch,
        'loss':loss_meter.avg,
        'activities_acc':activities_meter.avg*100,
        'actions_acc':actions_meter.avg*100
    }
    
    return train_info
        
    
def test_collective(data_loader, model, device, epoch, cfg):
    model.eval()
    
    actions_meter=AverageMeter()
    activities_meter=AverageMeter()
    loss_meter=AverageMeter()
    
    epoch_timer=Timer()
    with torch.no_grad():
        for batch_data in data_loader:
            # prepare batch data
            batch_data=[b.to(device=device) for b in batch_data]
            batch_size=batch_data[0].shape[0]
            num_frames=batch_data[0].shape[1]
            
            actions_in=batch_data[2].reshape((batch_size,num_frames,cfg.num_boxes))
            activities_in=batch_data[3].reshape((batch_size,num_frames))
            bboxes_num=batch_data[4].reshape(batch_size,num_frames)

            # forward
            actions_scores,activities_scores=model((batch_data[0],batch_data[1],batch_data[4]))
            
            actions_in_nopad=[]
            
            if cfg.training_stage==1:
                actions_in=actions_in.reshape((batch_size*num_frames,cfg.num_boxes,))
                bboxes_num=bboxes_num.reshape(batch_size*num_frames,)
                for bt in range(batch_size*num_frames):
                    N=bboxes_num[bt]
                    actions_in_nopad.append(actions_in[bt,:N])
            else:
                for b in range(batch_size):
                    N=bboxes_num[b][0]
                    actions_in_nopad.append(actions_in[b][0][:N])
            actions_in=torch.cat(actions_in_nopad,dim=0).reshape(-1,)  #ALL_N,
            
            if cfg.training_stage==1:
                activities_in=activities_in.reshape(-1,)
            else:
                activities_in=activities_in[:,0].reshape(batch_size,)

            actions_loss=F.cross_entropy(actions_scores,actions_in)  
            actions_labels=torch.argmax(actions_scores,dim=1)  #ALL_N,
            actions_correct=torch.sum(torch.eq(actions_labels.int(),actions_in.int()).float())

            # Predict activities
            activities_loss=F.cross_entropy(activities_scores,activities_in)
            activities_labels=torch.argmax(activities_scores,dim=1)  #B,
            activities_correct=torch.sum(torch.eq(activities_labels.int(),activities_in.int()).float())
            
            # Get accuracy
            actions_accuracy=actions_correct.item()/actions_scores.shape[0]
            activities_accuracy=activities_correct.item()/activities_scores.shape[0]

            actions_meter.update(actions_accuracy, actions_scores.shape[0])
            activities_meter.update(activities_accuracy, activities_scores.shape[0])

            # Total loss
            total_loss=activities_loss+cfg.actions_loss_weight*actions_loss
            loss_meter.update(total_loss.item(), batch_size)

    test_info={
        'time':epoch_timer.timeit(),
        'epoch':epoch,
        'loss':loss_meter.avg,
        'activities_acc':activities_meter.avg*100,
        'actions_acc':actions_meter.avg*100
    }
    
    return test_info
