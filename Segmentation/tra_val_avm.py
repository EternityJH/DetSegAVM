import time
import torch
import numpy as np
import scipy.io as sio
import os
from loss_fun_avm import make_one_hot


def train(**kwargs):
    
    #path,model,loss,optimizer,dataloader,epoch,scheduler):

    path = kwargs['path']
    model = kwargs['model']
    loss = kwargs['loss']
    optimizer = kwargs['optimizer']
    dataloader = kwargs['dataloader']
    epoch = kwargs['epoch']
    scheduler = kwargs['scheduler']


    
    f = open(path + '/log.txt', 'a')
    
    model.train() # Turn on the train mode
    
    total_loss = 0.
    total_loss2 = 0.
    start_time = time.time()

    for step, (batch_x, batch_y) in enumerate(dataloader):
        
        # reset gradient of the optimizer
        optimizer.zero_grad()
                
        # apply GPU setup
        batch_x = batch_x.cuda() # data
        batch_y = batch_y.cuda() # ground truth
        
        # model prediction
        output = model(batch_x)
        
        # loss calculation
        loss_fl = loss[0](output, batch_y)
        loss_dice = loss[1](
            make_one_hot(torch.argmax(output,dim=1).reshape(output.shape[0],1,output.shape[2],output.shape[3]),2).cuda(),
            make_one_hot(batch_y.reshape(batch_y.shape[0],1,batch_y.shape[1],batch_y.shape[2]),2).cuda())
        
        # backward   
        loss_fl.backward()
        
        # update model weights
        optimizer.step()        
            
        # print log        
        total_loss += loss_fl.item()
        total_loss2 += loss_dice.item()
        log_interval = 100
        
        if (step+1) % log_interval == 0 or step+1 == len(dataloader):
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.7f} | ms/batch {:5.2f} | '
                  'loss_fl {:5.6f} | loss_dice {:5.2f} | '.format(
                    epoch, step+1, len(dataloader), scheduler.get_last_lr()[0],
                    elapsed * 1000, total_loss/(step+1), total_loss2/(step+1)))
            f.write('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.7f} | ms/batch {:5.2f} | '
                  'loss_fl {:5.6f} | loss_dice {:5.2f} | '.format(
                    epoch, step+1, len(dataloader), scheduler.get_last_lr()[0],
                    elapsed * 1000, total_loss/(step+1), total_loss2/(step+1))+'\r\n')
            
            start_time = time.time()
    
    f.close()
            
            
def validation(**kwargs):
    
    #path,model,loss,dataloader,epoch):

    path = kwargs['path']
    model = kwargs['model']
    loss = kwargs['loss']
    dataloader = kwargs['dataloader']
    epoch = kwargs['epoch']

    f = open(path + '/log.txt', 'a')

    with torch.no_grad():
        model.to(torch.device("cuda"))
        model.train(False)
    
        total_loss = 0.
        total_loss2 = 0.
        start_time = time.time()
        
        for step, (batch_x, batch_y) in enumerate(dataloader):
                
            # apply GPU setup
            batch_x = batch_x.cuda() # data
            batch_y = batch_y.cuda() # ground truth
            
            # model prediction
            output = model(batch_x)

            # loss calculation
            loss_fl = loss[0](output, batch_y)
            loss_dice = loss[1](
                make_one_hot(torch.argmax(output,dim=1).reshape(output.shape[0],1,output.shape[2],output.shape[3]),2).cuda(),
                make_one_hot(batch_y.reshape(batch_y.shape[0],1,batch_y.shape[1],batch_y.shape[2]),2).cuda())    
            
            
            # print log        
            total_loss += loss_fl.item()
            total_loss2 += loss_dice.item()
            log_interval = 100
            
            if (step+1) % log_interval == 0 or step+1 == len(dataloader):
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                    '| ms/batch {:5.2f} | '
                    'val_loss_fl {:5.6f} | val_loss_dice {:5.2f} | '.format(
                        epoch, step+1, len(dataloader),
                        elapsed * 1000, total_loss/(step+1), total_loss2/(step+1)))
                f.write('| epoch {:3d} | {:5d}/{:5d} batches | '
                    '| ms/batch {:5.2f} | '
                    'val_loss_fl {:5.6f} | val_loss_dice {:5.2f} | '.format(
                        epoch, step+1, len(dataloader),
                        elapsed * 1000, total_loss/(step+1), total_loss2/(step+1))+'\r\n')
                
                start_time = time.time()

    f.close()
            
    return total_loss
            
    
def train_out(path):
    with torch.no_grad():
        model.to(torch.device("cuda"))
        model.train(False)
    
    total_loss = 0.
    total_loss2 = 0.
    start_time = time.time()    
    
    for step, (batch_x, batch_y, bbx_dia, list_) in enumerate(train_loader_out):
               
        # apply GPU setup
        batch_x = batch_x.cuda() # data
        batch_y = batch_y.cuda() # ground truth
        
        # model prediction
        output = model(batch_x)
        
        # loss calculation
        loss_fl = criterion_FL(output, batch_y)
        loss_dice = criterion_DICE(
            make_one_hot(torch.argmax(output,dim=1).reshape(output.shape[0],1,output.shape[2],output.shape[3]),5).cuda(),
            make_one_hot(batch_y.reshape(batch_y.shape[0],1,batch_y.shape[1],batch_y.shape[2]),5).cuda())
        
        # save mat
        matData = {'t_img': np.double(batch_x[0,0,:,:].detach().cpu().numpy()), 
                   't_lab': np.uint8(batch_y.detach().cpu().numpy()),
                   'p_lab': np.uint8(torch.argmax(output,dim=1)[0,:,:].detach().cpu().numpy()),
                  'bbx_dia': np.uint8(bbx_dia)}
        
        '''
        if lab==0:
            fname = path + '/0/' + os.path.split(list_[0])[-1][0:-4] + '_dice_' + str(1-loss_dice.item()) + '.mat'
        elif lab == 1:
            fname = path + '/1/' + os.path.split(list_[0])[-1][0:-4] + '_dice_' + str(1-loss_dice.item()) + '.mat'
        '''
        
        fname = path + '/1/' + os.path.split(list_[0])[-1][0:-4] + '_dice_' + str(1-loss_dice.item()) + '.mat'
        sio.savemat(fname,matData,do_compression=True)
        
        
        # print log        
        total_loss += loss_fl.item()
        total_loss2 += loss_dice.item()
        log_interval = 100
        
        if (step+1) % log_interval == 0 or step+1 == len(train_loader_out):
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'ms/batch {:5.2f} | '
                  'val_loss_fl {:5.6f} | val_loss_dice {:5.2f} | '.format(
                    epoch, step+1, len(train_loader_out),
                    elapsed * 1000, total_loss/(step+1), total_loss2/(step+1)))
            f.write('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'ms/batch {:5.2f} | '
                  'val_loss_bce {:5.6f} | val_loss_dice {:5.2f} | '.format(
                    epoch, step+1, len(train_loader_out),
                    elapsed * 1000, total_loss/(step+1), total_loss2/(step+1))+'\r\n')
            
            start_time = time.time()
            
            
    
def inference_out(path,model,score,dataloader):
    
    f = open(path + '/log.txt', 'a')

    with torch.no_grad():
        model.to(torch.device("cuda"))
        model.train(False)
    
        start_time = time.time()    
        
        for step, (batch_x, batch_y, bbx_mask, list_) in enumerate(dataloader):
                
            # apply GPU setup
            batch_x = batch_x.cuda() # data
            batch_y = batch_y.cuda() # ground truth
            
            # model prediction
            output = model(batch_x)
            
            
            # dice calculation
            bbx_dice = score[0](batch_y.cpu().view(1,*batch_y.shape),bbx_mask.view(1,*batch_y.shape))
            output_dice = score[0](batch_y.cpu().view(1,*batch_y.shape),torch.argmax(output,dim=1).cpu().view(1,*batch_y.shape))
            
            # save mat
            matData = {'t_lab': np.squeeze(np.uint8(batch_y.detach().cpu().numpy())),
                       'p_lab': np.squeeze(np.uint8(torch.argmax(output,dim=1)[0,:,:].detach().cpu().numpy())),
                       'bbx_mask': np.squeeze(np.uint8(bbx_mask.detach().cpu().numpy()))}
            
            fname = path + '/' + os.path.split(list_[0])[-1][0:-4] + '_bbxDice_' + str(np.round(bbx_dice,4)) + '_outDice_' + str(np.round(output_dice,4)) + '.mat'
            sio.savemat(fname,matData,do_compression=True)
            
            # print log            
            log_interval = 100
            
            if (step+1) % log_interval == 0 or step+1 == len(dataloader):
                elapsed = time.time() - start_time
                print('| {:5d}/{:5d} batches | ms/batch {:5.2f} | '.format(
                        step+1, len(dataloader), elapsed*1000))
                f.write(' {:5d}/{:5d} batches | ms/batch {:5.2f} | '.format(
                        step+1, len(dataloader), elapsed*1000)+'\r\n')
                
                start_time = time.time()

    f.close()
