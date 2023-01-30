import torch.utils.data as Data
import torch
import scipy.io as sio
import pandas as pd
import numpy as np
from preprocess_avm import gen_bx
import torchvision.transforms as T



class Dataset(Data.Dataset):
    
    def __init__(self, list_, path_lab_txt, **kwargs):
            self.data_len = len(list_)
            self.list_ = list_
            self.path_lab_txt = path_lab_txt
            if 'rand_dilate' in kwargs:
                self.rand_dilate = kwargs['rand_dilate']
                if self.rand_dilate:
                    self.max_dilate_factor = kwargs['max_dilate_factor']

            #self.lab = list_lab
            self.transform = T.Resize((512,512))
            
    def __len__(self):        
        return self.data_len
    
    #def get_labels(self):         
        #return self.lab
    
    def __getitem__(self, index):
        
        tmp = sio.loadmat(self.list_[index])
        tmp_str = self.list_[index].split('/')
        tmp_str = tmp_str[-1][0:-4]        
        
        bx = pd.read_csv(
            self.path_lab_txt+'/'+tmp_str+'.txt', sep=" ",header=None)

        for i in range(bx.shape[0]):
            bx_tmp = np.array(bx.iloc[i,:])

            if self.rand_dilate:
                mask_tmp = gen_bx(tmp['img'].shape,bx_tmp,
                                  rand_dilate=self.rand_dilate,
                                  max_dilate_factor=self.max_dilate_factor)
            else:
                mask_tmp = gen_bx(tmp['img'].shape,bx_tmp,
                                  rand_dilate=self.rand_dilate)

            if i == 0:
                mask = mask_tmp
            else:
                mask = np.logical_or(mask,mask_tmp)
        
        # assign entry and para
        img = tmp['img']
        img = np.multiply(img,mask)
        img = img.reshape(img.shape[0],img.shape[1],1)
        img = np.transpose(img,(2,0,1))
        
        #img2 = tmp['img_bx_masked']
        
        lab = tmp['mask']
        
        # to tensro format
        img = torch.from_numpy(img).type(torch.FloatTensor)
        lab = torch.from_numpy(lab).type(torch.int64)
        
        # resize
        img = self.transform(img)
        lab = self.transform(lab.view(-1,*lab.shape)).view(512,512)

        return img, lab #, np.array(bx)


class Dataset_out(Data.Dataset):
    def __init__(self, list_, path_lab_txt, dilate_factor):
            self.data_len = len(list_)
            self.list_ = list_
            self.path_lab_txt = path_lab_txt
            self.dilate_factor = dilate_factor
            #self.lab = list_lab
            
    def __len__(self):        
        return self.data_len
    
    #def get_labels(self):         
        #return self.lab
    
    def __getitem__(self, index):
        
        tmp = sio.loadmat(self.list_[index])
        tmp_str = self.list_[index].split('/')
        tmp_str = tmp_str[-1][0:-4]        
        
        bx = pd.read_csv(
            self.path_lab_txt+'/'+tmp_str+'.txt', sep=" ",header=None)

        # a slice may have multiple bx
        for i in range(bx.shape[0]):
            bx_tmp = np.array(bx.iloc[i,:])
            mask_tmp = gen_bx(tmp['img'].shape,bx_tmp,
                              dilate=True,dilate_factor=self.dilate_factor)
            if i == 0:
                mask_bx = mask_tmp
            else:
                mask_bx = np.logical_or(mask_bx,mask_tmp)
        
        # assign entry and para
        img = tmp['img']
        img = np.multiply(img,mask_bx)
        img = img.reshape(img.shape[0],img.shape[1],1)
        img = np.transpose(img,(2,0,1))
        
        #img2 = tmp['img_bx_masked']
        
        lab = tmp['mask']
        
        # to tensro format
        img = torch.from_numpy(img).type(torch.FloatTensor)
        lab = torch.from_numpy(lab).type(torch.int64)
        mask_bx = torch.from_numpy(mask_bx).type(torch.int64)

        return img, lab, mask_bx, self.list_[index]