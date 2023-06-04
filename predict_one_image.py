"""Sample PyTorch Inference script
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import logging
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import PILToTensor
from torchvision import transforms

from timm.data import Dataset, create_loader, resolve_data_config , create_transform
from timm.utils import AverageMeter, setup_default_logging
from timm.utils import *

import ckdn

torch.backends.cudnn.benchmark = True

class Person:
  def __init__(self, name, age):
    self.name = name
    self.age = age


class IQA_CKDN:
    def __init__(self):
        cnf = {
                "interpolation":"",
                "mean":None,
                "model":"resnet",
                "std":None,
            }

        # create model
        self.model = ckdn.model()
        #model = vgg.vgg16()
        logging.info('Model %s created, param count: %d' %
                        (cnf["model"], sum([m.numel() for m in self.model.parameters()])))

        state_dicts = torch.load("model_best.pth.tar")['state_dict']
        new_state_dict = {}
        for k in state_dicts.keys():
            new_state_dict[k.split('module.')[1]] = state_dicts[k]

        self.model.load_state_dict(new_state_dict,strict=False) 

        self.config = resolve_data_config(cnf, model=self.model)

        num_gpu = 1

        if num_gpu > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(num_gpu))).cuda()
        else:
            self.model = self.model.cuda()


    def predict(self, restored_addr , degraded_addr):
        self.model.eval()


        with torch.no_grad():
            rest = Image.open(restored_addr).convert('RGB')
            dist = Image.open(degraded_addr).convert('RGB')

            my_transform = transforms.Compose([create_transform(input_size=self.config['input_size'])])
            rest = my_transform(rest)  
            dist = my_transform(dist)
            output = self.model.forward_test(rest.unsqueeze(0).cuda() , dist.unsqueeze(0).cuda())
            return output[:,0].cpu().numpy()[0]

# def IQA(restored_addr , degraded_addr):
   
#     # might as well try to do something useful...
#     cnf = {
#          "interpolation":"",
#          "mean":None,
#          "model":"resnet",
#          "std":None,
#         }

#     # create model
#     model = ckdn.model()
#     #model = vgg.vgg16()
#     logging.info('Model %s created, param count: %d' %
#                  (cnf["model"], sum([m.numel() for m in model.parameters()])))

#     state_dicts = torch.load("model_best.pth.tar")['state_dict']
#     new_state_dict = {}
#     for k in state_dicts.keys():
#         new_state_dict[k.split('module.')[1]] = state_dicts[k]

#     model.load_state_dict(new_state_dict,strict=False) 

#     config = resolve_data_config(cnf, model=model)

#     num_gpu = 1
    
#     if num_gpu > 1:
#         model = torch.nn.DataParallel(model, device_ids=list(range(num_gpu))).cuda()
#     else:
#         model = model.cuda()


#     model.eval()


#     with torch.no_grad():
#         rest = Image.open(restored_addr).convert('RGB')
#         dist = Image.open(degraded_addr).convert('RGB')

#         my_transform = transforms.Compose([create_transform(input_size=config['input_size'])])
#         rest = my_transform(rest)  
#         dist = my_transform(dist)
#         output = model.forward_test(rest.unsqueeze(0).cuda() , dist.unsqueeze(0).cuda())
#         return output[:,0].cpu().numpy()[0]
            

