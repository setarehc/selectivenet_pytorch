import os
import sys

import torch

import json

def print_metric_dict(epoch, num_epochs, metric_dict:dict, reverse:bool=True, overwrite:bool=False, mode:str='train'):
    """
    print metric dict info to command line
    Args:
        epoch:          current epoch
        num_epochs:     total number of training epochs
        metric_dict:    metric_dict
        reverse:        print reverse order
        overwrite:      overwrite print message or not
    """
    modes = set(['train', 'val', 'test'])
    if mode not in modes: raise ValueError('mode is invalid')

    print_message =  '\r\033[K' if overwrite is True else ''
    # add epoch
    if (mode=='train') or (mode=='val'): 
        print_message += 'epoch [{:d}/{:d}] '.format(epoch+1, num_epochs)
    # add mode
    print_message += ' ({}) '.format(mode)
    # add metric
    dict_items = reversed(list(metric_dict.items())) if reverse is True else metric_dict.items()
    for k,v in dict_items:
        print_message += '{}:{:.4f} '.format(k,v) 
    # add new line
    print_message += '' if overwrite is True else '\n'
    sys.stdout.write(print_message)
    sys.stdout.flush()

def save_checkpoint(checkpoint_dicts, path, orator=True):
    os.makedirs(path, exist_ok=True)
    # final epoch
    torch.save(checkpoint_dicts[0], os.path.join(path, 'checkpoint_{}.pth'.format('final')))
    # best validation loss
    torch.save(checkpoint_dicts[1], os.path.join(path, 'checkpoint_{}.pth'.format('best_val')))
    # best validation tf loss
    torch.save(checkpoint_dicts[2], os.path.join(path, 'checkpoint_{}.pth'.format('best_val_tf')))
    if orator: print('>>> Model was saved to "{}"'.format(path))

def load_checkpoint(model, path, orator=True, optimizer=None):
    if os.path.isfile(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None: # used in train mode to continue training
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if orator: print('>>> Checkpoint was loaded from "{}"'.format(path))
    else:
       raise ValueError('Incorrect checkpoint: file does not exist')

def create_log_path(path, experminet):
    experminet.save()
    log_path = os.path.join(path, experminet.name)
    os.makedirs(log_path, exist_ok=True)
    return log_path

def print_config(path, orator=True):
    if os.path.isfile(path):
        config_file = open(path)
        config_dict = json.load(config_file)
    if orator: print('>>> Checkpoint config is; ', config_dict);    