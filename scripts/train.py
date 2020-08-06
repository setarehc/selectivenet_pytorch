import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

import click
from collections import OrderedDict

import torch
import torchvision

from external.dada.flag_holder import FlagHolder
from external.dada.metric import MetricDict
from external.dada.io import print_metric_dict
from external.dada.io import save_checkpoint
from external.dada.io import create_log_path
from external.dada.logger import Logger

from selectivenet.vgg_variant import vgg16_variant
from selectivenet.model import SelectiveNet
from selectivenet.loss import SelectiveLoss
from selectivenet.data import DatasetBuilder
from selectivenet.evaluator import Evaluator

import numpy as np

import wandb
WANDB_PROJECT_NAME="selective_net"
if "--unobserve" in sys.argv:
    os.environ["WANDB_MODE"] = "dryrun"
wandb.init(project=WANDB_PROJECT_NAME, tags=["pytorch"])

# options
@click.command()
# model
@click.option('--dim_features', type=int, default=512)
@click.option('--dropout_prob', type=float, default=0.3)
@click.option('--div_by_ten', is_flag=True, default=False, help='divide by 10 when calculating g')
# data
@click.option('-d', '--dataset', type=str, required=True)
@click.option('--dataroot', type=str, default='../data', help='path to dataset root')
@click.option('-j', '--num_workers', type=int, default=8)
@click.option('-N', '--batch_size', type=int, default=128)
@click.option('--normalize', is_flag=True, default=True)
@click.option('--augmentation', type=str, default='original', help='type of augmentation set to original, tf or lili') # just for trials
# optimization
@click.option('--num_epochs', type=int, default=300)
@click.option('--lr', type=float, default=0.1, help='learning rate')
@click.option('--wd', type=float, default=5e-4, help='weight decay')
@click.option('--momentum', type=float, default=0.9)
@click.option('--nesterov', is_flag=True, default=False)
@click.option('--tf_opt', is_flag=True, default=False, help='minimizes tf loss') # just for trials
# loss
@click.option('--coverage', type=float, required=True)
@click.option('--alpha', type=float, default=0.5, help='balancing parameter between selective_loss and ce_loss')
# logging
@click.option('-s', '--suffix', type=str, default='')
@click.option('-l', '--log_dir', type=str, default='../logs/train')
# wandb
@click.option('--unobserve', is_flag=True, default=False)

def main(**kwargs):
    train(**kwargs)

def train(**kwargs):
    wandb.config.update(kwargs)

    FLAGS = FlagHolder()
    FLAGS.initialize(**kwargs)
    FLAGS.summary()
    log_path = wandb.run.dir
    FLAGS.dump(path=os.path.join(log_path, 'flags{}.json'.format(FLAGS.suffix)))
    
    # dataset
    dataset_builder = DatasetBuilder(name=FLAGS.dataset, root_path=FLAGS.dataroot)
    train_dataset = dataset_builder(train=True, normalize=FLAGS.normalize, augmentation=FLAGS.augmentation)
    val_dataset = dataset_builder(train=False, normalize=FLAGS.normalize, augmentation=FLAGS.augmentation)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=FLAGS.num_workers, pin_memory=True)

    # model
    features = vgg16_variant(dataset_builder.input_size, FLAGS.dropout_prob).cuda()
    model = SelectiveNet(features, FLAGS.dim_features, dataset_builder.num_classes, div_by_ten=FLAGS.div_by_ten).cuda()
    if torch.cuda.device_count() > 1: model = torch.nn.DataParallel(model)

    # optimizer
    params = model.parameters() 
    optimizer = torch.optim.SGD(params, lr=FLAGS.lr, momentum=FLAGS.momentum, weight_decay=FLAGS.wd, nesterov=FLAGS.nesterov)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

    # loss
    base_loss = torch.nn.CrossEntropyLoss()
    SelectiveCELoss = SelectiveLoss(base_loss, coverage=FLAGS.coverage, alpha=FLAGS.alpha)

    # logger
    train_logger = Logger(path=os.path.join(log_path,'train_log{}.csv'.format(FLAGS.suffix)), mode='train')
    val_logger   = Logger(path=os.path.join(log_path,'val_log{}.csv'.format(FLAGS.suffix)), mode='val')

    for epoch in range(FLAGS.num_epochs):
        #import pdb; pdb.set_trace()
        # per epoch
        train_metric_dict = MetricDict()
        val_metric_dict = MetricDict()

        # train
        for i, (x,t) in enumerate(train_loader):
            model.train() #TODO: check what this is and what it does
            x = x.to('cuda', non_blocking=True)
            t = t.to('cuda', non_blocking=True)

            # forward
            out_class, out_select, out_aux = model(x)

            # loss and metrics
            loss_dict = OrderedDict()
            loss_dict = SelectiveCELoss(out_class, out_select, out_aux, t)
            loss = loss_dict['loss_pytorch']
            loss_dict['loss_pytorch'] = loss.detach().cpu().item()
            loss_tf = loss_dict['loss']
            loss_dict['loss'] = loss_tf.detach().cpu().item()

            # backward
            optimizer.zero_grad()
            if FLAGS.tf_opt:
                loss_tf.backward()
            else:
                loss.backward()
            optimizer.step()

            train_metric_dict.update(loss_dict)
        # wandb log
        wandb.log(loss_dict, step=epoch)

        # validation
        with torch.autograd.no_grad():
            min_loss = float('inf')
            save_dict = {}
            min_loss_tf = float('inf')
            save_dict_tf = {}

            for i, (x,t) in enumerate(val_loader):
                model.eval()
                x = x.to('cuda', non_blocking=True)
                t = t.to('cuda', non_blocking=True)

                # forward
                out_class, out_select, out_aux = model(x)

                # loss and metrics
                loss_dict_val = OrderedDict()
                loss_dict_val = SelectiveCELoss(out_class, out_select, out_aux, t, mode='validation')
                loss_dict_val['val_loss_pytorch'] = loss_dict_val['val_loss_pytorch'].detach().cpu().item()
                loss_dict_val['val_loss'] = loss_dict_val['val_loss'].detach().cpu().item()

                # Save best models
                if loss_dict_val['val_loss_pytorch'] < min_loss:
                    save_dict_pytorch = {'epoch': epoch, 
                                'state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()}
                if loss_dict_val['val_loss'] < min_loss_tf:
                    save_dict = {'epoch': epoch, 
                                   'state_dict': model.state_dict(),
                                   'optimizer_state_dict': optimizer.state_dict()}

                # evaluation
                evaluator = Evaluator(out_class.detach(), t.detach(), out_select.detach())
                loss_dict_val.update(evaluator())

                val_metric_dict.update(loss_dict_val)
        # wandb log
        wandb.log(loss_dict_val, step=epoch)

        # post epoch
        # print_metric_dict(epoch, FLAGS.num_epochs, train_metric_dict.avg, mode='train')
        print_metric_dict(epoch, FLAGS.num_epochs, val_metric_dict.avg, mode='val')

        train_logger.log(train_metric_dict.avg, step=(epoch+1))
        val_logger.log(val_metric_dict.avg, step=(epoch+1))

        scheduler.step()

    # save checkpoints
    #run.save()
    checkpoint_dict = [{'state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                        save_dict_pytorch, 
                        save_dict]
    checkpoint_path = os.path.join(log_path, 'checkpoints')
    save_checkpoint(checkpoint_dict, checkpoint_path)


if __name__ == '__main__':
    main()
