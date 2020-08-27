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
from selectivenet.model import SelectiveNet, SelectiveNetRegression, ProbabilisticSelectiveNet
from selectivenet.loss import SelectiveLoss, NLLLoss, SelectiveLossNew
from selectivenet.data import DatasetBuilder
from selectivenet.evaluator import Evaluator

import numpy as np

import wandb
WANDB_PROJECT_NAME="selective_net"
if "--unobserve" in sys.argv:
    os.environ["WANDB_MODE"] = "dryrun"

# options
@click.command()
@click.option('--note', default=None)
@click.option('--prob', is_flag=True, default=False)
# model
@click.option('--init_weights', is_flag=True, default=False)
@click.option('--dim_features', type=int, default=64)
@click.option('--dropout_prob', type=float, default=0.3)
@click.option('--div_by_ten', is_flag=True, default=False, help='divide by 10 when calculating g')
# data
@click.option('-d', '--dataset', type=str, required=True)
@click.option('--dataroot', type=str, default='../data', help='path to dataset root')
@click.option('-j', '--num_workers', type=int, default=8)
@click.option('-N', '--batch_size', type=int, default=256)
@click.option('--normalize', is_flag=True, default=False)
@click.option('--augmentation', type=str, default='original', help='type of augmentation set to original, tf or lili') # just for trials
# optimization
@click.option('--num_epochs', type=int, default=800)
@click.option('--lr', type=float, default=5e-4, help='learning rate')
@click.option('--wd', type=float, default=1e-4, help='weight decay')
@click.option('--momentum', type=float, default=0.9)
@click.option('--nesterov', is_flag=True, default=False)
@click.option('--tf_opt', is_flag=True, default=False, help='minimizes tf loss') # just for trials
# loss
@click.option('--coverage', type=float, required=True)
@click.option('--alpha', type=float, default=0.5, help='balancing parameter between selective_loss and ce_loss')
@click.option('--lm', type=float, default=32.0)
@click.option('--distribution', type=str, default='Gaussian', help='type of likelihood in probabilistic model. Can be Gaussian or Laplace.') 
@click.option('--loss', type=str, required=True, help='base loss. L1 or L2 or NLL')
# logging
@click.option('-s', '--suffix', type=str, default='')
@click.option('-l', '--log_dir', type=str, default='../logs/train')
# wandb
@click.option('--unobserve', is_flag=True, default=False)

def main(**kwargs):
    train(**kwargs)

def train(**kwargs):

    FLAGS = FlagHolder()
    FLAGS.initialize(**kwargs)
    FLAGS.summary()
    
    wandb.init(project=WANDB_PROJECT_NAME, tags=["pytorch", "regression"], notes=FLAGS.note)
    wandb.config.update(kwargs)
    log_path = wandb.run.dir
    FLAGS.dump(path=os.path.join(log_path, 'flags{}.json'.format(FLAGS.suffix)))

    # dataset
    dataset_builder = DatasetBuilder(name=FLAGS.dataset, root_path=FLAGS.dataroot)
    train_dataset = dataset_builder(train=True, normalize=FLAGS.normalize, augmentation=FLAGS.augmentation)
    val_dataset = dataset_builder(train=False, normalize=FLAGS.normalize, augmentation=FLAGS.augmentation)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=FLAGS.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=FLAGS.num_workers, pin_memory=True)

    # model
    if FLAGS.prob:
        model = ProbabilisticSelectiveNet(dataset_builder.input_size, FLAGS.dim_features, init_weights=FLAGS.init_weights).cuda()
    else:
        model = SelectiveNetRegression(dataset_builder.input_size, FLAGS.dim_features, init_weights=FLAGS.init_weights).cuda()
    if torch.cuda.device_count() > 1: model = torch.nn.DataParallel(model)

    # optimizer
    params = model.parameters() 
    optimizer = torch.optim.Adam(params, lr=FLAGS.lr, weight_decay=FLAGS.wd)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

    # loss
    if FLAGS.prob:
        assert FLAGS.loss == 'NLL'
        base_loss = NLLLoss(FLAGS.distribution, reduction='none')
    else:
        if FLAGS.loss == 'L1':
            base_loss = torch.nn.L1Loss(reduction='none')
        elif FLAGS.loss == 'L2':
            base_loss = torch.nn.MSELoss(reduction='none') #Not sure what to use for reduction. Followed the train code
        else:
            raise Exception("Loss type incorrect!")
    SelectiveCELoss = SelectiveLoss(base_loss, coverage=FLAGS.coverage, alpha=FLAGS.alpha, lm=FLAGS.lm, regression=True, prob_mode=FLAGS.prob)

    # logger
    train_logger = Logger(path=os.path.join(log_path,'train_log{}.csv'.format(FLAGS.suffix)), mode='train')
    val_logger   = Logger(path=os.path.join(log_path,'val_log{}.csv'.format(FLAGS.suffix)), mode='val')

    for epoch in range(FLAGS.num_epochs):
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

            # update dictionary with results of batch
            train_metric_dict.update(loss_dict)

        # wandb log
        wandb.log(train_metric_dict.avg, step=epoch+1)

        # validation
        with torch.autograd.no_grad():
            min_loss = float('inf')
            save_dict = {}
            min_loss_tf = float('inf')
            save_dict_tf = {}
            min_coverage_diff = float('inf')
            save_dict_cov = {}

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
                
                # update dictionary with results of batch
                val_metric_dict.update(loss_dict_val)
        
        # print
        if (epoch+1) % 5 == 0 or epoch == FLAGS.num_epochs-1:
            print("Epoch [{}], val_loss: {:.4f}".format(epoch+1, val_metric_dict.avg['val_loss']))
        
        # wandb log
        wandb.log(val_metric_dict.avg, step=epoch+1)

        # find best checkpoints
        if val_metric_dict.avg['val_loss_pytorch'] < min_loss:
            save_dict_pytorch = {'epoch': epoch+1, 
                                'state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()}
            min_loss = val_metric_dict.avg['val_loss_pytorch']
        if val_metric_dict.avg['val_loss'] < min_loss_tf:
            save_dict = {'epoch': epoch+1, 
                        'state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}
            min_loss_tf = val_metric_dict.avg['val_loss']
        if (val_metric_dict.avg['val_selective_head_coverage'] - FLAGS.coverage)**2 < min_coverage_diff:
            save_dict_cov = {'epoch': epoch+1, 
                            'state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()}
            min_coverage_diff = (val_metric_dict.avg['val_selective_head_coverage'] - FLAGS.coverage)**2

        # print status
        # print_metric_dict(epoch, FLAGS.num_epochs, train_metric_dict.avg, mode='train')
        #print_metric_dict(epoch, FLAGS.num_epochs, val_metric_dict.avg, mode='val')

        train_logger.log(train_metric_dict.avg, step=(epoch+1))
        val_logger.log(val_metric_dict.avg, step=(epoch+1))

    # save checkpoints
    checkpoint_dict = [{'state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                        save_dict_pytorch, 
                        save_dict, 
                        save_dict_cov]
    checkpoint_path = os.path.join(log_path, 'checkpoints')
    save_checkpoint(checkpoint_dict, checkpoint_path)


if __name__ == '__main__':
    main()
