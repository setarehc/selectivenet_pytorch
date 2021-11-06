import os
import sys

from argparse import ArgumentParser

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

from collections import OrderedDict

import torch

from logger_utils.metric import MetricDict
from logger_utils.io import print_metric_dict
from logger_utils.io import save_checkpoint
from logger_utils.logger import Logger

from selectivenet.vgg_variant import vgg16_variant
from selectivenet.model import SelectiveNet
from selectivenet.loss import SelectiveLoss
from selectivenet.data import DatasetBuilder
from selectivenet.evaluator import Evaluator

import wandb
WANDB_PROJECT_NAME="selective_net"
if "--unobserve" in sys.argv:
    os.environ["WANDB_MODE"] = "dryrun"

def main(args):
    wandb.init(project=WANDB_PROJECT_NAME, tags=["pytorch"])
    train(args)

def train(args):
    wandb.config.update(args)

    log_path = wandb.run.dir
    
    # dataset
    dataset_builder = DatasetBuilder(name=args.dataset, root_path=args.dataroot)
    train_dataset = dataset_builder(train=True, normalize=args.normalize, augmentation=args.augmentation)
    val_dataset = dataset_builder(train=False, normalize=args.normalize, augmentation=args.augmentation)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # model
    features = vgg16_variant(dataset_builder.input_size, args.dropout_prob).cuda()
    model = SelectiveNet(features, args.dim_features, dataset_builder.num_classes, div_by_ten=args.div_by_ten).cuda()
    if torch.cuda.device_count() > 1: model = torch.nn.DataParallel(model)

    # optimizer
    params = model.parameters() 
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=args.nesterov)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

    # loss
    base_loss = torch.nn.CrossEntropyLoss()
    SelectiveCELoss = SelectiveLoss(base_loss, coverage=args.coverage, alpha=args.alpha)

    # logger
    train_logger = Logger(path=os.path.join(log_path,'train_log{}.csv'.format(args.suffix)), mode='train')
    val_logger   = Logger(path=os.path.join(log_path,'val_log{}.csv'.format(args.suffix)), mode='val')

    for epoch in range(args.num_epochs):

        train_metric_dict = MetricDict()
        val_metric_dict = MetricDict()

        # train
        for i, (x,t) in enumerate(train_loader):
            model.train()
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
            if args.tf_opt:
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
        print_metric_dict(epoch, args.num_epochs, val_metric_dict.avg, mode='val')

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
    parser = ArgumentParser()
    # model
    parser.add_argument('--dim_features', type=int, default=512)
    parser.add_argument('--dropout_prob', type=float, default=0.3)
    parser.add_argument('--div_by_ten', action='store_true', help='divide by 10 when calculating g') # flag - default is false
    # data
    parser.add_argument('-d', '--dataset', type=str, required=True)
    parser.add_argument('--dataroot', type=str, default='../data', help='path to dataset root')
    parser.add_argument('-j', '--num_workers', type=int, default=8)
    parser.add_argument('-N', '--batch_size', type=int, default=128)
    parser.add_argument('--normalize', action='store_false') # flag - default is true
    parser.add_argument('--augmentation', type=str, default='original', help='type of augmentation set to original, tf or lili') # just for trials
    # optimization
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--wd', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', action='store_true') # flag - default is false 
    parser.add_argument('--tf_opt', action='store_true', help='minimizes tf loss') # flag - default is false 
    # loss
    parser.add_argument('--coverage', type=float, required=True)
    parser.add_argument('--alpha', type=float, default=0.5, help='balancing parameter between selective_loss and ce_loss')
    # logging
    parser.add_argument('-s', '--suffix', type=str, default='')
    parser.add_argument('-l', '--log_dir', type=str, default='../logs/train')
    # wandb 
    parser.add_argument('--unobserve', action='store_true', help='disable Weights & Biases') # flag - default is false 
    args = parser.parse_args()
    main(args)