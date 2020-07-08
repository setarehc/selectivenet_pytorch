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
from external.dada.io import load_checkpoint
from external.dada.logger import Logger

from selectivenet.vgg_variant import vgg16_variant
from selectivenet.model import SelectiveNet
from selectivenet.loss import SelectiveLoss
from selectivenet.data import DatasetBuilder
from selectivenet.evaluator import Evaluator

import wandb
WANDB_PROJECT_NAME="selective_net"
if "--unobserve" in sys.argv:
    os.environ["WANDB_MODE"] = "dryrun"
run = wandb.init(project=WANDB_PROJECT_NAME, tags=["pytorch", "test"])

# options
@click.command()
# model
@click.option('--dim_features', type=int, default=512)
@click.option('--dropout_prob', type=float, default=0.3)
@click.option('-c', '--checkpoint', type=str, default='setarehc/selective_net', help='checkpoint path')
@click.option('-w', '--weight', type=str, default='final', help='model weight to load') # final, best_val or best_val_tf
@click.option('--exp_id', type=str, required=True, help='checkpoint experiment id in wandb')
# data
@click.option('-d', '--dataset', type=str, required=True)
@click.option('--dataroot', type=str, default='/home/setarehc/selectivenet_pytorch/data', help='path to dataset root')
@click.option('-j', '--num_workers', type=int, default=8)
@click.option('-N', '--batch_size', type=int, default=128)
@click.option('--normalize', is_flag=True, default=True)
@click.option('--augmentation', type=str, default='original', help='type of augmentation set to original, tf or lili') # just for trials
# loss
@click.option('--coverage', type=float, required=True)
@click.option('--alpha', type=float, default=0.5, help='balancing parameter between selective_loss and ce_loss')

def main(**kwargs):
    test(**kwargs)

def test(**kwargs):
    wandb.config.update(kwargs)

    FLAGS = FlagHolder()
    FLAGS.initialize(**kwargs)
    FLAGS.summary()

    # dataset
    dataset_builder = DatasetBuilder(name=FLAGS.dataset, root_path=FLAGS.dataroot)
    test_dataset = dataset_builder(train=False, normalize=FLAGS.normalize, augmentation=FLAGS.augmentation)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=FLAGS.batch_size, shuffle=False, num_workers=FLAGS.num_workers, pin_memory=True)

    # model
    features = vgg16_variant(dataset_builder.input_size, FLAGS.dropout_prob).cuda()
    model = SelectiveNet(features, FLAGS.dim_features, dataset_builder.num_classes).cuda()
    best_model = wandb.restore(os.path.join('checkpoints', 'checkpoint_{}.pth'.format(FLAGS.weight)), run_path=os.path.join(FLAGS.checkpoint, FLAGS.exp_id)) # model file
    load_checkpoint(model=model, path=best_model.name)

    if torch.cuda.device_count() > 1: model = torch.nn.DataParallel(model)

    # loss
    base_loss = torch.nn.CrossEntropyLoss(reduction='none')
    SelectiveCELoss = SelectiveLoss(base_loss, coverage=FLAGS.coverage)
   
    # pre epoch
    test_metric_dict = MetricDict()
    
    # test
    with torch.autograd.no_grad():
        for i, (x,t) in enumerate(test_loader):
            model.eval()
            x = x.to('cuda', non_blocking=True)
            t = t.to('cuda', non_blocking=True)

            # forward
            out_class, out_select, out_aux = model(x)

            # compute selective loss
            loss_dict = OrderedDict()
            loss_dict = SelectiveCELoss(out_class, out_select, out_aux, t, mode='test')
            loss = loss_dict['loss_pytorch']
            loss_dict['loss_pytorch'] = loss.detach().cpu().item()
            loss_tf = loss_dict['loss']
            loss_dict['loss'] = loss_tf.detach().cpu().item()

            # evaluation
            evaluator = Evaluator(out_class.detach(), t.detach(), out_select.detach())
            loss_dict.update(evaluator())

            test_metric_dict.update(loss_dict)

    # post epoch
    print_metric_dict(None, None, test_metric_dict.avg, mode='test')

    return test_metric_dict.avg

if __name__ == '__main__':
    main()