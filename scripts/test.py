from argparse import ArgumentParser
import os
import sys

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

from collections import OrderedDict

import torch

from logger_utils.metric import MetricDict
from logger_utils.io import print_metric_dict
from logger_utils.io import load_checkpoint
from logger_utils.io import print_config

from selectivenet.vgg_variant import vgg16_variant
from selectivenet.model import SelectiveNet
from selectivenet.loss import SelectiveLoss
from selectivenet.data import DatasetBuilder
from selectivenet.evaluator import Evaluator

from selectivenet.utils import post_calibrate

import wandb
WANDB_PROJECT_NAME="selective_net"
if "--unobserve" in sys.argv:
    os.environ["WANDB_MODE"] = "dryrun"

def main(args):
    wandb.init(project=WANDB_PROJECT_NAME, tags=["pytorch", "test"])
    test(args)

def test(args):
    wandb.config.update(args)
    
    # dataset
    dataset_builder = DatasetBuilder(name=args.dataset, root_path=args.dataroot)
    test_dataset = dataset_builder(train=False, normalize=args.normalize, augmentation=args.augmentation)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # model
    features = vgg16_variant(dataset_builder.input_size, args.dropout_prob).cuda()
    model = SelectiveNet(features, args.dim_features, dataset_builder.num_classes, div_by_ten=args.div_by_ten).cuda()
    model_config = wandb.restore('args.json', run_path=os.path.join(args.checkpoint, args.exp_id), replace=True)
    print_config(path=model_config.name)
    best_model = wandb.restore(os.path.join('checkpoints', 'checkpoint_{}.pth'.format(args.weight)), run_path=os.path.join(args.checkpoint, args.exp_id), replace=True) # model file
    load_checkpoint(model=model, path=best_model.name)

    if torch.cuda.device_count() > 1: model = torch.nn.DataParallel(model)

    # post calibration
    if args.calibrate:
        val_dataset = dataset_builder(train=False, normalize=args.normalize, augmentation=args.augmentation)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, num_workers=args.num_workers, pin_memory=True)
        threshold = post_calibrate(model, val_loader, args.coverage)
    else:
        threshold = 0.5

    # loss
    base_loss = torch.nn.CrossEntropyLoss(reduction='none')
    SelectiveCELoss = SelectiveLoss(base_loss, coverage=args.coverage)
   
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
            loss_dict = SelectiveCELoss(out_class, out_select, out_aux, t, threshold, mode='test')
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
    parser = ArgumentParser()
    # model
    parser.add_argument('--dim_features', type=int, default=512)
    parser.add_argument('--dropout_prob', type=float, default=0.3)
    parser.add_argument('-c', '--checkpoint', type=str, default='setarehc/selective_net', help='checkpoint path')
    parser.add_argument('-w', '--weight', type=str, default='final', help='model weight to load') # final, best_val or best_val_tf
    parser.add_argument('--exp_id', type=str, required=True, help='checkpoint experiment id in wandb')
    parser.add_argument('--div_by_ten', action='store_true', help='divide by 10 when calculating g')
    # data
    parser.add_argument('-d', '--dataset', type=str, required=True)
    parser.add_argument('--dataroot', type=str, default='/home/setarehc/selectivenet_pytorch/data', help='path to dataset root')
    parser.add_argument('-j', '--num_workers', type=int, default=8)
    parser.add_argument('-N', '--batch_size', type=int, default=128)
    parser.add_argument('--normalize', action='store_false')
    parser.add_argument('--augmentation', type=str, default='original', help='type of augmentation set to original, tf or lili') # just for trials
    # loss
    parser.add_argument('--coverage', type=float, required=True)
    parser.add_argument('--alpha', type=float, default=0.5, help='balancing parameter between selective_loss and ce_loss')
    # general
    parser.add_argument('--calibrate', action='store_true', help='performs post calibration if True')
    # wandb 
    parser.add_argument('--unobserve', action='store_true', help='disable Weights & Biases') # flag - default is false 
    args = parser.parse_args()
    main(args)