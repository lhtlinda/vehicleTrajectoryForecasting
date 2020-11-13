import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.nn.functional as F

import argparse
import configargparse

import numpy as np
import time
import os
import errno
import os.path as osp
import subprocess
import random
# from tqdm import tqdm
from torch_utils import AverageMeter

from models.model import MultiHeadAttention

# from data.kitti_loader_lidar import KittiDataset, KittiDataset_Fusion
from data.argo_loader import Argo
from utils.logger import set_logger, get_logger



def get_eval_dataset(args):
    eval_data = Argo(root= args.train_dir, split="val")
    eval_loader = DataLoader(
        eval_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker,
        pin_memory=True)
    return eval_data, eval_loader


def get_train_dataset(args):
    # n_features = 35 if args.no_reflex else 36
    train_data = Argo(root= args.train_dir,split="train")
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker,
        pin_memory=True)
    return train_data, train_loader


def display_args(args, logger):
    use_gpu = torch.cuda.is_available()
    num_gpu = list(range(torch.cuda.device_count()))

    logger.info("========== training info ==========")
    logger.info("host: {}".format(os.getenv('HOSTNAME')))
    logger.info("gpu: {}".format(use_gpu))
    if use_gpu:
        logger.info("gpu_dev: {}".format(num_gpu))

    for arg in vars(args):
        logger.info("{} = {}".format(arg, getattr(args, arg)))
    logger.info("===================================")


def train(args):
    use_gpu = torch.cuda.is_available()
    num_gpu = list(range(torch.cuda.device_count()))
    #assert use_gpu, "GPU not enabled. This codebase assumes use of GPU. Exiting."

    logger = get_logger(name=args.shortname)
    display_args(args, logger)
    ts = time.time()

    ## INITIALIZE DATASETS
    train_data, train_loader = get_train_dataset(args)
    eval_data, eval_loader = get_eval_dataset(args)

    ## INITIALIZE MODELS
    
    model = MultiHeadAttention(args.n_head, args.d_inputs, args.d_inputs_emb)
#    model = model.cuda()
#    model = nn.DataParallel(model, device_ids=num_gpu)
    model = nn.DataParallel(model)


    ## INITIALIZE OPTIMIZERS AND SCHEDULERS
    reg_criterion = nn.SmoothL1Loss(reduction='none')
    reg_criterion = reg_criterion.cuda()
    
    
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay)
    # placeholder, list parsing doesn't work correctly atm
    lr_milestones = [15,25,35]
    scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=lr_milestones, gamma=args.gamma)

    # Resuming model
    if args.resume:
        logger.info("Resuming...")
        checkpoint_path = osp.join(savepath, args.checkpoint)
        if os.path.isfile(checkpoint_path):
            logger.info("Loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            logger.info(
                "Resumed successfully from epoch {}.".format(start_epoch))
        else:
            logger.warning(
                "Model {} not found. "
                "Train from scratch".format(checkpoint_path))
            start_epoch = 0
    else:
        start_epoch = 0

    logger.info("Finish initialization, beginning training... Time elapsed {:.3f} s".format(time.time() - ts))

    ## TRAINING LOOP
    processes = []
    for epoch in range(start_epoch, args.epochs):
        model.train()
        avg_loss =  AverageMeter('Loss', len(train_loader))
        avg_time =  AverageMeter('Time', len(train_loader))
#        ts = time.time()
#
#        avg_class_loss = AverageMeter()
#        avg_reg_loss = AverageMeter()
#        avg_total_loss = AverageMeter()
#        avg_load_time = AverageMeter()
#        avg_fw_time = AverageMeter()
#        avg_bw_time = AverageMeter()
#        load_start = time.time()
        for iteration, batch in enumerate(train_loader):
#            import pdb; pdb.set_trace()

#            lidar = batch['lidar'].cuda(non_blocking=True)
#            hdmap = batch['hdmap'].cuda(non_blocking=True)
#            # class_labels = batch['cl'].cuda(non_blocking=True)
#            # reg_labels = batch['rl'].cuda(non_blocking=True)
#            bboxes = batch['bboxes'].cuda(non_blocking=True)
#            pred_labels = batch['future_xy'].cuda(non_blocking=True)
#
#            avg_load_time.update(time.time() - load_start)
#            forward_start = time.time()

            ## DETECTION LOSS
            # TODO: ask Xiangyu which feature to output
#            class_outs, reg_outs, features = pixor(lidar)  # TODO: change to output features
            # class_outs = class_outs.squeeze(1)
            # class_loss, reg_loss = \
            #     compute_detection_loss(epoch, class_outs, reg_outs,
            #         class_labels, reg_labels, class_criterion,
            #         reg_criterion, args)
            # avg_fw_time.update(time.time() - forward_start)
            # avg_class_loss.update(class_loss.item())
            # avg_reg_loss.update(reg_loss.item() \
            #     if not isinstance(reg_loss, int) else reg_loss)

            ## PREDICTION LOSS
            # features: shape B x 256 x H x W
            # actor_batch_ind: list of len N (N is the number of actors in the batch)
            # ie: if N = 3, batchsize = 2: [0, 0, 1]
#            actor_features, actor_bboxes, actor_batch_ind, actor_gt_ind = postprocess_detections(features, class_outs, reg_outs, reg_labels)
            # predictions: shape No x T x 5
            # det: 5 x det
            # batch_ind: [0, 0, 1]
            # gt: 7 x det
            # gt_ind: [0, 2, 4, 3, 1]
#            predictions = spagnn(actor_features, actor_bboxes, actor_batch_ind)  # output: mu_x, mu_y, sigma_x, sigma_y, rho
            # returns for each ground truth other, pair up prediction + label
#            pred_nll_loss = Gaussian2DLikelihood(
#                epoch, pred_labels, predictions, actor_gt_ind
#            )  # Junan
            # pred_labels: n_gt x T x 2
            
#            loss = alpha1 * pred_nll_loss  # + alpha2 * class_loss + alpha3 * reg_loss
#            avg_total_loss.update(loss.item())

#            backward_start = time.time()
            start = time.time()
            optimizer.zero_grad()
            
            x = batch['x']
            x = x.view(x.shape[0]*x.shape[1],x.shape[2],x.shape[3])
            
            y = batch['y']
            y = y.view(y.shape[0]*y.shape[2], y.shape[1])
        
            output,attn = model(x)
            output = output.squeeze(1)
#            print('output', output.shape)
#            print('y', y.shape)
            loss = reg_criterion(output, y)
            loss = torch.flatten(loss).sum()
            avg_loss.update(loss.item())
            loss.backward()
            optimizer.step()
            avg_time.update(time.time()-start)
#            avg_bw_time.update(time.time() - backward_start)

            if iteration % args.logevery == 0:
                logger.info("epoch {:d}, iter {:d}, loss: {:.5f},"
                            " runtime: {:.3f} s  "
                            .format(
                                epoch,
                                iteration, avg_loss.avg, avg_time.avg ))
            del loss

        scheduler.step()

        logger.info("Finish epoch {}, time elapsed {:.3f} s".format(
            epoch, time.time() - ts))

        if epoch % args.eval_every_epoch == 0 and epoch >= args.start_eval:
            print("Eval not implemented, continuing.")
            continue
            logger.info("Evaluation begins at epoch {}".format(epoch))

#            # TODO (Xiangyu): evaluation logic for prediction
#            evaluate(eval_data, eval_loader, pixor,
#                        args.batch_size, gpu=use_gpu, logger=logger,
#                        args=args, epoch=epoch, processes=processes)
            avg_eval_loss = AverageMeter()
            with torch.no_grad():
                for iteration, batch in enumerate(train_loader):
                    x = batch['x']
                    x = x.view(x.shape[0]*x.shape[1],x.shape[2],x.shape[3])
            
                    y = batch['y']
                    y = y.view(y.shape[0]*y.shape[2], y.shape[1])
                    output, attn = model(x)
                    output = output.squeeze(1)
                    
                    loss = reg_criterion(output, y)
                    
                    loss = torch.flatten(loss).sum()
                    avg_eval_loss.update(loss.item())
            logger.info("evaluation at epoch {:d}, loss: {:.5f},"
                            .format(
                                epoch,
                                avg_loss.avg))
           
        if epoch % args.save_every == 0:
            saveto = osp.join(savepath, "checkpoint_{}.pth.tar".format(epoch))
            torch.save({'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch
                        }, saveto)
            logger.info("model saved to {}".format(saveto))
            #symlink_force(saveto, osp.join(savepath, "checkpoint.pth.tar"))

def evol(s):
    try:
        return eval(s)
    except:
        return s

def parse_args():
    parser = configargparse.ArgParser(
        description="Train model",
        default_config_files=["configs/attn.cfg"])
    # parser.add('-c', '--config', required=True,
    #     is_config_file=True, help='config file')
#    parser.add('-c', '--config_file', required=True, is_config_file=True, help='config file path')
    args, unparsed = parser.parse_known_args()
    for arg in unparsed:
        split = arg.find("=")
        setattr(args, arg[2:split], evol(arg[split+1:]))
    return args


if __name__ == "__main__":
   
    args = parse_args()
    
    if args.jobid is not None:
        args.run_name = args.run_name + '-' + args.jobid
        # if args.losswise_tag is not None:
        #     args.losswise_tag = args.losswise_tag + '-' + args.jobid

    try:
        args.shortname = args.run_name
    except:
        setattr(args, "shortname", args.run_name)
    # create dir for saving
    args.saverootpath = osp.abspath(args.saverootpath)
    savepath = osp.join(args.saverootpath, args.run_name)
    if not osp.exists(savepath):
        os.makedirs(savepath)

    np.random.seed(args.seed)
    random.seed(args.seed)
    logger = set_logger(name=args.shortname, level=args.loglevel,
                        filepath=osp.join(savepath, 'log.txt'))
    logger.info("=> Training mode")
    train(args)
