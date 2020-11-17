import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
# from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader
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
from utils.avg_meters import AverageMeter

from models.model import transformer

# from data.kitti_loader_lidar import KittiDataset, KittiDataset_Fusion
from data.argo_loader import Argo, Argo_geometric
from utils.logger import set_logger, get_logger



def get_eval_dataset(args):
    eval_data = Argo_geometric(
        obs_len = args.obs_len, pred_len = args.pred_len, raw_data_path = args.train_dir, 
        processed_data_path = args.processed_data_path+ 'val.pt', split = 'val', root=args.train_dir)

    eval_loader = DataLoader(
        eval_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_worker,
        pin_memory=True)
    return eval_data, eval_loader


def get_train_dataset(args):
    # n_features = 35 if args.no_reflex else 36
    train_data = Argo_geometric(
        obs_len = args.obs_len, pred_len = args.pred_len, raw_data_path = args.train_dir, 
        processed_data_path = args.processed_data_path + 'train.pt', split = 'train', root=args.train_dir)

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
    
    model = transformer(args.n_head, args.d_inputs, args.d_inputs_emb)
    model = model.cuda()
#    model = nn.DataParallel(model, device_ids=num_gpu)
    model = nn.DataParallel(model)


    ## INITIALIZE OPTIMIZERS AND SCHEDULERS
    reg_criterion = nn.SmoothL1Loss(reduction='none')
    reg_criterion = reg_criterion.cuda()
    
    
    # optimizer = optim.RMSprop(model.parameters(), lr=args.lr,
    #                               momentum=args.momentum,
    #                               weight_decay=args.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
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
        avg_loss =  AverageMeter()#('Loss', len(train_loader))
        avg_time =  AverageMeter()#('Time', len(train_loader))
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

#
#            avg_load_time.update(time.time() - load_start)
#            forward_start = time.time()


#            backward_start = time.time()
            start = time.time()
            optimizer.zero_grad()
            
            x = batch.x.cuda(non_blocking=True)
            
            y = batch.y.cuda(non_blocking=True)

        
            output = model(x)
            output = output.squeeze(1)

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

    # np.random.seed(args.seed)
    # random.seed(args.seed)
    logger = set_logger(name=args.shortname, level=args.loglevel,
                        filepath=osp.join(savepath, 'log.txt'))
    logger.info("=> Training mode")
    train(args)
