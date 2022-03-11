from __future__ import division

import os
import random
import argparse
import time
import math
import numpy as np
from datetime import datetime

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from data import *
import tools

from utils.augmentations import SSDAugmentation
from utils.cocoapi_evaluator import COCOAPIEvaluator
from utils.vocapi_evaluator import VOCAPIEvaluator
from utils.widerfaceapi_evaluator import WiderfaceAPIEvaluator
from utils.customapi_evaluator import CustomAPIEvaluator


# cards_id = [0, 1, 2, 3]
# use_horovod = True

cards_id = [0]
use_horovod = False


os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(f"{id}" for id in cards_id)

if use_horovod:
    import horovod.torch as hvd
    # Initialize Horovod
    hvd.init()
    # Pin GPU to be used to process local rank (one GPU per process)
    torch.cuda.set_device(hvd.local_rank())


class MyDataParallel(torch.nn.DataParallel):
    pass

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Detection')
    parser.add_argument('-v', '--version', default='slim_yolo_v2',
                        help='yolo_v2, yolo_v3, yolo_v3_spp, slim_yolo_v2, tiny_yolo_v3')
    parser.add_argument('-d', '--dataset', default='widerface',
                        help='voc or coco or custom')
    parser.add_argument('-hr', '--high_resolution', action='store_true', default=True,
                        help='use high resolution to pretrain.')  
    parser.add_argument('-ms', '--multi_scale', action='store_true', default=True,
                        help='use multi-scale trick')                  
    parser.add_argument('--batch_size', default=32, type=int, 
                        help='Batch size for training')
    parser.add_argument('--lr', default=1e-4, type=float, 
                        help='initial learning rate')
    parser.add_argument('-cos', '--cos', action='store_true', default=False,
                        help='use cos lr')
    parser.add_argument('-no_wp', '--no_warm_up', action='store_true', default=False,
                        help='yes or no to choose using warmup strategy to train')
    parser.add_argument('--wp_epoch', type=int, default=6,
                        help='The upper bound of warm-up')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='start epoch to train')
    parser.add_argument('-r', '--resume', default=None, type=str, 
                        help='keep training')
    parser.add_argument('--momentum', default=0.9, type=float, 
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float, 
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, 
                        help='Gamma update for SGD')
    parser.add_argument('--num_workers', default=16, type=int, 
                        help='Number of workers used in dataloading')
    parser.add_argument('--eval_epoch', type=int,
                            default=10, help='interval between evaluations')
    parser.add_argument('--cuda', action='store_true',
                        help='use cuda.')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug mode where only one image is trained')
    parser.add_argument('--save_folder', default='weights/', type=str, 
                        help='Gamma update for SGD')

    return parser.parse_args()


def train():
    args = parse_args()

    path_to_save = os.path.join(args.save_folder, args.dataset, args.version)
    os.makedirs(path_to_save, exist_ok=True)

    # use hi-res backbone
    if args.high_resolution:
        print('use hi-res backbone')
        hr = True
    else:
        hr = False
    
    # cuda
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("use cpu")

    # multi-scale
    if args.multi_scale:
        print('use the multi-scale trick ...')
        train_size = [640, 640]
        val_size = [416, 416]
    else:
        train_size = [416, 416]
        val_size = [416, 416]

    cfg = train_cfg
    # dataset and evaluator
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")
    print('Loading the dataset...')

    if args.dataset == 'voc':
        data_dir = VOC_ROOT
        num_classes = len(VOC_CLASSES)
        dataset = VOCDetection(root=data_dir, 
                                transform=SSDAugmentation(train_size)
                                )

        evaluator = VOCAPIEvaluator(data_root=data_dir,
                                    img_size=val_size,
                                    device=device,
                                    transform=BaseTransform(val_size),
                                    labelmap=VOC_CLASSES
                                    )

    elif args.dataset == 'coco':
        data_dir = coco_root
        num_classes = 80
        dataset = COCODataset(
                    data_dir=data_dir,
                    img_size=train_size[0],
                    transform=SSDAugmentation(train_size),
                    debug=args.debug)


        evaluator = COCOAPIEvaluator(
                        data_dir=data_dir,
                        img_size=val_size,
                        device=device,
                        transform=BaseTransform(val_size)
                        )
    
    elif args.dataset == 'widerface':

        data_dir = WIDERFACE_ROOT
        num_classes = len(WIDERFACE_CLASSES)
        dataset = WiderfaceDetection(root=data_dir, 
                                transform=SSDAugmentation(train_size)
                                )

        evaluator = WiderfaceAPIEvaluator(data_root=data_dir,
                                    img_size=val_size,
                                    device=device,
                                    transform=BaseTransform(val_size),
                                    labelmap=WIDERFACE_CLASSES,
                                    use_horovod=use_horovod
                                    )
    elif args.dataset == 'custom':

        data_dir = CUSTOM_ROOT
        num_classes = len(CUSTOM_CLASSES)
        dataset = CustomDetection(root=data_dir, 
                                transform=SSDAugmentation(train_size, mean=(0.5, 0.5, 0.5), std=(128/255.0, 128/255.0, 128/255.0))
                                )

        evaluator = CustomAPIEvaluator(data_root=data_dir,
                                    img_size=val_size,
                                    device=device,
                                    transform=BaseTransform(val_size),
                                    labelmap=CUSTOM_CLASSES,
                                    use_horovod=use_horovod
                                    )
    else:
        print('unknow dataset !! Only support voc and coco !!')
        exit(0)
    
    print('Training model on:', dataset.name)
    print('The dataset size:', len(dataset))
    print("----------------------------------------------------------")

    # dataloader
    if use_horovod:
        # Partition dataset among workers using DistributedSampler
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=hvd.size(), rank=hvd.rank())
        dataloader = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size=args.batch_size, 
                    # shuffle=True, 
                    collate_fn=detection_collate,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    sampler=train_sampler
                    )
    else:
        dataloader = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size=args.batch_size, 
                    shuffle=True, 
                    collate_fn=detection_collate,
                    num_workers=args.num_workers,
                    pin_memory=True
                    )

    # build model
    if args.version == 'yolo_v2':
        from models.yolo_v2 import myYOLOv2
        anchor_size = ANCHOR_SIZE if args.dataset == 'voc' else ANCHOR_SIZE_COCO
    
        yolo_net = myYOLOv2(device, input_size=train_size, num_classes=num_classes, trainable=True, anchor_size=anchor_size, hr=hr)
        print('Let us train yolo_v2 on the %s dataset ......' % (args.dataset))

    elif args.version == 'yolo_v3':
        from models.yolo_v3 import myYOLOv3
        anchor_size = MULTI_ANCHOR_SIZE if args.dataset == 'voc' else MULTI_ANCHOR_SIZE_COCO
        
        yolo_net = myYOLOv3(device, input_size=train_size, num_classes=num_classes, trainable=True, anchor_size=anchor_size, hr=hr)
        print('Let us train yolo_v3 on the %s dataset ......' % (args.dataset))

    elif args.version == 'yolo_v3_spp':
        from models.yolo_v3_spp import myYOLOv3Spp
        anchor_size = MULTI_ANCHOR_SIZE if args.dataset == 'voc' else MULTI_ANCHOR_SIZE_COCO
        
        yolo_net = myYOLOv3Spp(device, input_size=train_size, num_classes=num_classes, trainable=True, anchor_size=anchor_size, hr=hr)
        print('Let us train yolo_v3_spp on the %s dataset ......' % (args.dataset))

    elif args.version == 'slim_yolo_v2':
        # cards_id = [0, 1, 2, 3]
        # os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(f"{id}" for id in cards_id)

        from models.slim_yolo_v2 import SlimYOLOv2
        anchor_size = ANCHOR_SIZE if args.dataset == 'voc' else (ANCHOR_SIZE_COCO if args.dataset == "coco" else ANCHOR_SIZE_WIDER_FACE)
    
        yolo_net = SlimYOLOv2(device, input_size=train_size, num_classes=num_classes, trainable=True, anchor_size=anchor_size, hr=hr)
        print('Let us train slim_yolo_v2 on the %s dataset ......' % (args.dataset))
        # yolo_net = MyDataParallel(yolo_net, device_ids=[0, 1])
        # yolo_net = yolo_net.cuda()

        # from torchsummary import summary
        # summary(yolo_net.to("cpu"), input_size=(3, 416, 416), device="cpu")
        # while 1:
        #     pass

    elif args.version == 'tiny_yolo_v3':
        from models.tiny_yolo_v3 import YOLOv3tiny
        anchor_size = TINY_MULTI_ANCHOR_SIZE if args.dataset == 'voc' else TINY_MULTI_ANCHOR_SIZE_COCO
    
        yolo_net = YOLOv3tiny(device, input_size=train_size, num_classes=num_classes, trainable=True, anchor_size=anchor_size, hr=hr)
        print('Let us train tiny_yolo_v3 on the %s dataset ......' % (args.dataset))

    else:
        print('Unknown version !!!')
        exit()

    model = yolo_net
    model.to(device).train()

    # use tfboard
    if args.tfboard:
        print('use tensorboard')
        from torch.utils.tensorboard import SummaryWriter
        c_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        log_path = os.path.join('log/coco/', args.version, c_time)
        os.makedirs(log_path, exist_ok=True)

        writer = SummaryWriter(log_path)
    
    # keep training
    if args.resume is not None:
        print('keep training model: %s' % (args.resume))
        model.load_state_dict(torch.load(args.resume, map_location=device))

    # optimizer setup
    base_lr = args.lr
    tmp_lr = base_lr
    optimizer = optim.SGD(model.parameters(), 
                            lr=args.lr, 
                            momentum=args.momentum,
                            weight_decay=args.weight_decay
                            )
    if use_horovod:
        # Broadcast parameters and optimizer state from rank 0 to all other processes.
        hvd.broadcast_parameters(yolo_net.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        # Add Horovod Distributed Optimizer
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=yolo_net.named_parameters())

    max_epoch = cfg['max_epoch']
    epoch_size = len(dataset) // args.batch_size

    # start training loop
    t0 = time.time()

    for epoch in range(args.start_epoch, max_epoch):
        print(datetime.now())
        if use_horovod:
            train_sampler.set_epoch(epoch)
        # use cos lr
        if args.cos and epoch > 20 and epoch <= max_epoch - 20:
            # use cos lr
            tmp_lr = 0.00001 + 0.5*(base_lr-0.00001)*(1+math.cos(math.pi*(epoch-20)*1./ (max_epoch-20)))
            set_lr(optimizer, tmp_lr)

        elif args.cos and epoch > max_epoch - 20:
            tmp_lr = 0.00001
            set_lr(optimizer, tmp_lr)
        
        # use step lr
        else:
            if epoch in cfg['lr_epoch']:
                tmp_lr = tmp_lr * 0.1
                set_lr(optimizer, tmp_lr)
    

        for iter_i, (images, targets) in enumerate(dataloader):
            # WarmUp strategy for learning rate
            if not args.no_warm_up:
                if epoch < args.wp_epoch:
                    tmp_lr = base_lr * pow((iter_i+epoch*epoch_size)*1. / (args.wp_epoch*epoch_size), 4)
                    # tmp_lr = 1e-6 + (base_lr-1e-6) * (iter_i+epoch*epoch_size) / (epoch_size * (args.wp_epoch))
                    set_lr(optimizer, tmp_lr)

                elif epoch == args.wp_epoch and iter_i == 0:
                    tmp_lr = base_lr
                    set_lr(optimizer, tmp_lr)
        
            # to device
            images = images.to(device)

            # multi-scale trick
            if iter_i % 10 == 0 and iter_i > 0 and args.multi_scale:
                # randomly choose a new size
                size = random.randint(10, 19) * 32
                train_size = [size, size]
                model.set_grid(train_size)
            if args.multi_scale:
                # interpolate
                images = torch.nn.functional.interpolate(images, size=train_size, mode='bilinear', align_corners=False)
            
            # make labels
            targets = [label.tolist() for label in targets]
            if args.version == 'yolo_v2' or args.version == 'slim_yolo_v2':
                targets = tools.gt_creator(input_size=train_size, 
                                           stride=yolo_net.stride, 
                                           label_lists=targets, 
                                           anchor_size=anchor_size
                                           )
            else:
                targets = tools.multi_gt_creator(input_size=train_size, 
                                                 strides=yolo_net.stride, 
                                                 label_lists=targets, 
                                                 anchor_size=anchor_size
                                                 )
            targets = torch.tensor(targets).float().to(device)

            # forward and loss
            conf_loss, cls_loss, txtytwth_loss, total_loss = model(images, target=targets)

            # backprop
            total_loss.backward()        
            optimizer.step()
            optimizer.zero_grad()

            # display
            if iter_i % 10 == 0:
                if args.tfboard:
                    # viz loss
                    writer.add_scalar('object loss', conf_loss.item(), iter_i + epoch * epoch_size)
                    writer.add_scalar('class loss', cls_loss.item(), iter_i + epoch * epoch_size)
                    writer.add_scalar('local loss', txtytwth_loss.item(), iter_i + epoch * epoch_size)
                
                t1 = time.time()
                print('[Epoch %d/%d][Iter %d/%d][lr %.6f]'
                    '[Loss: obj %.2f || cls %.2f || bbox %.2f || total %.2f || size %d || time: %.2f]'
                        % (epoch+1, max_epoch, iter_i, epoch_size, tmp_lr,
                            conf_loss.item(), cls_loss.item(), txtytwth_loss.item(), total_loss.item(), train_size[0], t1-t0),
                        flush=True)

                t0 = time.time()

        # evaluation
        if (epoch + 1) % args.eval_epoch == 0:
            if not use_horovod or (use_horovod and hvd.rank() == 0):
                model.trainable = False
                model.set_grid(val_size)
                model.eval()

                # evaluate
                evaluator.evaluate(model)

                # convert to training mode.
                model.trainable = True
                model.set_grid(train_size)
                model.train()

        # save model
        if (epoch + 1) % 10 == 0:
            if not use_horovod or (use_horovod and hvd.rank() == 0):
                print('Saving state, epoch:', epoch + 1)
                torch.save(model.state_dict(), os.path.join(path_to_save, 
                            args.version + '_' + repr(epoch + 1) + '.pth')
                            )  


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':

    train()
