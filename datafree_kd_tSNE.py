# modify datafree/fast_meta.py:195 to support visualization
import argparse
from math import gamma
import os
import random
import shutil
import warnings

import registry
import datafree

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from kornia import augmentation
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.cm as cm

parser = argparse.ArgumentParser(description='Data-free Knowledge Distillation')

# Data Free
parser.add_argument('--method', required=True, choices=['zskt', 'dfad', 'dafl', 'deepinv', 'dfq', 'cmi', 'fast', 'fast_meta'])
parser.add_argument('--adv', default=0, type=float, help='scaling factor for adversarial distillation')
parser.add_argument('--bn', default=0, type=float, help='scaling factor for BN regularization')
parser.add_argument('--oh', default=0, type=float, help='scaling factor for one hot loss (cross entropy)')
parser.add_argument('--act', default=0, type=float, help='scaling factor for activation loss used in DAFL')
parser.add_argument('--balance', default=0, type=float, help='scaling factor for class balance')
parser.add_argument('--save_dir', default='run/synthesis', type=str)

parser.add_argument('--cr', default=1, type=float, help='scaling factor for contrastive model inversion')
parser.add_argument('--cr_T', default=0.5, type=float, help='temperature for contrastive model inversion')
parser.add_argument('--cmi_init', default=None, type=str, help='path to pre-inverted data')


# Basic
parser.add_argument('--data_root', default='./data')
parser.add_argument('--teacher', default='wrn40_2')
parser.add_argument('--student', default='wrn16_1')
parser.add_argument('--dataset', default='cifar100')
parser.add_argument('--lr', default=0.1, type=float,
                    help='initial learning rate for KD')
parser.add_argument('--lr_decay_milestones', default="120,150,180", type=str,
                    help='milestones for learning rate decay')
parser.add_argument('--lr_g', default=1e-3, type=float, 
                    help='initial learning rate for generator')
parser.add_argument('--lr_z', default=1e-3, type=float,
                    help='initial learning rate for latent code')
parser.add_argument('--T', default=1, type=float)

parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--g_steps', default=1, type=int, metavar='N',
                    help='number of iterations for generation')
parser.add_argument('--kd_steps', default=400, type=int, metavar='N',
                    help='number of iterations for KD after generation')
parser.add_argument('--ep_steps', default=400, type=int, metavar='N',
                    help='number of total iterations in each epoch')
parser.add_argument('--warmup', default=0, type=int, metavar='N',
                    help='which epoch to start kd')

parser.add_argument('--reset_l0', default=0, type=int,
                    help='reset l0 in the generator during training')
parser.add_argument('--reset_bn', default=0, type=int,
                    help='reset bn layers during training')
parser.add_argument('--bn_mmt', default=0, type=float,
                    help='momentum when fitting batchnorm statistics')
parser.add_argument('--is_maml', default=1, type=int,
                    help='meta: is maml or reptile')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate_only', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--batch_size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--synthesis_batch_size', default=None, type=int,
                    metavar='N',
                    help='mini-batch size (default: None) for synthesis, this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

# Device
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
# TODO: Distributed and FP-16 training 
parser.add_argument('--world_size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist_url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--fp16', action='store_true',
                    help='use fp16')

# Misc
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training.')
parser.add_argument('--log_tag', default='')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print_freq', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')

best_acc1 = 0
time_cost = 0

def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    args.ngpus_per_node = ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    global time_cost
    args.gpu = gpu
    ############################################
    # GPU and FP16
    ############################################
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    if args.fp16:
        from torch.cuda.amp import autocast, GradScaler
        args.scaler = GradScaler() if args.fp16 else None 
        args.autocast = autocast
    else:
        args.autocast = datafree.utils.dummy_ctx


    ############################################
    # Logger
    ############################################
    if args.log_tag != '':
        args.log_tag = '-'+args.log_tag
    log_name = 'R%d-%s-%s-%s%s'%(args.rank, args.dataset, args.teacher, args.student, args.log_tag) if args.multiprocessing_distributed else '%s-%s-%s'%(args.dataset, args.teacher, args.student)
    args.logger = datafree.utils.logger.get_logger(log_name, output='checkpoints/datafree-%s/log-%s-%s-%s%s.txt'%(args.method, args.dataset, args.teacher, args.student, args.log_tag))
    if args.rank<=0:
        for k, v in datafree.utils.flatten_dict( vars(args) ).items(): # print args
            args.logger.info( "%s: %s"%(k,v) )

    ############################################
    # Setup dataset
    ############################################
    num_classes, ori_dataset, val_dataset = registry.get_dataset(name=args.dataset, data_root=args.data_root)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    evaluator = datafree.evaluators.classification_evaluator(val_loader)
    valData = []
    for datas, labels in val_loader:
        valData.append([datas, labels])
    
    ############################################
    # Output tSNE
    ############################################
    def outputCSV(myModel, metaData, synData, saveEpoch):

        aug = transforms.Compose([
            augmentation.RandomCrop(
                size=[32, 32], padding=4),
            augmentation.RandomHorizontalFlip(),
            normalizer,
        ])

        realData, realLabel = valData[random.randint(0, 9)]

        CLASSES = ['airplane', 'automobile', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        # MARKERS = matplotlib.markers.MarkerStyle.filled_markers

        meta_out, meta_fea = myModel(aug(metaData), return_features=True)
        meta_target = torch.argmax(meta_out, dim=-1).to('cpu').numpy()

        syn_out, syn_fea = myModel(
            aug(synData), return_features=True)
        syn_target = torch.argmax(
            syn_out, dim=-1).to('cpu').numpy()

        _, real_fea = myModel(realData.cuda(), return_features=True)

        meta_fea = meta_fea.cuda().data.cpu()
        syn_fea = syn_fea.cuda().data.cpu()
        real_fea = real_fea.cuda().data.cpu()

        allTarget = np.concatenate(
            (meta_target, syn_target, realLabel), axis=0)

        allFea = np.concatenate((meta_fea, syn_fea, real_fea), axis=0)

        tsne = TSNE(init='pca', n_components=2).fit_transform(allFea)

        x_min, x_max = np.min(tsne, 0), np.max(tsne, 0)
        data = (tsne - x_min) / (x_max - x_min)

        allLabel = np.array([CLASSES[x] for x in allTarget]
                            ).reshape(data.shape[0], 1)

        dataType = np.concatenate((np.full((256, 1), 'Meta Data'), np.full((256, 1), 'Synthesis Data'), np.full((256, 1), "Real Data")),
                                  axis=0)

        resDataFrame = np.hstack(
            (np.hstack((data, allLabel)), dataType))

        outputData = pd.DataFrame(resDataFrame, columns=[
            'x1', 'x2', 'class', 'dataType'])

        outputPath = 'checkpoints/datafree-%s/tsne_epoch%d.csv' % (
            args.method, saveEpoch)
        outputData.to_csv(outputPath, index=False)

        return outputPath

    def drawPDF(path, frac):

        OUT_PDF_NAME = {'Meta Data': 'SR', 'Synthesis Data': 'MR'}
        DATA = pd.read_csv(path, header=0)
        CLASSDICT = {'Meta Data': '+', 'Synthesis Data': 'x', 'Real Data': 'o'}
        CLASSES = ['airplane', 'automobile', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        nameIndex = path.rfind('/')
        SR_PATH = os.path.join(path[:nameIndex], 'SR')
        MR_PATH = os.path.join(path[:nameIndex], 'MR')
        if not os.path.exists(SR_PATH):
            os.mkdir(SR_PATH)
        if not os.path.exists(MR_PATH):
            os.mkdir(MR_PATH)
        pdfName = os.path.splitext(path[nameIndex+1:])[0]

        data_real = DATA[DATA['dataType'] == 'Real Data'].sample(frac=frac)
        colors = cm.tab10(np.linspace(0, 1, len(CLASSES)))

        for ffilter, value in OUT_PDF_NAME.items():

            data_filter = DATA[(DATA['dataType'] != 'Real Data') & (
                DATA['dataType'] != ffilter)].sample(frac=frac)

            data = pd.concat([data_real, data_filter], axis=0)

            labels = data['class'].values.tolist()
            t1 = data['x1'].values.tolist()
            t2 = data['x2'].values.tolist()
            CLASS = data['dataType'].values.tolist()

            fig, ax = plt.subplots(figsize=(12, 10))

            for idx in range(0, len(labels)):
                ax.scatter(t1[idx],
                           t2[idx],
                           marker=CLASSDICT[CLASS[idx]],
                           s=66,
                           color=colors[CLASSES.index(labels[idx])][:3])

            dataTypeColor = []

            for kk, vv in CLASSDICT.items():
                tmp = ax.scatter([], [], color='red',
                                 marker=vv, label=kk)
                if kk != ffilter:
                    dataTypeColor.append(tmp)

            l1 = ax.legend(handles=dataTypeColor, edgecolor=(0.5, 0.5, 0.5, 0), markerscale=2,
                           handletextpad=0.5,
                           borderpad=0.5,
                           title=r'DataType', loc='upper right', fontsize=20, title_fontsize=24)

            ax.text(0.5, 0.012, "Ten colors represent the 10 categories of cifar-10", horizontalalignment='center',
                    verticalalignment='center', size=12,
                    transform=ax.transAxes, alpha=0.2)
            plt.xticks([])
            plt.yticks([])

            plt.gca().add_artist(l1)
            plt.tight_layout()

            fig.savefig(os.path.join(
                path[:nameIndex], value, value + '_' + pdfName + '.pdf'))

            plt.close(fig)

    ############################################
    # Setup models
    ############################################
    def prepare_model(model):
        if not torch.cuda.is_available():
            print('using CPU, this will be slow')
            return model
        elif args.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                model.cuda(args.gpu)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                args.batch_size = int(args.batch_size / ngpus_per_node)
                args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
                return model
            else:
                model.cuda()
                model = torch.nn.parallel.DistributedDataParallel(model)
                return model
        elif args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
            return model
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            model = torch.nn.DataParallel(model).cuda()
            return model

    student = registry.get_model(args.student, num_classes=num_classes)
    teacher = registry.get_model(args.teacher, num_classes=num_classes, pretrained=True).eval()
    args.normalizer = normalizer = datafree.utils.Normalizer(**registry.NORMALIZE_DICT[args.dataset])
    teacher.load_state_dict(torch.load('checkpoints/pretrained/%s_%s.pth'%(args.dataset, args.teacher), map_location='cpu')['state_dict'])
    student = prepare_model(student)
    teacher = prepare_model(teacher)
    criterion = datafree.criterions.KLDiv(T=args.T)
    ############################################
    # Setup the data-free synthesizer
    ############################################
    if args.synthesis_batch_size is None:
        args.synthesis_batch_size = args.batch_size

    if args.method=='deepinv':
        synthesizer = datafree.synthesis.DeepInvSyntheiszer(
                 teacher=teacher, student=student, num_classes=num_classes, 
                 img_size=(3, 32, 32), iterations=args.g_steps, lr_g=args.lr_g,
                 synthesis_batch_size=args.synthesis_batch_size, sample_batch_size=args.batch_size, 
                 adv=args.adv, bn=args.bn, oh=args.oh, tv=0.0, l2=0.0,
                 save_dir=args.save_dir, transform=ori_dataset.transform,
                 normalizer=args.normalizer, device=args.gpu)
    elif args.method in ['zskt', 'dfad', 'dfq', 'dafl']:
        nz = 512 if args.method=='dafl' else 256
        generator = datafree.models.generator.Generator(nz=nz, ngf=64, img_size=32, nc=3)
        generator = prepare_model(generator)
        criterion = torch.nn.L1Loss() if args.method=='dfad' else datafree.criterions.KLDiv()
        synthesizer = datafree.synthesis.GenerativeSynthesizer(
                 teacher=teacher, student=teacher, generator=generator, nz=nz, 
                 img_size=(3, 32, 32), iterations=args.g_steps, lr_g=args.lr_g,
                 synthesis_batch_size=args.synthesis_batch_size, sample_batch_size=args.batch_size, 
                 adv=args.adv, bn=args.bn, oh=args.oh, act=args.act, balance=args.balance, criterion=criterion,
                 normalizer=args.normalizer, device=args.gpu)
    elif args.method=='cmi':
        nz = 256
        generator = datafree.models.generator.Generator(nz=nz, ngf=64, img_size=32, nc=3)
        generator = prepare_model(generator)
        feature_layers = None # use outputs from all conv layers
        if args.teacher=='resnet34': # use block outputs
            feature_layers = [teacher.layer1, teacher.layer2, teacher.layer3, teacher.layer4]
        synthesizer = datafree.synthesis.CMISynthesizer(teacher, student, generator,
                 nz=nz, num_classes=num_classes, img_size=(3, 32, 32), feature_reuse=False,
                 # if feature layers==None, all convolutional layers will be used by CMI.
                 feature_layers=feature_layers, bank_size=40960, n_neg=4096, head_dim=256, init_dataset=args.cmi_init,
                 iterations=args.g_steps, lr_g=args.lr_g, progressive_scale=False,
                 synthesis_batch_size=args.synthesis_batch_size, sample_batch_size=args.batch_size, 
                 adv=args.adv, bn=args.bn, oh=args.oh, cr=args.cr, cr_T=args.cr_T,
                 save_dir=args.save_dir, transform=ori_dataset.transform,
                 normalizer=args.normalizer, device=args.gpu)
    elif args.method=='fast':
        nz = 256
        generator = datafree.models.generator.Generator(nz=nz, ngf=64, img_size=32, nc=3)
        generator = prepare_model(generator)
        synthesizer = datafree.synthesis.FastSynthesizer(teacher, student, generator,
                 nz=nz, num_classes=num_classes, img_size=(3, 32, 32), init_dataset=args.cmi_init,
                 save_dir=args.save_dir, device=args.gpu,
                 transform=ori_dataset.transform, normalizer=args.normalizer,
                 synthesis_batch_size=args.synthesis_batch_size, sample_batch_size=args.batch_size,
                 iterations=args.g_steps, warmup=args.warmup, lr_g=args.lr_g, lr_z=args.lr_z,
                 adv=args.adv, bn=args.bn, oh=args.oh,
                 reset_l0=args.reset_l0, reset_bn=args.reset_bn,
                 bn_mmt=args.bn_mmt, is_maml=args.is_maml)
    elif args.method=='fast_meta':
        nz = 256
        generator = datafree.models.generator.Generator(nz=nz, ngf=64, img_size=32, nc=3)
        generator = prepare_model(generator)
        synthesizer = datafree.synthesis.FastMetaSynthesizer(teacher, student, generator,
                 nz=nz, num_classes=num_classes, img_size=(3, 32, 32),
                 init_dataset=args.cmi_init, iterations=args.g_steps, lr_g=args.lr_g,
                 synthesis_batch_size=args.synthesis_batch_size, sample_batch_size=args.batch_size,
                 adv=args.adv, bn=args.bn, oh=args.oh,
                 save_dir=args.save_dir, transform=ori_dataset.transform,
                 normalizer=args.normalizer, device=args.gpu, lr_z=args.lr_z,
                 warmup=args.warmup, reset_l0=args.reset_l0,
                 reset_bn=args.reset_bn, bn_mmt=args.bn_mmt, is_maml=args.is_maml)
    else: raise NotImplementedError
        
    ############################################
    # Setup optimizer
    ############################################
    optimizer = torch.optim.SGD(student.parameters(), args.lr, weight_decay=args.weight_decay, momentum=0.9)
    milestones = [ int(ms) for ms in args.lr_decay_milestones.split(',') ]
    scheduler = torch.optim.lr_scheduler.MultiStepLR( optimizer, milestones=milestones, gamma=0.1)

    ############################################
    # Resume
    ############################################
    args.current_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume, map_location='cpu')
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)

            if isinstance(student, nn.Module):
                student.load_state_dict(checkpoint['state_dict'])
            else:
                student.module.load_state_dict(checkpoint['state_dict'])
            best_acc1 = checkpoint['best_acc1']
            try: 
                args.start_epoch = checkpoint['epoch']
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
            except: print("Fails to load additional model information")
            print("[!] loaded checkpoint '{}' (epoch {} acc {})"
                  .format(args.resume, checkpoint['epoch'], best_acc1))
        else:
            print("[!] no checkpoint found at '{}'".format(args.resume))
        
    ############################################
    # Evaluate
    ############################################
    if args.evaluate_only:
        student.eval()
        eval_results = evaluator(student, device=args.gpu)
        print('[Eval] Acc={acc:.4f}'.format(acc=eval_results['Acc']))
        return

    ############################################
    # Train Loop
    ############################################
    for epoch in range(args.start_epoch, args.epochs):
        #if args.distributed:
        #    train_sampler.set_epoch(epoch)
        args.current_epoch=epoch
        for _ in range( args.ep_steps//args.kd_steps ): # total kd_steps < ep_steps
            # 1. Data synthesis
            vis_results, meta = synthesizer.synthesize() # g_steps
            # time_cost += cost
            # 2. Knowledge distillation
            if epoch >= args.warmup:
                train( synthesizer, [student, teacher], criterion, optimizer, args) # # kd_steps
        for vis_name, vis_image in vis_results.items():
            datafree.utils.save_image_batch( vis_image, 'checkpoints/datafree-%s/%s%s.png'%(args.method, vis_name, args.log_tag) )
        for meta_name, meta_image in meta.items():
            datafree.utils.save_image_batch( meta_image, 'checkpoints/datafree-%s/%s%s.png' %(args.method, meta_name, args.log_tag) )
        if (epoch+1) % 5 == 0 or epoch == 0:
            saveCSV = outputCSV(teacher, next(iter(meta.values())), next(
                iter(vis_results.values())), epoch)
            drawPDF(saveCSV, 0.75)
        
        student.eval()
        eval_results = evaluator(student, device=args.gpu)
        (acc1, acc5), val_loss = eval_results['Acc'], eval_results['Loss']
        args.logger.info('[Eval] Epoch={current_epoch} Acc@1={acc1:.4f} Acc@5={acc5:.4f} Loss={loss:.4f} Lr={lr:.4f}'
                .format(current_epoch=args.current_epoch, acc1=acc1, acc5=acc5, loss=val_loss, lr=optimizer.param_groups[0]['lr']))
        scheduler.step()
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        _best_ckpt = 'checkpoints/datafree-%s/%s-%s-%s.pth'%(args.method, args.dataset, args.teacher, args.student)
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.student,
                'state_dict': student.state_dict(),
                'best_acc1': float(best_acc1),
                'optimizer' : optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, _best_ckpt)
    if args.rank<=0:
        args.logger.info("Best: %.4f"%best_acc1)
        args.logger.info("Generation Cost: %1.3f" % (time_cost/3600.) )


def train(synthesizer, model, criterion, optimizer, args):
    global time_cost
    loss_metric = datafree.metrics.RunningLoss(datafree.criterions.KLDiv(reduction='sum'))
    acc_metric = datafree.metrics.TopkAccuracy(topk=(1,5))
    student, teacher = model
    optimizer = optimizer
    student.train()
    teacher.eval()
    for i in range(args.kd_steps):
        # start = time.time()
        images = synthesizer.sample()
        # end = time.time()
        # time_cost += (end - start)
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        with args.autocast():
            with torch.no_grad():
                t_out, t_feat = teacher(images, return_features=True)
            s_out = student(images.detach())
            loss_s = criterion(s_out, t_out.detach())
        optimizer.zero_grad()
        if args.fp16:
            scaler_s = args.scaler_s
            scaler_s.scale(loss_s).backward()
            scaler_s.step(optimizer)
            scaler_s.update()
        else:
            loss_s.backward()
            optimizer.step()
        acc_metric.update(s_out, t_out.max(1)[1])
        loss_metric.update(s_out, t_out)
        if args.print_freq == -1 and i % 10 == 0 and args.current_epoch >= 150:
            (train_acc1, train_acc5), train_loss = acc_metric.get_results(), loss_metric.get_results()
            args.logger.info(
                '[Train] Epoch={current_epoch} Iter={i}/{total_iters}, train_acc@1={train_acc1:.4f}, train_acc@5={train_acc5:.4f}, train_Loss={train_loss:.4f}, Lr={lr:.4f}'
                .format(current_epoch=args.current_epoch, i=i, total_iters=args.kd_steps, train_acc1=train_acc1,
                        train_acc5=train_acc5, train_loss=train_loss, lr=optimizer.param_groups[0]['lr']))
            loss_metric.reset(), acc_metric.reset()
        elif args.print_freq>0 and i % args.print_freq == 0:
            (train_acc1, train_acc5), train_loss = acc_metric.get_results(), loss_metric.get_results()
            args.logger.info('[Train] Epoch={current_epoch} Iter={i}/{total_iters}, train_acc@1={train_acc1:.4f}, train_acc@5={train_acc5:.4f}, train_Loss={train_loss:.4f}, Lr={lr:.4f}'
              .format(current_epoch=args.current_epoch, i=i, total_iters=args.kd_steps, train_acc1=train_acc1, train_acc5=train_acc5, train_loss=train_loss, lr=optimizer.param_groups[0]['lr']))
            loss_metric.reset(), acc_metric.reset()
    
def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    if is_best:
        torch.save(state, filename)

if __name__ == '__main__':
    main()