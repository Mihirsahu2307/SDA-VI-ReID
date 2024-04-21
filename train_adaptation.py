from __future__ import print_function
import argparse
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, TestData, RegDBData_DA, SYSUData_DA
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from model_mine import embed_net
from model_agw import embed_net as agw

from utils import *
from loss import OriTripletLoss,  CenterTripletLoss, CrossEntropyLabelSmooth, TripletLoss_WRT, MMD_Loss, MarginMMD_Loss, CMD_loss
from tensorboardX import SummaryWriter
from re_rank import random_walk, k_reciprocal

from hdmmd import HDMMD
from random_aug import RandomErasing
from itertools import cycle

import numpy as np
np.set_printoptions(threshold=np.inf)

"""Note: batch_size is P from the paper, and num_pos is K"""

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.1 , type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume', '-r', default='', type=str,
                    help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='save_model/', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=100, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str,
                    help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log/', type=str,
                    help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=4, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--method', default='base', type=str,
                    metavar='m', help='method type: base or agw')
parser.add_argument('--margin', default=0.3, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')

parser.add_argument('--share_net', default=3, type=int,
                    metavar='share', help='[1,2,3,4,5]the start number of shared network in the two-stream networks')
parser.add_argument('--re_rank', default='no', type=str, help='performing reranking. [random_walk | k_reciprocal | no]')
parser.add_argument('--pcb', default='off', type=str, help='performing PCB, on or off')
parser.add_argument('--w_center', default=2.0, type=float, help='the weight for center loss')

parser.add_argument('--local_feat_dim', default=256, type=int,
                    help='feature dimention of each local feature in PCB')
parser.add_argument('--num_strips', default=6, type=int,
                    help='num of local strips in PCB')

parser.add_argument('--aug', action='store_true', help='Use Random Erasing Augmentation')
parser.add_argument('--label_smooth', default='off', type=str, help='performing label smooth or not')
parser.add_argument('--dist_disc', type=str, help='Include Distribution Discripeancy Loss', default=None)
parser.add_argument('--margin_mmd', default=0, type=float, help='Value of Margin For MMD Loss')

parser.add_argument('--dist_w', default=0.25, type=float, help='Weight of Distribution Discrepancy Loss')
parser.add_argument('--run_name', type=str,
                    help='Run Name for following experiment', default='test_run')

# New args:
parser.add_argument('--target_ids', default=30, type=int,
                    help='Number of target train ids')
parser.add_argument('--source_model_path', default='', type=str,
                    help='Load model trained on source. Different from resume')
parser.add_argument('--source_weight', default=0.25, type=float,
                    help='Weight of source loss during adaptation')
parser.add_argument('--hdmmd_weight', default=1, type=float,
                    help='Weight of DMMD loss during adaptation')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

set_seed(args.seed)

dataset = args.dataset
if dataset == 'sysu':
    target_data_path = './SYSU-MM01'
    source_data_path = './RegDB/RegDB/'
    log_path = args.log_path + 'sysu_log/'
    test_mode = [1, 2]  # thermal to visible
elif dataset == 'regdb':
    source_data_path = './SYSU-MM01'
    target_data_path = './RegDB/RegDB/'
    log_path = args.log_path + 'regdb_log/'
    test_mode = [2, 1]  # visible to thermal

checkpoint_path = args.model_path

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(args.vis_log_path):
    os.makedirs(args.vis_log_path)

suffix = args.run_name + '_' + dataset+'_c_tri_pcb_{}_w_tri_{}'.format(args.pcb,args.w_center)
if args.pcb=='on':
    suffix = suffix + '_s{}_f{}'.format(args.num_strips, args.local_feat_dim)

suffix = suffix + '_share_net{}'.format(args.share_net)
if args.method=='agw':
    suffix = suffix + '_agw_k{}_p{}_lr_{}_seed_{}'.format(args.num_pos, args.batch_size, args.lr, args.seed)
else:
    suffix = suffix + '_base_gm10_k{}_p{}_lr_{}_seed_{}'.format(args.num_pos, args.batch_size, args.lr, args.seed)


if not args.optim == 'sgd':
    suffix = suffix + '_' + args.optim

if dataset == 'regdb':
    suffix = suffix + '_trial_{}'.format(args.trial)

sys.stdout = Logger(log_path + suffix + '_os.txt')

vis_log_dir = args.vis_log_path + suffix + '/'

if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)
writer = SummaryWriter(vis_log_dir)
print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

if args.aug:
    transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
    RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

else:
    transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
    ])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()

if dataset == 'sysu':
    
    source_trainset = RegDBData_DA(source_data_path, num_ids=206, use_test=False, trial=args.trial, transform=transform_train)
    target_trainset = SYSUData_DA(target_data_path, num_ids=args.target_ids, transform=transform_train)

    # Training labels will be re-labelled. Testing labels don't need to be relabelled as we don't utilize them for optimization
    target_trainset.train_color_label, target_trainset.train_thermal_label = relabel(target_trainset.train_color_label, target_trainset.train_thermal_label, offset=len(np.unique(source_trainset.train_color_label)))
    
    # generate the idx of each person identity
    source_color_pos, source_thermal_pos = GenIdx(source_trainset.train_color_label, source_trainset.train_thermal_label)
    target_color_pos, target_thermal_pos = GenIdx(target_trainset.train_color_label, target_trainset.train_thermal_label)

    # testing set - target set
    query_img, query_label, query_cam = process_query_sysu(target_data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(target_data_path, mode=args.mode, trial=0)
    
    print("\n\n##### Checking target and source unique labels ######\n")
    print("Unique ids in Source: ", set(source_trainset.train_color_label))
    print("Unique ids in Target: ", set(target_trainset.train_color_label))
    print("Source RGB Images: ", len(source_trainset.train_color_label))

# ie, targetset = regdb
elif dataset == 'regdb':
    
    source_trainset = SYSUData_DA(source_data_path, num_ids=395, use_test=False, transform=transform_train)
    target_trainset = RegDBData_DA(target_data_path, args.trial, num_ids=args.target_ids, transform=transform_train)

    # Training labels will be re-labelled. Testing labels don't need to be relabelled as we don't utilize them for optimization
    target_trainset.train_color_label, target_trainset.train_thermal_label = relabel(target_trainset.train_color_label, target_trainset.train_thermal_label, offset=len(np.unique(source_trainset.train_color_label)))
    
    # generate the idx of each person identity
    source_color_pos, source_thermal_pos = GenIdx(source_trainset.train_color_label, source_trainset.train_thermal_label)
    target_color_pos, target_thermal_pos = GenIdx(target_trainset.train_color_label, target_trainset.train_thermal_label)

    # testing set - target set
    query_img, query_label = process_test_regdb(target_data_path, trial=args.trial, modal='visible')
    gall_img, gall_label = process_test_regdb(target_data_path, trial=args.trial, modal='thermal')
    
    print("\n\n##### Checking target and source unique labels ######\n")
    print("Unique ids in Source: ", set(source_trainset.train_color_label))
    print("Source RGB Images: ", len(source_trainset.train_color_label))
    print("Unique ids in Target: ", set(target_trainset.train_color_label))
    

gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

# testing data loader
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

n_class = len(np.unique(source_trainset.train_color_label)) + len(np.unique(target_trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)

print('Target Dataset {} statistics:'.format(dataset)) # Target
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(target_trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(target_trainset.train_thermal_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building model..')
if args.method =='base':
    net = embed_net(n_class, no_local= 'off', gm_pool =  'on', arch=args.arch, share_net=args.share_net, pcb=args.pcb, local_feat_dim=args.local_feat_dim, num_strips=args.num_strips)
elif args.method=='agw':
    # net = embed_net(n_class, no_local= 'on', gm_pool = 'on', arch=args.arch,  share_net=args.share_net, pcb=args.pcb)
    net = agw(n_class)
net.to(device)


cudnn.benchmark = True

if len(args.resume) > 0:
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))
        
        
def load_source_trained_model():
    if len(args.source_model_path) > 0:
        model_path = args.source_model_path
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(model_path))
            checkpoint = torch.load(model_path)
            
            # Source classifier weight could be smaller
            # Get the state_dict of the loaded model
            state_dict = checkpoint['net']
            
            # Remove keys related to the final classifier layer
            keys_to_remove = ['classifier.weight']  # Assuming the final layer is named 'classifier'
            for key in keys_to_remove:
                del state_dict[key]
                
            net.load_state_dict(state_dict, strict=False)
            print('==> loaded checkpoint {}'
                .format(args.source_model_path))
        else:
            print('==> no checkpoint found at {}'.format(model_path))

# define loss function
if args.label_smooth == 'off':
    criterion_id = nn.CrossEntropyLoss()
else:
    criterion_id = CrossEntropyLabelSmooth(n_class)

if args.method == 'agw':
    criterion_tri = TripletLoss_WRT()
else:
    loader_batch = args.batch_size * args.num_pos
    #criterion_tri= OriTripletLoss(batch_size=loader_batch, margin=args.margin)
    criterion_tri= CenterTripletLoss(batch_size=loader_batch, margin=args.margin)

criterion_id.to(device)
criterion_tri.to(device)

criterion_mmd = MMD_Loss().to(device)
criterion_margin_mmd = MarginMMD_Loss(margin=args.margin_mmd, P=args.batch_size, K=args.num_pos).to(device)
criterion_cmd = CMD_loss(P=args.batch_size, K=args.num_pos).to(device)
# criterion_dmmd = DMMD(P=args.batch_size, K=args.num_pos).to(device)
criterion_hdmmd = HDMMD(P=args.batch_size, K=args.num_pos).to(device)


if args.optim == 'sgd':
    if args.pcb == 'on':
        ignored_params = list(map(id, net.local_conv_list.parameters())) \
                        + list(map(id, net.fc_list.parameters())) 
        
        base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

        optimizer = optim.SGD([
            {'params': base_params, 'lr': 0.1 * args.lr},
            {'params': net.local_conv_list.parameters(), 'lr': args.lr},
            {'params': net.fc_list.parameters(), 'lr': args.lr}
            ],
            weight_decay=5e-4, momentum=0.9, nesterov=True)
    else:
        ignored_params = list(map(id, net.bottleneck.parameters())) \
                        + list(map(id, net.classifier.parameters())) 

        base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

        optimizer = optim.SGD([
            {'params': base_params, 'lr': 0.1 * args.lr},
            {'params': net.bottleneck.parameters(), 'lr': args.lr},
            {'params': net.classifier.parameters(), 'lr': args.lr}],
            weight_decay=5e-4, momentum=0.9, nesterov=True)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif epoch >= 10 and epoch < 20:
        lr = args.lr
    elif epoch >= 20 and epoch < 50:
        lr = args.lr * 0.1
    elif epoch >= 50:
        lr = args.lr * 0.01

    optimizer.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr

    return lr


def train(epoch, weight_hdmmd = args.hdmmd_weight, weight_source = args.source_weight):

    current_lr = adjust_learning_rate(optimizer, epoch)
    target_train_loss = AverageMeter()
    target_id_loss = AverageMeter()
    target_tri_loss = AverageMeter()
    target_batch_acc = 0
    target_correct = 0
    target_total = 0
    
    source_train_loss = AverageMeter()
    source_batch_acc = 0
    source_correct = 0
    source_total = 0
    
    train_loss = AverageMeter()
    batch_time = AverageMeter()
    
    # switch to train mode
    net.train()
    end = time.time()
    
    # debug:
    # torch.autograd.set_detect_anomaly(True)

    # Modify trainloader to consider source and target datasets
    # print(f"Loader lengths: {len(source_loader)}, {len(target_loader)}")
    target_loader_cycle = cycle(target_loader)  # Create a cyclic iterator for the target loader
    # if args.dataset == 'sysu':
    #     source_loader_cycle = cycle(source_loader)  # Create a cyclic iterator for the source loader
    #     target_loader_cycle = target_loader        
    # else:
    #     source_loader_cycle = source_loader
    #     target_loader_cycle = cycle(target_loader)  # Create a cyclic iterator for the target loader
        
    
    
    # Note that you could also make a common source+target loader which would cover all quadrplets
    # This cyclic loader optimization works fine because the source domain is much greater than the target domain (for RegDB)
    # However, SYSU could potentially benefit from covering all quadruplets since RegDB is inadequate for a source dataset
    
    
    # for batch_idx, ((source_input1, source_input2, source_label1, source_label2), 
    #                 (target_input1, target_input2, target_label1, target_label2)) in enumerate(zip(source_loader_cycle, target_loader_cycle)):
    for batch_idx, ((source_input1, source_input2, source_label1, source_label2), 
                    (target_input1, target_input2, target_label1, target_label2)) in enumerate(zip(source_loader, target_loader_cycle)):
        
        labels = torch.cat((source_label1, target_label1, source_label2, target_label2), 0)
        input1 = torch.cat((source_input1, target_input1), 0) # RGB
        input2 = torch.cat((source_input2, target_input2), 0) # IR
        
        input1 = Variable(input1.cuda())
        input2 = Variable(input2.cuda())

        labels = Variable(labels.cuda())
        
        if args.pcb ==  'on':
            feat, out0, feat_all = net(input1, input2)  
            loss_id = criterion_id(out0[0], labels)
            loss_tri_l, batch_acc = criterion_tri(feat[0], labels)
            for i in range(len(feat)-1):
                loss_id += criterion_id(out0[i+1], labels)
                loss_tri_l += criterion_tri(feat[i+1], labels)[0]
            loss_tri, batch_acc = criterion_tri(feat_all, labels)
            loss_tri += loss_tri_l * args.w_center  # 
            correct += batch_acc
            loss =  loss_id + loss_tri 
        else:
            feat, out0 = net(input1, input2)
            
            source_rgb_feat, target_rgb_feat, source_ir_feat, target_ir_feat = torch.split(feat, [source_label1.size(0),target_label1.size(0), source_label2.size(0),target_label2.size(0)], dim=0)
            source_rgb_labels, target_rgb_labels, source_ir_labels, target_ir_labels = torch.split(labels, [source_label1.size(0),target_label1.size(0), source_label2.size(0),target_label2.size(0)], dim=0)
                        
            # print(f"Shapes\nsource_rgb_feat: {source_rgb_feat.shape}, source_rgb_labels: {source_rgb_labels.shape}\n")
            
            source_feat = torch.cat((source_rgb_feat, source_ir_feat), dim=0)
            target_feat = torch.cat((target_rgb_feat, target_ir_feat), dim=0)
            
            source_labels = torch.cat((source_rgb_labels, source_ir_labels), dim=0)
            target_labels = torch.cat((target_rgb_labels, target_ir_labels), dim=0)
            
            # print(f"Shapes\nsource_feat: {source_feat.shape}, source_labels: {source_labels.shape}\n")
            
            source_rgb_out, target_rgb_out, source_ir_out, target_ir_out = torch.split(out0, [source_label1.size(0),target_label1.size(0), source_label2.size(0),target_label2.size(0)], dim=0)
            source_out = torch.cat((source_rgb_out, source_ir_out), dim=0)
            target_out = torch.cat((target_rgb_out, target_ir_out), dim=0)
            
            source_loss_id = criterion_id(source_out, source_labels)
            target_loss_id = criterion_id(target_out, target_labels)
            
            source_loss_tri, source_batch_acc = criterion_tri(source_feat, source_labels)
            target_loss_tri, target_batch_acc = criterion_tri(target_feat, target_labels)
            
            source_correct += (source_batch_acc / 2) # target
            _, source_predicted = source_out.max(1)
            source_correct += (source_predicted.eq(source_labels).sum().item() / 2)
            
            target_correct += (target_batch_acc / 2) # target
            _, target_predicted = target_out.max(1)
            target_correct += (target_predicted.eq(target_labels).sum().item() / 2)
            
            source_loss =  source_loss_id + source_loss_tri * args.w_center 
            target_loss =  target_loss_id + target_loss_tri * args.w_center
            
            if args.dist_disc == 'mmd':
                ## Apply Global MMD Loss on Pooling Layer
                source_loss_dist, _, _ = criterion_mmd(source_rgb_feat, source_ir_feat) ## Use Global MMD
                target_loss_dist, _, _ = criterion_mmd(target_rgb_feat, target_ir_feat) ## Use Global MMD
                
            elif args.dist_disc == 'margin_mmd':
                ## Apply Margin MMD-ID Loss on Pooling Layer
                source_loss_dist, _, _ = criterion_margin_mmd(source_rgb_feat, source_ir_feat) ## Use MMD-ID
                target_loss_dist, _, _ = criterion_margin_mmd(target_rgb_feat, target_ir_feat) ## Use MMD-ID

            elif args.dist_disc == 'cmd':
                source_loss_dist = criterion_cmd(source_rgb_feat, source_ir_feat) ## Use CMD
                target_loss_dist = criterion_cmd(target_rgb_feat, target_ir_feat) ## Use CMD
                
            if args.dist_disc is not None:
                source_loss = source_loss + source_loss_dist * args.dist_w ## Add Discrepancy Loss
                target_loss = target_loss + target_loss_dist * args.dist_w ## Add Discrepancy Loss

            # Direct DMMD loss
            # hdmmd_loss = sum(criterion_dmmd(source_feat, target_feat)) / 3 ## Use DMMD mean
            
            # HDMMD loss
            hdmmd_loss = criterion_hdmmd(source_rgb_feat, source_ir_feat, target_rgb_feat, target_ir_feat) ## Use DMMD mean
            # hdmmd_loss = 0

        
        loss = target_loss + weight_source * source_loss + weight_hdmmd * hdmmd_loss
        
        
        
        optimizer.zero_grad()        
        loss.backward()
        optimizer.step()

        # update P
        train_loss.update(loss.item(), 2 * input1.size(0))
        target_train_loss.update(target_loss.item(), input1.size(0))
        source_train_loss.update(source_loss.item(), input1.size(0))
        
        target_id_loss.update(target_loss_id.item(), input1.size(0))
        target_tri_loss.update(target_loss_tri, input1.size(0))
        
        target_total += input1.size(0)
        source_total += input2.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        iter = int(len(source_loader) / 2) # Assuming source > target
        if batch_idx % iter == 0:
            print('Epoch: [{}][{}/{}] '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'lr:{:.3f} '
                  'totLoss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                  'tarLoss: {target_train_loss.val:.4f} ({target_train_loss.avg:.4f}) '
                  'soLoss: {source_train_loss.val:.4f} ({source_train_loss.avg:.4f}) '
                  'hdmmd: {dmmd:.4f} '
                  'tarIDLoss: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
                  'tarTriLoss: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) '
                  'tarAccu: {:.2f} '
                  'soAccu: {:.2f} '.format(
                epoch, batch_idx, max(len(target_loader), len(source_loader)),
                current_lr,
                100. * target_correct / target_total,
                100. * source_correct / source_total,
                batch_time=batch_time, train_loss=train_loss, target_train_loss=target_train_loss,
                source_train_loss=source_train_loss, hdmmd=hdmmd_loss, id_loss=target_id_loss, tri_loss=target_tri_loss)) 

    writer.add_scalar('total_loss', train_loss.avg, epoch)
    writer.add_scalar('target_id_loss', target_id_loss.avg, epoch)
    writer.add_scalar('target_tri_loss', target_tri_loss.avg, epoch)
    writer.add_scalar('lr', current_lr, epoch)


def test(epoch):
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    if args.pcb == 'on':
        feat_dim = args.num_strips * args.local_feat_dim
    else:
        feat_dim = 2048
    gall_feat = np.zeros((ngall, feat_dim))
    gall_feat_att = np.zeros((ngall, feat_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            if args.pcb == 'on':
                feat = net(input, input, test_mode[0])
                gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            else:
                feat, feat_att = net(input, input, test_mode[0])
                gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                gall_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0

    query_feat = np.zeros((nquery, feat_dim))
    query_feat_att = np.zeros((nquery, feat_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            if args.pcb == 'on':
                feat = net(input, input, test_mode[1])
                query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            else:
                feat, feat_att = net(input, input, test_mode[1])
                query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                query_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()

    
    if args.re_rank == 'random_walk':
        distmat = random_walk(query_feat, gall_feat)
        if args.pcb == 'off': distmat_att = random_walk(query_feat_att, gall_feat_att) 
    elif args.re_rank == 'k_reciprocal':
        distmat = k_reciprocal(query_feat, gall_feat)
        if args.pcb == 'off': distmat_att = k_reciprocal(query_feat_att, gall_feat_att)
    elif args.re_rank == 'no':
        # compute the similarity
        distmat = -np.matmul(query_feat, np.transpose(gall_feat))
        if args.pcb == 'off': distmat_att = -np.matmul(query_feat_att, np.transpose(gall_feat_att))

    # evaluation
    if dataset == 'regdb':
        cmc, mAP, mINP      = eval_regdb(distmat, query_label, gall_label)
        if args.pcb == 'off': cmc_att, mAP_att, mINP_att  = eval_regdb(distmat_att, query_label, gall_label)
    elif dataset == 'sysu':
        
        # regdb testset for DA
        # cmc, mAP, mINP      = eval_regdb(distmat, query_label, gall_label)
        # if args.pcb == 'off': cmc_att, mAP_att, mINP_att  = eval_regdb(distmat_att, query_label, gall_label)
        
        cmc, mAP, mINP = eval_sysu(distmat, query_label, gall_label, query_cam, gall_cam)
        if args.pcb == 'off': cmc_att, mAP_att, mINP_att = eval_sysu(distmat_att, query_label, gall_label, query_cam, gall_cam)
    
    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

    writer.add_scalar('rank1', cmc[0], epoch)
    writer.add_scalar('mAP', mAP, epoch)
    writer.add_scalar('mINP', mINP, epoch)
    if args.pcb == 'off':
        writer.add_scalar('rank1_att', cmc_att[0], epoch)
        writer.add_scalar('mAP_att', mAP_att, epoch)
        writer.add_scalar('mINP_att', mINP_att, epoch)
        
        return cmc, mAP, mINP, cmc_att, mAP_att, mINP_att
    else:
        return cmc, mAP, mINP

weights = [0.1, 0.5, 1]
for wd in weights:
    print(f"\n\n\n------------ Now training for weight: {wd} -------------\n\n")
    # training
    print('==> Start Training...')
    
    if args.method =='base':
        net = embed_net(n_class, no_local= 'off', gm_pool =  'on', arch=args.arch, share_net=args.share_net, pcb=args.pcb, local_feat_dim=args.local_feat_dim, num_strips=args.num_strips)
    elif args.method=='agw':
        # net = embed_net(n_class, no_local= 'on', gm_pool = 'on', arch=args.arch,  share_net=args.share_net, pcb=args.pcb)
        net = agw(n_class)
    net.to(device)
    
    if args.optim == 'sgd':
        if args.pcb == 'on':
            ignored_params = list(map(id, net.local_conv_list.parameters())) \
                            + list(map(id, net.fc_list.parameters())) 
            
            base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

            optimizer = optim.SGD([
                {'params': base_params, 'lr': 0.1 * args.lr},
                {'params': net.local_conv_list.parameters(), 'lr': args.lr},
                {'params': net.fc_list.parameters(), 'lr': args.lr}
                ],
                weight_decay=5e-4, momentum=0.9, nesterov=True)
        else:
            ignored_params = list(map(id, net.bottleneck.parameters())) \
                            + list(map(id, net.classifier.parameters())) 

            base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

            optimizer = optim.SGD([
                {'params': base_params, 'lr': 0.1 * args.lr},
                {'params': net.bottleneck.parameters(), 'lr': args.lr},
                {'params': net.classifier.parameters(), 'lr': args.lr}],
                weight_decay=5e-4, momentum=0.9, nesterov=True)
            
    # load_source_trained_model()
    
    weight_best_r1 = 0
    weight_best_epoch = -1
            
    for epoch in range(start_epoch, 47 - start_epoch):

        print('==> Preparing Data Loader...')
        print(epoch)

        loader_batch = args.batch_size * args.num_pos
        
        # Identity sampler for source dataset
        source_sampler = IdentitySampler(source_trainset.train_color_label,
                                        source_trainset.train_thermal_label,
                                        source_color_pos, source_thermal_pos,
                                        args.num_pos, args.batch_size,
                                        epoch)
        
        # Identity sampler for target dataset
        target_sampler = IdentitySampler(target_trainset.train_color_label,
                                        target_trainset.train_thermal_label,
                                        target_color_pos, target_thermal_pos,
                                        args.num_pos, args.batch_size,
                                        epoch, offset=len(np.unique(source_trainset.train_color_label)))
        
        source_trainset.cIndex = source_sampler.index1  # color index
        source_trainset.tIndex = source_sampler.index2  # thermal index
        
        target_trainset.cIndex = target_sampler.index1  # color index
        target_trainset.tIndex = target_sampler.index2  # thermal index
        
        # Data loaders for source and target datasets
        source_loader = data.DataLoader(source_trainset, batch_size=loader_batch,
                                        sampler=source_sampler, num_workers=args.workers,
                                        drop_last=True)
        
        target_loader = data.DataLoader(target_trainset, batch_size=loader_batch,
                                        sampler=target_sampler, num_workers=args.workers,
                                        drop_last=True)

        # training
        # train(epoch, weight_source=ws)
        
        start_time = time.time()

        train(epoch, weight_hdmmd=wd)
        # train(epoch)
        
        end_time = time.time()
        execution_time = end_time - start_time
        print("Epoch Execution time:", execution_time)    

        if epoch % 2 == 0: 
            print('Test Epoch: {}'.format(epoch))

            # testing
            if args.pcb == 'off': 
                cmc, mAP, mINP, cmc_fc, mAP_fc, mINP_fc = test(epoch)
            else:
                cmc_fc, mAP_fc, mINP_fc = test(epoch)
            # save model
            if cmc_fc[0] > best_acc:  # not the real best for sysu-mm01
                best_acc = cmc_fc[0]
                best_epoch = epoch
                best_mAP = mAP_fc
                best_mINP = mINP_fc
                state = {
                    'net': net.state_dict(),
                    'cmc': cmc_fc,
                    'mAP': mAP_fc,
                    'mINP': mINP_fc,
                    'epoch': epoch,
                }
                print("Saved model at path: " + checkpoint_path + suffix + '_best.t')
                torch.save(state, checkpoint_path + suffix + '_best.t')
                
            if cmc_fc[0] > weight_best_r1:
                weight_best_r1 = cmc_fc[0]
                weight_best_epoch = epoch

            if args.pcb == 'off': 
                print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
            
            print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_fc[0], cmc_fc[4], cmc_fc[9], cmc_fc[19], mAP_fc, mINP_fc))
            print('Best Epoch [{}], Rank-1: {:.2%} |  mAP: {:.2%}| mINP: {:.2%}'.format(best_epoch, best_acc, best_mAP, best_mINP))
            
    print('~~~ For weight {} ~~~\n\nBest Epoch [{}], Rank-1: {:.2%}'.format(wd, weight_best_epoch, weight_best_r1))
    