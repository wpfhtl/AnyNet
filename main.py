import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import time
from dataloader import listflowfile as lt
from dataloader import SecenFlowLoader as DA
import utils.logger as logger

import models.anynet
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import math

parser = argparse.ArgumentParser(description='AnyNet with Flyingthings3d')
parser.add_argument('--maxdisp', type=int, default=192, help='maxium disparity')
parser.add_argument('--loss_weights', type=float, nargs='+', default=[0.25, 0.5, 1., 1.])
parser.add_argument('--maxdisplist', type=int, nargs='+', default=[12, 3, 3])
parser.add_argument('--datapath', default='dataset/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train')
parser.add_argument('--train_bsize', type=int, default=16,
                    help='batch size for training (default: 12)')
parser.add_argument('--test_bsize', type=int, default=1,
                    help='batch size for testing (default: 8)')
parser.add_argument('--save_path', type=str, default='results/pretrained_anynet/',
                    help='the path of saving checkpoints and log')
# parser.add_argument('--resume', type=str, default=None,
#                     help='resume path')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='learning rate')
parser.add_argument('--with_spn', action='store_true', help='with spn network or not')
parser.add_argument('--print_freq', type=int, default=5, help='print frequence')
parser.add_argument('--init_channels', type=int, default=1, help='initial channels for 2d feature extractor')
parser.add_argument('--nblocks', type=int, default=2, help='number of layers in each stage')
parser.add_argument('--channels_3d', type=int, default=4, help='number of initial channels of the 3d network')
parser.add_argument('--layers_3d', type=int, default=4, help='number of initial layers of the 3d network')
parser.add_argument('--growth_rate', type=int, nargs='+', default=[4,1,1], help='growth rate in the 3d network')
parser.add_argument('--spn_init_channels', type=int, default=8, help='initial channels for spnet')
parser.add_argument('--gpuid', type=str, default='0', help='the id of gpu to use')
parser.add_argument('--start_epoch', type=int, default=1, help='start epoch')
parser.add_argument('--loadmodel', default=None, help='load model')
parser.add_argument('--trainfull', type=int, default=1, help='train all parameters or partially(default: 1)')
parser.add_argument('--fixnum', type=int, default=0, help='number of parameters to be fixed')


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
gpuid = args.gpuid
print("use gpu {}".format(gpuid))

log_comment = '_sf'
log_comment += '_b' + str(args.train_bsize)
writer = SummaryWriter(comment=log_comment)

def main():
    global args

    train_left_img, train_right_img, train_left_disp, test_left_img, test_right_img, test_left_disp = lt.dataloader(
        args.datapath)

    TrainImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(train_left_img, train_right_img, train_left_disp, True),
        batch_size=args.train_bsize, shuffle=True, num_workers=4, drop_last=False)

    TestImgLoader = torch.utils.data.DataLoader(
        DA.myImageFloder(test_left_img, test_right_img, test_left_disp, False),
        batch_size=args.test_bsize, shuffle=False, num_workers=4, drop_last=False)

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    # log = logger.setup_logger(args.save_path + '/training.log')
    # for key, value in sorted(vars(args).items()):
    #     log.info(str(key) + ': ' + str(value))

    model = models.anynet.AnyNet(args)
    model = nn.DataParallel(model).cuda()
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    # log.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    num_pretrain_items = 0
    num_model_items = 0
    if args.loadmodel is not None:
        pretrained_dict = torch.load(args.loadmodel)
        # start_epoch = pretrained_dict['epoch'] + 1
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict['state_dict'].items() if k in model_dict}
        num_pretrain_items = len(pretrained_dict.items())
        num_model_items = len(model_dict.items())
        print('Number of pretrained items: {:d}'.format(num_pretrain_items))
        print('Number of model items: {:d}'.format(num_model_items))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        # state_dict = torch.load(args.loadmodel)
        # model.load_state_dict(state_dict['state_dict'])
    else:
        start_epoch = 1
        model_dict = model.state_dict()
        num_model_items = len(model_dict.items())
        print('Number of model items: {:d}'.format(num_model_items))

    if args.start_epoch is not 1:
        start_epoch = args.start_epoch
    else:
        start_epoch = 1
    print(model)
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if args.trainfull:
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    else:
        for i, p in enumerate(model.parameters()):
            print(i, p.shape)
            if i < args.fixnum:
                p.requires_grad = False
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, betas=(0.9, 0.999))


    # args.start_epoch = 0
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         log.info("=> loading checkpoint '{}'".format(args.resume))
    #         checkpoint = torch.load(args.resume)
    #         args.start_epoch = checkpoint['epoch']
    #         model.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         log.info("=> loaded checkpoint '{}' (epoch {})"
    #                  .format(args.resume, checkpoint['epoch']))
    #     else:
    #         log.info("=> no checkpoint found at '{}'".format(args.resume))
    #         log.info("=> Will start from scratch.")
    # else:
    #     log.info('Not Resume')

    train_step = 0
    test_step = 0
    start_full_time = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        # log.info('This is {}-th epoch'.format(epoch))
        print('This is {}-th epoch'.format(epoch))

        # train(TrainImgLoader, model, optimizer, log, epoch)
        train_losses, train_step = train(TrainImgLoader, model, optimizer, epoch, train_step)
        test_losses, test_step = test(TestImgLoader, model, epoch, test_step)

        savefilename = args.save_path + 'sf_' + str(epoch) + '.tar'
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, savefilename)

    # test(TestImgLoader, model, log)
    # log.info('full training time = {:.2f} Hours'.format((time.time() - start_full_time) / 3600))
    print('full training time = %.2f HR' % ((time.time() - start_full_time) / 3600))


# def train(dataloader, model, optimizer, log, epoch=1):
def train(dataloader, model, optimizer, epoch=1, train_step=0):

    stages = 3 + args.with_spn
    losses = [AverageMeter() for _ in range(stages)]
    length_loader = len(dataloader)

    model.train()

    for batch_idx, (imgL, imgR, disp_L) in enumerate(dataloader):
        train_step += 1
        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        disp_L = disp_L.float().cuda()

        optimizer.zero_grad()
        mask = disp_L < args.maxdisp
        mask.detach_()
        outputs = model(imgL, imgR)
        outputs = [torch.squeeze(output, 1) for output in outputs]
        loss = [args.loss_weights[x] * F.smooth_l1_loss(outputs[x][mask], disp_L[mask], size_average=True)
                for x in range(stages)]
        if True in [math.isnan(loss[x]) for x in range(stages)]:
            print('catched nan')
        sum(loss).backward()
        optimizer.step()

        for idx in range(stages):
            losses[idx].update(loss[idx].item()/args.loss_weights[idx])
        if stages == 4:
            writer.add_scalars('train_batch', {"error_stage_0": losses[0].val,
                                               "error_stage_1": losses[1].val,
                                               "error_stage_2": losses[2].val,
                                               "error_stage_3": losses[3].val}, train_step)
        else:
            writer.add_scalars('train_batch', {"error_stage_0": losses[0].val,
                                               "error_stage_1": losses[1].val,
                                               "error_stage_2": losses[2].val}, train_step)
        if not batch_idx % args.print_freq:
            info_str = ['Stage {} = {:.2f}({:.2f})'.format(x, losses[x].val, losses[x].avg) for x in range(stages)]
            info_str = '\t'.join(info_str)

    #         log.info('Epoch{} [{}/{}] {}'.format(
    #             epoch, batch_idx, length_loader, info_str))
            print('Epoch{} [{}/{}] {}'.format(epoch, batch_idx, length_loader, info_str))
    info_str = '\t'.join(['Stage {} = {:.2f}'.format(x, losses[x].avg) for x in range(stages)])
    # log.info('Average train loss = ' + info_str)
    print('Average train loss = ' + info_str)
    if stages == 4:
        writer.add_scalars('train_epoch', {"error_stage_0": losses[0].avg,
                                           "error_stage_1": losses[1].avg,
                                           "error_stage_2": losses[2].avg,
                                           "error_stage_3": losses[3].avg}, epoch)
    else:
        writer.add_scalars('train_epoch', {"error_stage_0": losses[0].avg,
                                           "error_stage_1": losses[1].avg,
                                           "error_stage_2": losses[2].avg}, epoch)
    return losses, train_step


# def test(dataloader, model, log):
def test(dataloader, model, epoch, test_step):

    stages = 3 + args.with_spn
    EPEs = [AverageMeter() for _ in range(stages)]
    length_loader = len(dataloader)

    model.eval()

    for batch_idx, (imgL, imgR, disp_L) in enumerate(dataloader):
        test_step += 1
        imgL = imgL.float().cuda()
        imgR = imgR.float().cuda()
        disp_L = disp_L.float().cuda()

        mask = disp_L < args.maxdisp
        with torch.no_grad():
            outputs = model(imgL, imgR)
            for x in range(stages):
                if len(disp_L[mask]) == 0:
                    EPEs[x].update(0)
                    continue
                output = torch.squeeze(outputs[x], 1)
                output = output[:, 4:, :]
                EPEs[x].update((output[mask] - disp_L[mask]).abs().mean())

        if stages == 4:
            writer.add_scalars('test_batch', {"error_stage_0": EPEs[0].val,
                                              "error_stage_1": EPEs[1].val,
                                              "error_stage_2": EPEs[2].val,
                                              "error_stage_3": EPEs[3].val}, test_step)
        else:
            writer.add_scalars('test_batch', {"error_stage_0": EPEs[0].val,
                                              "error_stage_1": EPEs[1].val,
                                              "error_stage_2": EPEs[2].val}, test_step)
        info_str = '\t'.join(['Stage {} = {:.2f}({:.2f})'.format(x, EPEs[x].val, EPEs[x].avg) for x in range(stages)])
    #
    #     log.info('[{}/{}] {}'.format(
    #         batch_idx, length_loader, info_str))
        print('[{}/{}] {}'.format(batch_idx, length_loader, info_str))
    #
    info_str = ', '.join(['Stage {}={:.2f}'.format(x, EPEs[x].avg) for x in range(stages)])
    # log.info('Average test EPE = ' + info_str)
    print('Average test EPE = ' + info_str)
    if stages == 4:
        writer.add_scalars('test_epoch', {"error_stage_0": EPEs[0].avg,
                                          "error_stage_1": EPEs[1].avg,
                                          "error_stage_2": EPEs[2].avg,
                                          "error_stage_3": EPEs[3].avg}, epoch)
    else:
        writer.add_scalars('test_epoch', {"error_stage_0": EPEs[0].avg,
                                          "error_stage_1": EPEs[1].avg,
                                          "error_stage_2": EPEs[2].avg}, epoch)
    return EPEs, test_step

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()
