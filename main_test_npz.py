
import torch
# print(torch.__version__)
import os
from config import Config
opt = Config('training.yml')
gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
torch.backends.cudnn.benchmark = True
import cv2
from model.model import *

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import time
import numpy as np
import utils
from dataset_RGB import *

from U_model import unet
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR

######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)


def main():
    start_epoch = 1
    session = opt.MODEL.SESSION

    result_dir = os.path.join(opt.TRAINING.SAVE_DIR, 'results', session)
    model_dir = os.path.join(opt.TRAINING.SAVE_DIR, 'models', session)

    utils.mkdir(result_dir)
    utils.mkdir(model_dir)


    ######### Model ###########

    model_restoration = unet.Restoration(3, 6, 3,opt)

    # print(model_restoration)
    model_restoration.cuda()

    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 1:
        print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")

    new_lr = opt.OPTIM.LR_INITIAL

    optimizer = optim.Adam(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)

    ######### Scheduler ###########
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.OPTIM.NUM_EPOCHS - warmup_epochs,
                                                            eta_min=opt.OPTIM.LR_MIN)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                       after_scheduler=scheduler_cosine)
    scheduler.step()

    ######### Resume ###########
    if opt.TRAINING.RESUME:
        path_chk_rest = utils.get_last_path(model_dir, '_epoch_317.pth')

        print('path_chk_rest', path_chk_rest)
        utils.load_checkpoint(model_restoration, path_chk_rest[0])
        start_epoch = utils.load_start_epoch(path_chk_rest[0]) + 1

        utils.load_optim(optimizer, path_chk_rest[0])

        for i in range(0, start_epoch):
            scheduler.step()
        new_lr = scheduler.get_lr()[0]
        print('------------------------------------------------------------------------------')
        print("==> Resuming Training with learning rate:", new_lr)
        print('------------------------------------------------------------------------------')

    if len(device_ids) > 1:
        model_restoration = nn.DataParallel(model_restoration, device_ids=device_ids)

    ##data prepare
    test_files_dirs=sorted(os.listdir(os.path.join(opt.father_test_path_npy, 'blur')))

    ######### DataLoaders ###########


    print('===> Start Epoch {} End Epoch {}'.format(start_epoch, opt.OPTIM.NUM_EPOCHS + 1))
    print('===> Loading datasets')

    epoch=0
    #### Evaluation ####
    if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
        model_restoration.eval()
        psnr_val_rgb = []
        ssim_val_rgb = []

        for val_file in test_files_dirs:

            single_psnr_val_rgb = []
            single_ssim_val_rgb = []
            out_path = os.path.join(opt.result_dir, val_file)
            isExists = os.path.exists(out_path)
            if not isExists:
                os.makedirs(out_path)

            val_dataset = DataLoaderTest_npz(opt.father_test_path_npz, val_file,opt)
            val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False,
                                    pin_memory=True)
            for ii, data_val in enumerate(tqdm(val_loader), 0):
                input_img = data_val[0].cuda()  ##1xWxH
                input_event = data_val[1].cuda()  ##10xWxH
                input_target = data_val[2].cuda()  ##5xWxH

                with torch.no_grad():
                    restored,en_img,out_common_img,out_differential_img = model_restoration(input_img, input_event)  ###[8,6,3, 720, 1280]


                res = torch.clamp(restored, 0, 1)[0, :, :, :]  ##
                tar=input_target[0, :, :, :]
                input1 = res.cpu().numpy().transpose([1, 2, 0])
                input2 = tar.cpu().numpy().transpose([1, 2, 0])

                ssim_rgb = SSIM(input1, input2, multichannel=True)
                single_ssim_val_rgb.append(ssim_rgb)
                ssim_val_rgb.append(ssim_rgb)

                psnr_rgb = PSNR(input1, input2)
                single_psnr_val_rgb.append(psnr_rgb)
                psnr_val_rgb.append(psnr_rgb)

                output = restored[0, :, :, :] * 255
                output.clamp_(0.0, 255.0)
                output = output.byte()
                output = output.cpu().numpy()
                output = output.transpose([1, 2, 0])  # height * width * channel

                cv2.imwrite((os.path.join(out_path, str(ii).rjust(4, '0') + '.png')), output)

                # torch.cuda.empty_cache()


            print("Name: %s PSNR: %.4f SSIM: %.4f" % (val_file,np.mean(single_psnr_val_rgb),np.mean(single_ssim_val_rgb)))


        ssim_val_rgb = np.mean(ssim_val_rgb)
        psnr_val_rgb = np.mean(psnr_val_rgb)

    print('ALL_SSIM', np.mean(ssim_val_rgb))
    print('ALL_PSNR', np.mean(psnr_val_rgb))



#
if __name__ == '__main__':
    main()


