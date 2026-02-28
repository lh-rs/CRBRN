import cv2
import torch
import numpy as np
from user_cd.dataload import getRGBData, getP0Data
from models.MCAE_CBAM_Cross import MCAE, clNet
from user_cd.evaluate import evaluate, metric
from models.loss import  Smooth_contrastive
import xlwt
import matplotlib.pyplot as plt
import os
import random
def save_matrix_heatmap_visual(similar_distance_map,save_change_map_dir):
    from matplotlib import cm
    cmap = cm.get_cmap('jet', 30)
    plt.set_cmap(cmap)
    plt.imsave(save_change_map_dir,similar_distance_map)

def showresult(data):
    data = np.expand_dims(data, -1)
    result = np.concatenate([data, data, data], -1)
    return result

def check_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
def data_Norm(data, norm=True):
    if norm:
        # data = data.detach().cpu().numpy()
        min = np.amin(data)
        max = np.amax(data)
        result = (data - min) / (max - min)
    return result

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    for dataset in ['#4Gloucester']:
    # for dataset in ['Tianhe', 'ShuguangVillage2', '#4Gloucester','Italy']:
        if dataset == 'Tianhe':
            setup_seed(2026)
            alpha = 0.15
            img1_path = 'dataset/Tianhe/ST_Regression.jpg'
            pse_path = 'dataset/Tianhe/ST_br.jpg'
            img2_path = 'dataset/Tianhe/im2.bmp'
            ref_path = 'dataset/Tianhe/im3.bmp'

        elif dataset == 'ShuguangVillage2':
            setup_seed(2021)
            alpha = 0.4
            img1_path = 'dataset/ShuguangVillage2/ST_Regression.jpg'
            pse_path = 'dataset/ShuguangVillage2/ST_br.jpg'
            img2_path = 'dataset/ShuguangVillage2/im2.png'
            ref_path = 'dataset/ShuguangVillage2/im3.png'

        elif dataset == '#4Gloucester':
            setup_seed(2019)
            alpha = 0.01
            img1_path = 'dataset/#4Gloucester/ST_Regression.jpg'
            pse_path = 'dataset/#4Gloucester/ST_br.jpg'
            img2_path = 'dataset/#4Gloucester/im2.jpg'
            ref_path = 'dataset/#4Gloucester/im3.jpg'

        elif dataset == 'Italy':
            setup_seed(2022)
            alpha = 0.09
            img1_path = 'dataset/Italy/ST_Regression.jpg'
            pse_path = 'dataset/Italy/ST_br.jpg'
            img2_path = 'dataset/Italy/im2.bmp'
            ref_path = 'dataset/Italy/im3.bmp'

        result_path0 = 'results' + '/' + dataset + '/'

        result_path = os.path.join(result_path0)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        check_dir(result_path)

        Channels = 20

        img1_data = getRGBData(img1_path)
        img2_data = getRGBData(img2_path)

        cva_ref_data = cv2.imread(pse_path)[..., 0]

        ref_data = cv2.imread(ref_path)[..., 0]
        sz_img1 = img1_data.shape
        sz_img2 = img2_data.shape

        N = sz_img1[1] * sz_img1[2] * sz_img1[3]
        img1_data = img1_data.cuda()
        img2_data = img2_data.cuda()

        # Randomly initializing
        pcx = torch.rand(sz_img1[1], sz_img1[2], sz_img1[3]).cuda()
        print('Randomly initializing Pc: {}'.format(torch.sum(pcx) / pcx.numel()))

        model = MCAE(sz_img1[1], Channels, sz_img2[1], Channels).cuda()
        net = clNet(2 * Channels, 1).cuda()

        SCL = Smooth_contrastive().cuda()
        optimizer = torch.optim.RMSprop([{'params': model.parameters(), 'lr': 0.001},
                                         {'params': net.parameters(), 'lr': 0.001}],
                                        )
        epochs = 50
        iters = 100

        model.train()
        for epoch in range(epochs):
            if epoch == 0:
                pse_path = pse_path
            else:
                pse_path = result_path + str(epoch - 1) + '_' + 'bestresult.jpg'
            pse_data = getP0Data(pse_path).unsqueeze(0).cuda()

            loss_best = 9999999
            for iter in range(iters):
                F1_1, f1_2, f1_3, F2_1, f2_2, f2_3 = model(img1_data, img2_data)
                diff = torch.sqrt(torch.sum((F1_1 - F2_1) ** 2, dim=1))

                unchanged_weithts = pse_data.sum() / (sz_img1[2] * sz_img1[3])
                changed_weithts = 1 - unchanged_weithts

                weight = pse_data * 1 + (1 - pse_data) * 0.1

                loss_1 = SCL(diff, pcx, alpha)

                pos_fea = torch.sqrt((F1_1 - F2_1) ** 2 + 1e-15)

                pre_data = net(pos_fea)

                loss_2 = torch.mean((pre_data - pse_data) ** 2 * weight)

                info = ''
                if iter <= 30:
                    loss = loss_2 + 10
                else:
                    loss = loss_1 + 5 * loss_2

                if loss_best > loss:
                    loss_best = loss
                    pre_data_best = pre_data
                    diff_best = diff
                    info = '\tsave best results....'
                else:
                    info = ' '
                print('epoch:{},iter: {}, diff:{}, pcx:{}, total_loss: {}, CL:{}, classify:{} {}'.
                      format(epoch, iter, torch.sum(diff) / diff.numel(), torch.sum(pcx) / pcx.numel(), loss, loss_1,
                             loss_2, info))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            pre_data = pre_data_best.squeeze(0).detach()
            pcx = pre_data

            img1 = diff.squeeze().cpu().detach().numpy()
            img2 = pcx.squeeze().cpu().detach().numpy()

            last_pcx_1 = pcx if epoch == 0 else last_pcx_1 + pcx
            img2_1 = (last_pcx_1 / (epoch + 1)).squeeze().detach().cpu().numpy()

            Fbefore = 0
            thresh = np.array(list(range(80, 200))) / 255
            best_th = 0
            for th in thresh:
                seclect_result = np.where(img2_1 > th, 255, 0)
                last_pc = cv2.imread(pse_path)[...,0]
                F0 = 1 / np.sum((seclect_result - last_pc) ** 2 + 1e-15)
                if F0 > Fbefore:
                    cv2.imwrite(result_path + str(epoch) + '_' + 'bestresult.jpg', showresult(seclect_result))
                    Fbefore = F0
                    best_th = th
            print('\nbest_th=======================', best_th)

            #evaluate
            i = 1
            wk = xlwt.Workbook()
            ws = wk.add_sheet('analysis')
            ws.write(0, 0, "epoch")
            ws.write(0, 1, "alpha")
            ws.write(0, 2, "FP")
            ws.write(0, 3, "FN")
            ws.write(0, 4, "OE")
            ws.write(0, 5, "PCC")
            ws.write(0, 6, "Kappa")
            ws.write(0, 7, "F")
            ws.write(0, 9, "acc_un")
            ws.write(0, 10, "acc_chg")
            ws.write(0, 11, "acc_all")
            ws.write(0, 12, "acc_tp")
            bestresult = cv2.imread(result_path + str(epoch) + '_' + 'bestresult.jpg')[..., 0]
            FP, FN, OE, FPR, FNR, OER, PCC, Kappa, F = evaluate(bestresult, ref_data)
            acc_un, acc_chg, acc_all, acc_tp = metric(bestresult, ref_data)

            ws.write(i, 0, epoch)
            ws.write(i, 1, alpha)
            ws.write(i, 2, FP)
            ws.write(i, 3, FN)
            ws.write(i, 4, OE)
            ws.write(i, 5, PCC)
            ws.write(i, 6, Kappa)
            ws.write(i, 7, F)
            ws.write(i, 9, acc_un)
            ws.write(i, 10, acc_chg)
            ws.write(i, 11, acc_all)
            ws.write(i, 12, acc_tp)

            i += 1

            wk.save(result_path + dataset + '_Eval.xls')
