#data for relationship detection
#need vrd_roidb.npz and vrd_detected_box.npz
import pickle
import numpy as np
import scipy.io
from model.config import cfg
from model.ass_fun import *





N_each_batch = cfg.VRD_BATCH_NUM_RELA
N_each_pair = cfg.VRD_AU_PAIR
iou_l = cfg.VRD_IOU_TRAIN
roidb_path = cfg.DIR + 'vtranse/input/vg_roidb.npz'
detected_box_path = 'det_roidb_vg.bin'
save_path = cfg.DIR + 'vtranse/input/vg_rela_roidb.npz'

roidb_read = read_roidb(roidb_path)
test_roidb = roidb_read['test_roidb']

with open(detected_box_path, 'rb') as f:
    test_detected_box = pickle.load(f)

N_test = len(test_roidb)


roidb = []
for i in range(N_test):
    if (i+1)%100 == 0:
        print(i+1)
    test_roidb_use = test_roidb[i]
    test_detected_box_use = test_detected_box[i]
    if len(test_detected_box_use) == 0:
        test_detected_box_use = np.zeros((1, 6))
    roidb_temp = generate_test_rela_roidb(test_roidb_use, test_detected_box_use, N_each_batch)
    roidb.append(roidb_temp)
test_roidb_new = roidb

roidb = {}
roidb['test_roidb'] = test_roidb_new
np.savez(save_path, roidb=roidb)