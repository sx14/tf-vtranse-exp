#data for relationship detection
#need vrd_roidb.npz and vrd_detected_box.npz
import pickle
import numpy as np
import scipy.io
from model.config import cfg
from model.ass_fun import *

from model.hier.vg.obj_hier import objnet


def read_vg_det(det_roidb, test_roidb):
    raw_labels = objnet.get_raw_labels()
    N_test = len(test_roidb)
    test_det_roidb = []
    for i in range(N_test):
        test_roidb_use = test_roidb[i]
        img_id = test_roidb_use['image'].split('/')[-1].split('.')[0]
        det_roidb_use = det_roidb[img_id]
        # box, cls, conf
        for i in range(det_roidb_use.shape[0]):
            h_ind = det_roidb_use[i, 4]
            h_node = objnet.get_node_by_index(int(h_ind))
            find = False
            for ri, raw_label in enumerate(raw_labels):
                if raw_label == h_node.name():
                    find = True
                    break
            assert find
            det_roidb_use[i, 4] = ri

        test_det_roidb.append(det_roidb_use)
    return test_det_roidb



N_each_batch = cfg.VRD_BATCH_NUM_RELA
N_each_pair = cfg.VRD_AU_PAIR
iou_l = cfg.VRD_IOU_TRAIN
roidb_path = cfg.DIR + 'vtranse/input/vg_roidb.npz'
detected_box_path = 'det_roidb_vg.bin'
save_path = cfg.DIR + 'vtranse/input/vg_rela_roidb.npz'

roidb_read = read_roidb(roidb_path)
test_roidb = roidb_read['test_roidb']

with open(detected_box_path, 'rb') as f:
    det_roidb = pickle.load(f)
test_detected_box = read_vg_det(det_roidb, test_roidb)

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