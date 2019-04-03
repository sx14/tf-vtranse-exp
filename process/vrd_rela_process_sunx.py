#data for relationship detection
#need vrd_roidb.npz and vrd_detected_box.npz
import numpy as np
import scipy.io
from model.config import cfg
from model.ass_fun import *


def read_vrd_det(test_roidb):
    det_mat = scipy.io.loadmat('objectDetRCNN.mat')
    det_boxes = det_mat['detection_bboxes'][0]
    det_labels = det_mat['detection_labels'][0]
    det_confs = det_mat['detection_confs'][0]

    img_paths_mat = scipy.io.loadmat('imagePath.mat')
    img_paths = img_paths_mat['imagePath'][0]

    det_roidb = {}
    N_det = 0.0
    for i in range(1000):
        img_path = img_paths[i][0]
        img_det_boxes = det_boxes[i]
        img_det_labels = det_labels[i]
        img_det_confs = det_confs[i]

        # x1, y1, x2 ,y2, cls, conf
        img_dets = []
        for j in range(img_det_boxes.shape[0]):
            N_det += 1
            box = img_det_boxes[j]
            det = box.tolist()

            label = img_det_labels[j, 0] -1
            det.append(label)

            conf = img_det_confs[j, 0]
            det.append(conf)

            img_dets.append(det)

        img_id = img_path.split('.')[0]
        det_roidb[img_id] = np.array(img_dets)

    N_test = len(test_roidb)
    for i in range(N_test):
        test_roidb_use = test_roidb[i]
        img_id = test_roidb_use['image'].split('/')[-1].split('.')[0]
        test_dets = det_roidb[img_id]
    return test_dets



N_each_batch = cfg.VRD_BATCH_NUM_RELA
N_each_pair = cfg.VRD_AU_PAIR
iou_l = cfg.VRD_IOU_TRAIN
roidb_path = cfg.DIR + 'vtranse/input/vrd_roidb.npz'
save_path = cfg.DIR + 'vtranse/input/vrd_rela_roidb.npz'

roidb_read = read_roidb(roidb_path)
test_roidb = roidb_read['test_roidb']
test_detected_box = read_vrd_det()

N_test = len(test_roidb)


roidb = []
for i in range(N_test):
	if (i+1)%100 == 0:
		print(i+1)
	test_roidb_use = test_roidb[i]
	test_detected_box_use = test_detected_box[i]
	roidb_temp = generate_test_rela_roidb(test_roidb_use, test_detected_box_use, N_each_batch)
	roidb.append(roidb_temp)
test_roidb_new = roidb

roidb = {}
roidb['test_roidb'] = test_roidb_new
np.savez(save_path, roidb=roidb)