import pickle
import numpy as np
from model.config import cfg



def get_predicate_box(sbj_box, obj_box):
    x1 = np.min((sbj_box[:, 0:1], obj_box[:, 0:1]), axis=0)
    y1 = np.min((sbj_box[:, 1:2], obj_box[:, 1:2]), axis=0)
    x2 = np.max((sbj_box[:, 2:3], obj_box[:, 2:3]), axis=0)
    y2 = np.max((sbj_box[:, 3:4], obj_box[:, 3:4]), axis=0)
    return np.concatenate((x1, y1, x2, y2), axis=1)


dataset = 'vg'
if dataset == 'vrd':
    from model.hier.vrd.pre_hier import prenet
    from model.hier.vrd.obj_hier import objnet
else:
    from model.hier.vg.pre_hier import prenet
    from model.hier.vg.obj_hier import objnet

# load predictions
save_path = cfg.DIR + 'vtranse/pred_res/%s_pred_roidb.npz' % dataset
with open(save_path, 'rb') as f:
    pred_roidb = np.load(f)
    pred_roidb = pred_roidb['roidb']
    pred_roidb = pred_roidb[()]
    pred_roidb = pred_roidb['pred_roidb']

# convert
raw_object_labels = objnet.get_raw_labels()[1:]
raw_predicate_labels = prenet.get_raw_labels()[1:]

pred_roidb_sunx = {}
for i in range(len(pred_roidb)):
    roidb_use = pred_roidb[i]
    img_id = roidb_use['image_path'].split('/')[-1].split('.')[0]
    sbj_box = roidb_use['sub_box_dete']
    sbj_cls = roidb_use['sub_dete']
    obj_box = roidb_use['obj_box_dete']
    obj_cls = roidb_use['obj_dete']
    pre_box = get_predicate_box(sbj_box, obj_box)
    pre_cls = roidb_use['pred_rela']
    pre_conf = roidb_use['pred_rela_score']

    for i in range(len(sbj_cls)):
        sbj_label = raw_object_labels[int(sbj_cls[i])]
        n = objnet.get_node_by_name(sbj_label)
        sbj_cls[i] = n.index()

    for i in range(len(obj_cls)):
        obj_label = raw_object_labels[int(obj_cls[i])]
        n = objnet.get_node_by_name(obj_label)
        obj_cls[i] = n.index()

    for i in range(len(pre_cls)):
        pre_label = raw_predicate_labels[int(pre_cls[i])]
        n = prenet.get_node_by_name(pre_label)
        pre_cls[i] = n.index()

    preds = np.concatenate((pre_box, pre_cls[:, np.newaxis],
                            sbj_box, sbj_cls[:, np.newaxis],
                            obj_box, obj_cls[:, np.newaxis],
                            pre_conf[:, np.newaxis]), axis=1)
    pred_roidb_sunx[img_id] = preds

save_path = 'pre_box_label_%s_vts.bin' % dataset
with open(save_path, 'wb') as f:
    pickle.dump(pred_roidb_sunx, f)