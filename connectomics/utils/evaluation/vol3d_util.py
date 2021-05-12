import sys
import numpy as np
import h5py
import cv2
from tqdm import tqdm

####
# list of utility functions
# 0. I/O util
# 1. binary pred -> instance seg
# 2. instance seg + pred heatmap -> instance score
# 3. instance seg -> bbox
# 4. instance seg + gt seg + instance score -> sorted match result

def readh5(path, vol=''):
    # do the first key
    fid = h5py.File(path, 'r')
    if vol == '': 
        if sys.version[0]=='3':
            vol = list(fid)[0]
        else: # python 2
            vol = fid.keys()[0] 
    return np.array(fid[vol]).squeeze()

def readh5_handle(path, vol=''):
    # do the first key
    fid = h5py.File(path, 'r')
    if vol == '': 
        if sys.version[0]=='3':
            vol = list(fid)[0]
        else: # python 2
            vol = fid.keys()[0]
            
    return fid[vol]


def getQueryCount(ui,uc,qid):
    # memory efficient
    ui_r = [ui[ui>0].min(),max(ui.max(),qid.max())]
    rl = np.zeros(1+int(ui_r[1]-ui_r[0]),uc.dtype)
    rl[ui[ui>0]-ui_r[0]] = uc[ui>0]

    cc = np.zeros(qid.shape,uc.dtype)
    gid = np.logical_and(qid>=ui_r[0], qid<=ui_r[1])
    cc[gid] = rl[qid[gid]-ui_r[0]]
    return cc

def unique_chunk(seg, slices, chunk_size = 50, do_count = True):
    # load unique segment ids and segment sizes (in voxels) chunk by chunk
    num_z = slices[1] - slices[0]
    num_chunk = (num_z + chunk_size -1 ) // chunk_size
    
    uc_arr = None
    ui = []
    for cid in range(num_chunk):
        # compute max index, modulo takes care of slices[1] = -1
        max_idx = min([(cid + 1) * chunk_size + slices[0], slices[1]])
        chunk = np.array(seg[cid * chunk_size + slices[0]: max_idx])

        if do_count:
            ui_c, uc_c = np.unique(chunk, return_counts = True)
            if uc_arr is None:
                uc_arr = np.zeros(ui_c.max()+1, int)
                uc_arr[ui_c] = uc_c
                uc_len = len(uc_arr)
            else:
                if uc_len <= ui_c.max():
                    # at least double the length
                    uc_arr = np.hstack([uc_arr, np.zeros(max(ui_c.max()-uc_len, uc_len) + 1, int)]) #max + 1 for edge case (uc_len = ui_c.max())
                    uc_len = len(uc_arr)
                uc_arr[ui_c] += uc_c
        else:
            ui = np.unique(np.hstack([ui, np.unique(chunk)]))

    if do_count:
        ui = np.where(uc_arr>0)[0]
        return ui, uc_arr[ui]
    else:
        return ui

def unique_chunks_bbox(seg1, seg2, seg2_val, bbox, chunk_size = 50, do_count = True):
    # load unique segment ids and segment sizes (in voxels) chunk by chunk
    num_z = bbox[1] - bbox[0]
    num_chunk = (num_z + chunk_size -1 ) // chunk_size
    
    uc_arr = None
    ui = []
    for cid in range(num_chunk):
        # compute max index, modulo takes care of slices[1] = -1
        max_idx = min([(cid + 1) * chunk_size + bbox[0], bbox[1]])
        chunk = np.array(seg1[cid * chunk_size + bbox[0]:max_idx, bbox[2]:bbox[3], bbox[4]:bbox[5]])
        chunk = chunk * (np.array(seg2[cid * chunk_size + bbox[0]:max_idx, bbox[2]:bbox[3], bbox[4]:bbox[5]]) == seg2_val)

        if do_count:
            ui_c, uc_c = np.unique(chunk, return_counts = True)
            if uc_arr is None:
                uc_arr = np.zeros(ui_c.max()+1, int)
                uc_arr[ui_c] = uc_c
                uc_len = len(uc_arr)
            else:
                if uc_len <= ui_c.max():
                    # at least double the length
                    uc_arr = np.hstack([uc_arr, np.zeros(max(ui_c.max()-uc_len, uc_len) + 1, int)]) #max + 1 for edge case (uc_len = ui_c.max())
                    uc_len = len(uc_arr)
                uc_arr[ui_c] += uc_c
        else:
            ui = np.unique(np.hstack([ui, np.unique(chunk)]))

    if do_count:
        ui = np.where(uc_arr>0)[0]
        return ui, uc_arr[ui]
    else:
        return ui


# 3. instance seg -> bbox
def seg_bbox3d(seg, slices, uid=None, chunk_size=50):
    """returns bounding box of segments"""
    sz = seg.shape
    assert len(sz)==3
    uic = None
    if uid is None:
        uid, uic = unique_chunk(seg, slices, chunk_size)
        uic = uic[uid>0]
        uid = uid[uid>0]
    um = int(uid.max())
    out = np.zeros((1+um,7),dtype=np.uint32)
    out[:,0] = np.arange(out.shape[0])
    out[:,1], out[:,3], out[:,5] = sz[0], sz[1], sz[2]

    num_z = slices[1] - slices[0]
    num_chunk = (num_z + chunk_size -1 ) // chunk_size
    for chunk_id in range(num_chunk):
        print('\t\t chunk %d' % chunk_id)
        z0 = chunk_id * chunk_size + slices[0]
        # compute max index, modulo takes care of slices[1] = -1
        max_idx = min([z0 + chunk_size, slices[1]])
        seg_c = np.array(seg[z0 : max_idx])
        # for each slice
        for zid in np.where((seg_c>0).sum(axis=1).sum(axis=1)>0)[0]:
            sid = np.unique(seg_c[zid])
            sid = sid[(sid>0)*(sid<=um)]
            out[sid,1] = np.minimum(out[sid,1], z0 + zid)
            out[sid,2] = np.maximum(out[sid,2], z0 + zid)

        # for each row
        for rid in np.where((seg_c>0).sum(axis=0).sum(axis=1)>0)[0]:
            sid = np.unique(seg_c[:,rid])
            sid = sid[(sid>0)*(sid<=um)]
            out[sid,3] = np.minimum(out[sid,3],rid)
            out[sid,4] = np.maximum(out[sid,4],rid)
        
        # for each col
        for cid in np.where((seg_c>0).sum(axis=0).sum(axis=0)>0)[0]:
            sid = np.unique(seg_c[:,:,cid])
            sid = sid[(sid>0)*(sid<=um)]
            out[sid,5] = np.minimum(out[sid,5],cid)
            out[sid,6] = np.maximum(out[sid,6],cid)
    # max + 1
    out[:,2::2] += 1
    return out[uid]

def seg_iou3d(pred, gt, slices, areaRng=np.array([]), todo_id=None, chunk_size=100, crumb_size = -1):
    # returns the matching pairs of ground truth IDs and prediction IDs, as well as the IoU of each pair.
    # (pred,gt)
    # return: id_1,id_2,size_1,size_2,iou
    pred_id, pred_sz = unique_chunk(pred, slices, chunk_size)
    if todo_id.max() > pred_id.max():
        raise ValueError('The predict-score has bigger id (%d) than the prediction (%d)' % (todo_id.max(), pred_id.max()))


    pred_sz = pred_sz[pred_id > 0]
    pred_id = pred_id[pred_id > 0]
    predict_sz_rl = np.zeros(int(pred_id.max()) + 1,int)
    predict_sz_rl[pred_id] = pred_sz
    
    gt_id, gt_sz = unique_chunk(gt, slices, chunk_size)
    gt_sz = gt_sz[gt_id > 0]
    gt_id = gt_id[gt_id > 0]
    rl_gt = None
    if crumb_size > -1:
        gt_id = gt_id[gt_sz >= crumb_size]
        gt_sz = gt_sz[gt_sz >= crumb_size]
    
    if todo_id is None:
        todo_id = pred_id
        todo_sz = pred_sz
    else:
        todo_sz = predict_sz_rl[todo_id]
   
    print('\t compute bounding boxes')
    bbs = seg_bbox3d(pred, slices, uid = todo_id, chunk_size = chunk_size)[:,1:]    
    
    result_p = np.zeros((len(todo_id), 2+3*areaRng.shape[0]), float)
    result_p[:,0] = todo_id
    result_p[:,1] = todo_sz

    gt_matched_id = np.zeros(1+gt_id.max(), int)
    gt_matched_iou = np.zeros(1+gt_id.max(), float)

    print('\t compute iou matching')
    for j,i in tqdm(enumerate(todo_id)):
        # Find intersection of pred and gt instance inside bbox, call intersection match_id
        bb = bbs[j]
        # can be big memory
        #match_id, match_sz=np.unique(np.array(gt[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]])*(np.array(pred[bb[0]:bb[1],bb[2]:bb[3], bb[4]:bb[5]])==i),return_counts=True)
        match_id, match_sz = unique_chunks_bbox(gt, pred, i, bb, chunk_size)
        match_id_g = np.isin(match_id, gt_id)
        match_sz = match_sz[match_id_g] # get intersection counts
        match_id = match_id[match_id_g] # get intersection ids        
        if len(match_id)>0:
            # get count of all preds inside bbox (assume gt_id,match_id are of ascending order)
            gt_sz_match = getQueryCount(gt_id, gt_sz, match_id)
            ious = match_sz.astype(float)/(todo_sz[j] + gt_sz_match - match_sz) #all possible iou combinations of bbox ids are contained
            
            for r in range(areaRng.shape[0]): # fill up all, then s, m, l
                gid = (gt_sz_match>areaRng[r,0])*(gt_sz_match<=areaRng[r,1])
                if sum(gid)>0: 
                    idx_iou_max = np.argmax(ious*gid)
                    result_p[j,2+r*3:2+r*3+3] = [ match_id[idx_iou_max], gt_sz_match[idx_iou_max], ious[idx_iou_max] ]            
            # update set2
            gt_todo = gt_matched_iou[match_id]<ious            
            gt_matched_iou[match_id[gt_todo]] = ious[gt_todo]
            gt_matched_id[match_id[gt_todo]] = i
                
    # get the rest: false negative + dup
    fn_gid = gt_id[np.isin(gt_id, result_p[:,2], assume_unique=False, invert=True)]
    fn_gic = gt_sz[np.isin(gt_id, fn_gid)]
    fn_iou = gt_matched_iou[fn_gid]
    fn_pid = gt_matched_id[fn_gid]
    fn_pic = predict_sz_rl[fn_pid]
    
    # add back duplicate
    # instead of bookkeeping in the previous step, faster to redo them    
    result_fn = np.vstack([fn_pid, fn_pic, fn_gid, fn_gic, fn_iou]).T
    
    return result_p, result_fn

def seg_iou3d_sorted(pred, gt, score, slices, areaRng = [0,1e10], chunk_size = 250, crumb_size = -1):
    # pred_score: Nx2 [id, score]
    # 1. sort prediction by confidence score
    relabel = np.zeros(int(np.max(score[:,0])+1), float)
    relabel[score[:,0].astype(int)] = score[:,1]
    
    # 1. sort the prediction by confidence
    pred_id = np.unique(score[:,0])
    pred_id = pred_id[pred_id>0]
    pred_id_sorted = np.argsort(-relabel[pred_id])
    
    result_p, result_fn = seg_iou3d(pred, gt, slices, areaRng, pred_id[pred_id_sorted], chunk_size, crumb_size)
    # format: pid,pc,p_score, gid,gc,iou
    pred_score_sorted = relabel[pred_id_sorted].reshape(-1,1)
    return result_p, result_fn, pred_score_sorted
