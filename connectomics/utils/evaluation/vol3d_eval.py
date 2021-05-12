import numpy as np
import datetime
import time

import csv

import pdb
class VOL3Deval:
    # Interface for evaluating video instance segmentation on the YouTubeVIS dataset.
    #
    # The usage for YTVOSeval is as follows:
    #  cocoGt=..., cocoDt=...       # load dataset and results
    #  E = VOL3Deval(cocoGt,cocoDt); # initialize YTVOSeval object
    #  E.params.recThrs = ...;      # set parameters as desired
    #  E.evaluate();                # run per image evaluation
    #  E.accumulate();              # accumulate per image results
    #  E.summarize();               # display summary metrics of results
    # For example usage see evalDemo.m and http://mscoco.org/.
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  iouType    - ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'
    #  iouType replaced the now DEPRECATED useSegm parameter.
    #  useCats    - [1] if true use category labels for evaluation
    # Note: if useCats=0 category labels are ignored as in proposal scoring.
    # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
    #
    # evaluate(): evaluates detections on every image and every category and
    # concats the results into the "evalImgs" with fields:
    #  dtIds      - [1xD] id for each of the D detections (dt)
    #  gtIds      - [1xG] id for each of the G ground truths (gt)
    #  dtMatches  - [TxD] matching gt id at each IoU or 0
    #  gtMatches  - [TxG] matching dt id at each IoU or 0
    #  dtScores   - [1xD] confidence of each dt
    #  gtIgnore   - [1xG] ignore flag for each gt
    #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
    #
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in "evalImgs" into the dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    # Note: precision and recall==-1 for settings with no gt objects.
    #
    # See also coco, mask, pycocoDemo, pycocoEvalDemo
    #
    # Microsoft COCO Toolbox.      version 2.0
    # Data, paper, and tutorials available at:  http://mscoco.org/
    # Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
    # Licensed under the Simplified BSD License [see coco/license.txt]
    def __init__(self, result_p, result_fn, score_p=None, model_num=None, path=None, iouType='segm', output_name=''):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param cocoGt: coco object with ground truth annotations
        :param cocoDt: coco object with detection results
        :return: None
        '''
        if not iouType:
            print('iouType not specified. use default iouType segm')
        # num_obj x {all, s, m ,l} x {id, size, IOU}

        # load false negative
        self.result_fn = result_fn
        self.result_p = result_p
        self.output_name = output_name
        self.model_num = model_num
        self.path = path

        # load detection
        self.cocoDt = result_p[:,:2] # detections COCO API
        self.D = self.cocoDt.shape[0]
        self.scores = score_p # detections COCO API
        if self.scores is None:
            self.scores = np.zeros(self.D)

        self.params = Params(iouType=iouType) # parameters
        self.th = self.params.iouThrs.repeat(self.D).reshape((-1,self.D)) #get same length as ious
        self.T = len(self.params.iouThrs)

        self.cocoGt = result_p[:,2:].reshape(-1,4,3)    # ground truth COCO API
        gid,gix = np.unique(np.hstack([self.result_fn[:,2],self.cocoGt[:,0,0]]), return_index=True)
        gic = np.hstack([self.result_fn[:,3],self.cocoGt[:,0,1]])[gix[gid>0]]
        self.gid = gid[gid>0].astype(int)
        self.gic = gic
        self.G = len(self.gid)

        self.eval     = {}                  # accumulated evaluation results
        self.stats = []                     # result summarization

    def get_dtm_by_area(self, area_id):
        """
        For each instance, we need the number of true positives, false positives and false negatives
        at each IoU threshold.
        """

        cocoGt = self.cocoGt[:,area_id]

        # gtIg: size self.G (include 0)
        gtIg = (self.gic<=self.params.areaRng[area_id,0])+(self.gic>self.params.areaRng[area_id,1])
        gtIg_id = self.gid[gtIg]

        # if no match in the area range, add back best
        match_id = cocoGt[:,0].astype(int)
        match_iou = cocoGt[:,2]
        match_iou[match_id==0] = self.cocoGt[match_id==0,0,2]
        match_id[match_id==0] = self.cocoGt[match_id==0,0,0]

        dtm = match_id*(match_iou>=self.th)
        # find detection outside the area range
        dtIg = (dtm>0)*np.isin(dtm,gtIg_id).reshape(dtm.shape)
        a = (self.cocoDt[:,1]<=self.params.areaRng[area_id,0])+(self.cocoDt[:,1]>self.params.areaRng[area_id,1])
        dtIg = np.logical_or(dtIg, np.logical_and(dtm==0, np.tile(a,(self.T,1))))

        tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
        fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )

        npig = (gtIg==0).sum()
        return tps, fps, npig

    def accumulate(self, p = None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''

        print('Accumulating evaluation results...')
        tic = time.time()
#         if not self.evalImgs:
#             print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        T           = len(p.iouThrs)
        R           = len(p.recThrs)
        A           = len(p.areaRng)
        precision   = -np.ones((T,R,A)) # -1 for the precision of absent categories
        recall      = -np.ones((T,A))
        scores      = -np.ones((T,R,A))

        # create dictionary for future indexing
        _pe = self.params
        setA = set(map(tuple, _pe.areaRng))
        # get inds to evaluate
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        Nk = A0
        for a, a0 in enumerate(a_list):
            tps,fps,npig = self.get_dtm_by_area(a)
            if npig == 0:
                continue

            tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
            fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)

            for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                tp = np.array(tp)
                fp = np.array(fp)
                nd = len(tp)
                rc = tp / npig
                pr = tp / (fp+tp+np.spacing(1))
                q  = np.zeros((R,))
                ss = np.zeros((R,))

                if nd:
                    recall[t,a] = rc[-1]
                else:
                    recall[t,a] = 0

                # numpy is slow without cython optimization for accessing elements
                # use python array gets significant speed improvement
                pr = pr.tolist(); q = q.tolist()

                for i in range(nd-1, 0, -1):
                    if pr[i] > pr[i-1]:
                        pr[i-1] = pr[i]

                inds = np.searchsorted(rc, p.recThrs, side='left')
                try:
                    for ri, pi in enumerate(inds):
                        q[ri] = pr[pi]
                        ss[ri] = self.scores[pi]
                except:
                    pass
                precision[t,:,a] = np.array(q)
                scores[t,:,a] = np.array(ss)
        self.eval = {
            'params': p,
            'counts': [T, R, A],
#             'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall':   recall,
            'scores': scores,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format( toc-tic))

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]

            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,:,aind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:,aind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])

            msg = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
            if self.output_writer is None:
                print(msg)
            else:
                self.output_writer.write(msg+'\n')

            return mean_s


        def write_csv(path, epoch, map75):
            path = path + "/sum_results_map75.csv"
            with open(path, 'a+') as f:
                csv_write = csv.writer(f)
                data_row = [epoch, map75]
                csv_write.writerow(data_row)

        def _summarizeDets():

            stats = np.zeros((10,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5)#, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.75)#, maxDets=self.params.maxDets[2])
            write_csv(path=self.path, epoch=self.model_num, map75=stats[2])
            stats[3] = _summarize(1, areaRng='small', iouThr=.75)#, maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='medium', iouThr=.75)#, maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='large', iouThr=.75)#, maxDets=self.params.maxDets[2])
            # no recall
            """
            stats[6] = _summarize(0)#, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, areaRng='small')
            stats[8] = _summarize(0, areaRng='medium')
            stats[9] = _summarize(0, areaRng='large')
            """
            return stats

        if not self.eval:
            raise Exception('Please run accumulate() first')

        self.output_writer = open(self.output_name+'_map.txt','w') if self.output_name!='' else None
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets

        self.stats = summarize()
        if self.output_writer is not None:
            self.output_writer.close()


    def save_match_p(self, output_name=''):
        header = '\tprediction  |\t\t gt all \t\t|\t\t gt small \t\t|\t\tgt medium \t\t|\t gt large\n' + \
                    'ID\tSIZE\t| ID\tSIZE\tIoU\t\t| ID\tSIZE\tIoU\t\t| ID\tSIZE\tIoU\t\t| ID\tSIZE\tIoU\n' + '-'*108
        rowformat = '%d\t\t%4d\t\t%d\t%4d\t%.4f\t\t%d\t%4d\t%.4f\t\t%d\t%4d\t%.4f\t\t%d\t%4d\t%.4f'
        np.savetxt(self.output_name+output_name+'_match_p.txt', self.result_p, fmt=rowformat, header=header)

    def save_match_fn(self, output_name=''):
        header = '\tprediction \t|\t\tgt \t\n' + \
                    'ID\tSIZE\t| ID\tSIZE\tIoU \n' + '-'*40
        rowformat = '%d\t\t%4d\t\t%d\t%4d\t%.4f'
        np.savetxt(self.output_name+output_name+'_match_fn.txt', self.result_fn, fmt=rowformat, header=header)

    def __str__(self):
        self.summarize()

class Params:
    '''
    Params for coco evaluation api
    '''
    def setDetParams(self):
    	# np.arange causes trouble.  the data point on arange is slightly larger than the true
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05) + 1), endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01) + 1), endpoint=True)
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 128 ** 2], [ 128 ** 2, 256 ** 2], [256 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']

    def __init__(self, iouType='segm'):
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None
