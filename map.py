import numpy as np

def iou_score(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    area1 = h1*w1
    area2 = h2*w2

    x = max(0, min(x1+w1, x2+w2) - max(x1, x2))
    y = max(0, min(y1+h1, y2+h2) - max(y1, y2))
    intersect_area = x * y

    intersect2 = area1 + area2 - intersect_area

    iou = intersect_area/intersect2
    return iou

def ap_value(recall, precision):
    average_precision = 0
    for i in range(len(recall)-1):
        average_precision += (recall[i+1] - recall[i]) * ((precision[i+1] + precision[i]) / 2)
    return average_precision

def map_score(y_true, y_pred, iou_threshold = np.linspace(0.5, 0.95, 1)):
    n_class = y_true.shape[1] - 4
    average_precisions = np.zeros(n_class)
    
    for i in range(n_class):
        y_pred_class = y_pred[y_true[:, i+4] == 1]
        y_true_class = y_true[y_true[:, i+4] == 1]
        
        n_gt = len(y_true_class)
        n_pred = len(y_pred_class)
        
        if n_gt == 0:
            continue
        
        if n_pred == 0:
            average_precisions[i] = 0
            continue
        
        tp = fp = np.zeros(n_pred)
        
        for j in range(n_pred):
            box_pred = y_pred_class[j, 1:5]
            iou_max = -1
            gt_match = -1
        
            for k in range(n_gt):
                if y_true_class[k, 0] != i:
                    continue
                
                box_true = y_true_class[k, 1:5]
                iou = iou_score(box_pred, box_true)
                
                if iou > iou_max:
                    iou_max = iou
                    gt_match = k
                
            if iou_max > iou_threshold:
                if y_true_class[gt_match, 2] == 0:
                    tp[j] = 1
                    y_true_class[gt_match, 2] = j
                else:
                    fp[j] = 1
            else:
                fp[j] = 1
                
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            
            recall = tp_cumsum / n_gt
            precision = tp_cumsum / (tp_cumsum + fp_cumsum)
            
            average_precisions[i] = ap_value(recall, precision)
        
        mapScore = np.mean(average_precisions)
        return mapScore

