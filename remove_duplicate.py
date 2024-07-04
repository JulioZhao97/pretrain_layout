import pdb
import tqdm
import json
import time
import torch
import random
import argparse
import itertools
import torchvision
import numpy as np
import multiprocessing
from scipy.optimize import linear_sum_assignment

# def stage2_box_iou(layout_json, index):
#     i, j = index
#     layout1, layout2 = layout_json[i]["boxes"], layout_json[j]["boxes"]
#     layout1 = np.array(layout1)
#     layout2 = np.array(layout2)
#     iou = np.zeros((layout1.shape[0], layout2.shape[0]))
#     for m in range(iou.shape[0]):
#         for n in range(iou.shape[1]):
#             iou[m][n] = compute_iou(layout1[m], layout2[n])
#     row_ind, col_ind = linear_sum_assignment(1-iou)
#     avg_iou = [iou[i][j] for i,j in zip(row_ind, col_ind)]
#     avg_iou = sum(avg_iou)/len(avg_iou)
#     return avg_iou

def stage2_box_iou(layout_json, index):
    i, j = index
    layout1, layout2 = layout_json[i]["boxes"], layout_json[j]["boxes"]
    layout1 = torch.Tensor(layout1)
    layout2 = torch.Tensor(layout2)
    iou = torchvision.ops.box_iou(layout1,layout2).numpy()
    row_ind, col_ind = linear_sum_assignment(1-iou)
    avg_iou = [iou[i][j] for i,j in zip(row_ind, col_ind)]
    avg_iou = sum(avg_iou)/len(avg_iou)
    return avg_iou

# def stage1_box_iou(layout_json, index):
#     i, j = index
#     if i >= j:
#         return None
#     layout1, layout2 = layout_json[i], layout_json[j]
#     x11, y11 = min([b[0] for b in layout1["boxes"]]), min([b[1] for b in layout1["boxes"]])
#     x12, y12 = max([b[2] for b in layout1["boxes"]]), max([b[3] for b in layout1["boxes"]])
#     box1 = torch.Tensor([x11, y11, x12, y12]).unsqueeze(0)
#     x21, y21 = min([b[0] for b in layout2["boxes"]]), min([b[1] for b in layout2["boxes"]])
#     x22, y22 = max([b[2] for b in layout2["boxes"]]), max([b[3] for b in layout2["boxes"]])
#     box2 = torch.Tensor([x21, y21, x22, y22]).unsqueeze(0)
#     iou = torchvision.ops.box_iou(box1,box2)
#     return iou[0][0]

def compute_iou(box1, box2, wh=False):
    """
    compute the iou of two boxes.
    Args:
        box1, box2: [xmin, ymin, xmax, ymax] (wh=False) or [xcenter, ycenter, w, h] (wh=True)
        wh: the format of coordinate.
    Return:
        iou: iou of box1 and box2.
    """
    if wh == False:
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2
    else:
        xmin1, ymin1 = int(box1[0]-box1[2]/2.0), int(box1[1]-box1[3]/2.0)
        xmax1, ymax1 = int(box1[0]+box1[2]/2.0), int(box1[1]+box1[3]/2.0)
        xmin2, ymin2 = int(box2[0]-box2[2]/2.0), int(box2[1]-box2[3]/2.0)
        xmax2, ymax2 = int(box2[0]+box2[2]/2.0), int(box2[1]+box2[3]/2.0)

    ## 获取矩形框交集对应的左上角和右下角的坐标（intersection）
    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])

    ## 计算两个矩形框面积
    area1 = (xmax1-xmin1) * (ymax1-ymin1) 
    area2 = (xmax2-xmin2) * (ymax2-ymin2)

    inter_area = (np.max([0, xx2-xx1])) * (np.max([0, yy2-yy1]))
    iou = inter_area / (area1+area2-inter_area+1e-6)

    return iou

def stage1_box_iou(layout_json, index):
    i, j = index
    if i >= j:
        return None
    layout1, layout2 = layout_json[i], layout_json[j]
    x11, y11 = min([b[0] for b in layout1["boxes"]]), min([b[1] for b in layout1["boxes"]])
    x12, y12 = max([b[2] for b in layout1["boxes"]]), max([b[3] for b in layout1["boxes"]])
    box1 = np.array([x11, y11, x12, y12])
    x21, y21 = min([b[0] for b in layout2["boxes"]]), min([b[1] for b in layout2["boxes"]])
    x22, y22 = max([b[2] for b in layout2["boxes"]]), max([b[3] for b in layout2["boxes"]])
    box2 = np.array([x21, y21, x22, y22])
    iou = compute_iou(box1,box2)
    return iou

def stage2_filter(layout_json, index, res):
    # stage2 filter duplicate
    if not res:
        return None
    if res > 0.8:
        avg_iou = stage2_box_iou(layout_json, index)
        if avg_iou > 0.9:
            return index
        else:
            return None
    else:
        return None

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--json-file', default=None, required=True, type=str, help='json file that contains all layout elements.')
    parser.add_argument('--output-file', default=None, required=True, type=str, help='json file whose duplicates are removed.')
    args = parser.parse_args()
    
    layout_json = json.load(open(args.json_file))
    chunksize = 1000
    num_chunks = len(layout_json) // chunksize
    
    filtered_layout_json = []
    
    for chunk_idx in tqdm.tqdm(range(num_chunks)):
        chunk_layout_json = layout_json[chunk_idx*chunksize: (chunk_idx+1)*chunksize]
        index_all = list(itertools.product([i for i in range(len(chunk_layout_json))], [i for i in range(len(chunk_layout_json))]))

        # stage1 filter
        start = time.time()
        n_jobs = 100
        with multiprocessing.Pool(n_jobs) as p:
            result = p.starmap(
                stage1_box_iou, 
                zip([chunk_layout_json for _ in range(len(index_all))], index_all)
            )
        p.close()
        p.join()
        # print(time.time() - start)
                    
        # stage2 filter
        start = time.time()
        n_jobs = 100
        with multiprocessing.Pool(n_jobs) as p:
            repeat_index = p.starmap(
                stage2_filter, 
                zip([chunk_layout_json for _ in range(len(index_all))], index_all, result)
            )
        p.close()
        p.join()
        # print(time.time() - start)
                    
        start = time.time()
        repeat_index = [index for index in repeat_index if index]
        repeat_index_all = []
        for i, index_i in enumerate(repeat_index):
            repeat_index2 = [index_i[0], index_i[1]]
            while True:
                length_old = len(repeat_index2)
                for j, index_j in enumerate(repeat_index):
                    if j == i:
                        continue
                    if index_j[0] in repeat_index2 and index_j[1] not in repeat_index2:
                        repeat_index2.append(index_j[1])
                        continue
                    elif index_j[1] in repeat_index2 and index_j[0] not in repeat_index2:
                        repeat_index2.append(index_j[0])
                        continue
                repeat_index2 = list(set(repeat_index2))
                repeat_index2 = sorted(repeat_index2)
                if len(repeat_index2) == length_old:
                    break
            if repeat_index2 not in repeat_index_all:
                repeat_index_all.append(repeat_index2)
                
        abandon_index_all = []
        for repeat_index in repeat_index_all:
            abandon_index_all.extend(random.sample(repeat_index, len(repeat_index)-1))
        # print(time.time() - start)  
        
        # filtered layout
        filtered_layout_json.extend([
            chunk_layout_json[idx] for idx in range(len(chunk_layout_json)) \
            if idx not in abandon_index_all
        ])
    
    with open(args.output_file, "w") as f:
        json.dump(filtered_layout_json, f)