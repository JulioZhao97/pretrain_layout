import json
import torch
import random
import torchvision

from .base import *

def read_data(json_file):
    data = json.load(open(json_file))
    element_all = []
    image2anno = {image["id"]:image for image in data["images"]}
    for anno in data["annotations"]:
        H, W = image2anno[anno["image_id"]]["height"], image2anno[anno["image_id"]]["width"]
        w, h = anno["bbox"][2], anno["bbox"][3]
        if w/W < 0.05 or h/H < 0.05:
            continue
        category_id = anno["category_id"]
        e = element(cx=None, cy=None, h=h/H, w=w/W, category=category_id)
        element_all.append(e)
    return element_all

def read_diffusion_data(json_file):
    data = json.load(open(json_file))
    layout_all = []
    for idx in range(len(data)):
        element_all = []
        for i in range(len(data[idx]["boxes"])):
            x1, y1, x2, y2 = data[idx]["boxes"][i]
            w, h = x2 - x1, y2 - y1
            cx, cy = (x1+x2)/2, (y1+y2)/2
            category_id = data[idx]["categories"][i]
            if w < 0.02 or h < 0.02:
                continue
            e = element(cx=cx, cy=cy, h=h, w=w, category=category_id)
            element_all.append(e)
        layout_all.append(element_all)
    return layout_all

def clamp(cand_elements):
    # post-process
    # clamp coordinates between [0,1]
    for e in cand_elements:
        x1, y1, x2, y2 = e.cx-e.w/2, e.cy-e.h/2, e.cx+e.w/2, e.cy+e.h/2
        if x1 < 0:
            x1 = random.uniform(1e-4, 1e-3)
        if y1 < 0:
            y1 = random.uniform(1e-4, 1e-3)
        if x2 > 1:
            x2 = 1 - random.uniform(1e-4, 1e-3)
        if y2 > 1:
            y2 = 1 - random.uniform(1e-4, 1e-3)
        e.cx, e.cy = (x1+x2)/2, (y1+y2)/2
        e.w, e.h = x2-x1, y2-y1
    return cand_elements

def iou_remove(cand_elements):
    # post-process
    # remove bboxes which ious > 0.5
    boxes = []
    abandon_idx = []
    for e in cand_elements:
        x1, y1, x2, y2 = e.cx-e.w/2, e.cy-e.h/2, e.cx+e.w/2, e.cy+e.h/2
        boxes.append([x1, y1, x2, y2])
    piou = torchvision.ops.box_iou(torch.Tensor(boxes), torch.Tensor(boxes))
    for i in range(piou.shape[0]):
        for j in range(piou.shape[0]):
            if i == j:
                continue
            if piou[i][j] > 0.5:
                # keep the large
                ei, ej = cand_elements[i], cand_elements[j]
                if ei.h*ei.w < ej.h*ej.w:
                    abandon_idx.append(i)
                else:
                    abandon_idx.append(j)
    abandon_idx = list(set(abandon_idx))
    cand_elements = [cand_elements[idx] for idx in range(len(cand_elements)) if idx not in abandon_idx]
    return cand_elements

def handle_overlap(cand_elements):
    # post-process
    # clamp overlapped bboxes
    abandon_idx = []
    for _ in range(3):
        for i, ei in enumerate(cand_elements):
            for j, ej in enumerate(cand_elements):
                if i == j:
                    continue
                xi1, yi1, xi2, yi2 = ei.cx-ei.w/2, ei.cy-ei.h/2, ei.cx+ei.w/2, ei.cy+ei.h/2
                xj1, yj1, xj2, yj2 = ej.cx-ej.w/2, ej.cy-ej.h/2, ej.cx+ej.w/2, ej.cy+ej.h/2
                if xi1 > xj1:
                    continue

                # overlap?
                if max(xi1, xj1) < min(xi2, xj2) and max(yi1, yj1) < min(yi2, yj2):
                    if (xj1 > xi1 and xj2 < xi2):
                        if yj1 < yi1 and yi1 < yj2 and yj2 < yi2:
                            min_dist = min([
                                abs(yj2-yi1)/(yj2-yj1), abs(yj2-yi1)/(yi2-yi1)
                            ])
                            if abs(yj2-yi1)/(yj2-yj1) == min_dist:
                                yj2 -= abs(yj2-yi1)
                                ej.cx, ej.cy = (xj1+xj2)/2, (yj1+yj2)/2
                                ej.w, ej.h = xj2-xj1, yj2-yj1
                                continue
                            elif abs(yj2-yi1)/(yi2-yi1) == min_dist:
                                yi1 += abs(yj2-yi1)
                                ei.cx, ei.cy = (xi1+xi2)/2, (yi1+yi2)/2
                                ei.w, ei.h = xi2-xi1, yi2-yi1
                                continue
                        elif yj1 < yi1 and yj2 > yi2:
                            if (ei.h*ei.w) < (ej.h*ej.w):
                                abandon_idx.append(i)
                            elif (ei.h*ei.w) > (ej.h*ej.w):
                                abandon_idx.append(j)
                            continue
                        elif yj1 > yi1 and yj2 < yi2:
                            if (ei.h*ei.w) < (ej.h*ej.w):
                                abandon_idx.append(i)
                            elif (ei.h*ei.w) > (ej.h*ej.w):
                                abandon_idx.append(j)
                            continue
                        elif yi1 < yj1 and yj1 < yi2 and yi2 < yj2:
                            min_dist = min([
                                abs(yi2-yj1)/(yj2-yj1), abs(yj2-yi1)/(yi2-yi1)
                            ])
                            if abs(yi2-yj1)/(yj2-yj1) == min_dist:
                                yj1 += abs(yi2-yj1)
                                ej.cx, ej.cy = (xj1+xj2)/2, (yj1+yj2)/2
                                ej.w, ej.h = xj2-xj1, yj2-yj1
                                continue
                            elif abs(yi2-yj1)/(yi2-yi1) == min_dist:
                                yi2 -= abs(yi2-yj1)
                                ei.cx, ei.cy = (xi1+xi2)/2, (yi1+yi2)/2
                                ei.w, ei.h = xi2-xi1, yi2-yi1
                                continue
                    elif (xj1 > xi1 and xj2 > xi2):
                        if yj1 < yi1 and yi1 < yj2 and yj2 < yi2:
                            min_dist = min([
                                abs(xi2-xj1)/(xi2-xi1), abs(xi2-xj1)/(xj2-xj1),
                                abs(yj2-yi1)/(yj2-yj1), abs(yj2-yi1)/(yi2-yi1),
                            ])
                            if abs(yj2-yi1)/(yj2-yj1) == min_dist:
                                yj2 -= abs(yj2-yi1)
                                ej.cx, ej.cy = (xj1+xj2)/2, (yj1+yj2)/2
                                ej.w, ej.h = xj2-xj1, yj2-yj1
                                continue
                            elif abs(yj2-yi1)/(yi2-yi1) == min_dist:
                                yi1 += abs(yj2-yi1)
                                ei.cx, ei.cy = (xi1+xi2)/2, (yi1+yi2)/2
                                ei.w, ei.h = xi2-xi1, yi2-yi1
                                continue
                            elif abs(xi2-xj1)/(xi2-xi1) == min_dist:
                                xi2 -= abs(xi2-xj1)
                                ei.cx, ei.cy = (xi1+xi2)/2, (yi1+yi2)/2
                                ei.w, ei.h = xi2-xi1, yi2-yi1
                                continue
                            elif abs(xi2-xj1)/(xj2-xj1) == min_dist:
                                xj1 += abs(xi2-xj1)
                                ej.cx, ej.cy = (xj1+xj2)/2, (yj1+yj2)/2
                                ej.w, ej.h = xj2-xj1, yj2-yj1
                                continue
                        elif yj1 < yi1 and yj2 > yi2:
                            min_dist = min([
                                abs(xi2-xj1)/(xi2-xi1), abs(xi2-xj1)/(xj2-xj1),
                            ])
                            if abs(xi2-xj1)/(xi2-xi1) == min_dist:
                                xi2 -= abs(xi2-xj1)
                                ei.cx, ei.cy = (xi1+xi2)/2, (yi1+yi2)/2
                                ei.w, ei.h = xi2-xi1, yi2-yi1
                                continue
                            elif abs(xi2-xj1)/(xj2-xj1) == min_dist:
                                xj1 += abs(xi2-xj1)
                                ej.cx, ej.cy = (xj1+xj2)/2, (yj1+yj2)/2
                                ej.w, ej.h = xj2-xj1, yj2-yj1
                                continue
                        elif yj1 > yi1 and yj2 < yi2:
                            min_dist = min([
                                abs(xi2-xj1)/(xi2-xi1), abs(xi2-xj1)/(xj2-xj1),
                            ])
                            if abs(xi2-xj1)/(xi2-xi1) == min_dist:
                                xi2 -= abs(xi2-xj1)
                                ei.cx, ei.cy = (xi1+xi2)/2, (yi1+yi2)/2
                                ei.w, ei.h = xi2-xi1, yi2-yi1
                                continue
                            elif abs(xi2-xj1)/(xj2-xj1) == min_dist:
                                xj1 += abs(xi2-xj1)
                                ej.cx, ej.cy = (xj1+xj2)/2, (yj1+yj2)/2
                                ej.w, ej.h = xj2-xj1, yj2-yj1
                                continue
                        elif yi1 < yj1 and yj1 < yi2 and yi2 < yj2:
                            min_dist = min([
                                abs(yi2-yj1)/(yj2-yj1), abs(yj2-yi1)/(yi2-yi1),
                                abs(xi2-xj1)/(xi2-xi1), abs(xi2-xj1)/(xj2-xj1),
                            ])
                            if abs(yi2-yj1)/(yj2-yj1) == min_dist:
                                yj1 += abs(yi2-yj1)
                                ej.cx, ej.cy = (xj1+xj2)/2, (yj1+yj2)/2
                                ej.w, ej.h = xj2-xj1, yj2-yj1
                                continue
                            elif abs(yi2-yj1)/(yi2-yi1) == min_dist:
                                yi2 -= abs(yi2-yj1)
                                ej.cx, ej.cy = (xj1+xj2)/2, (yj1+yj2)/2
                                ej.w, ej.h = xj2-xj1, yj2-yj1
                                continue
                            elif abs(xi2-xj1)/(xi2-xi1) == min_dist:
                                xi2 -= abs(xi2-xj1)
                                ei.cx, ei.cy = (xi1+xi2)/2, (yi1+yi2)/2
                                ei.w, ei.h = xi2-xi1, yi2-yi1
                                continue
                            elif abs(xi2-xj1)/(xj2-xj1) == min_dist:
                                xj1 += abs(xi2-xj1)
                                ej.cx, ej.cy = (xj1+xj2)/2, (yj1+yj2)/2
                                ej.w, ej.h = xj2-xj1, yj2-yj1
                                continue
    abandon_idx = list(set(abandon_idx))
    cand_elements = [cand_elements[i] for i in range(len(cand_elements)) if i not in abandon_idx]
    return cand_elements

def remove_overlap(cand_elements):
    # post-process
    # filter too small
    valid_idx = []
    for idx, e in enumerate(cand_elements):
        if e.h < 0.005 or e.w < 0.005:
            continue
        valid_idx.append(idx)
    cand_elements = [cand_elements[idx] for idx in valid_idx]
    
    # filter overlap
    boxes = []
    abandon_idx = []
    for e in cand_elements:
        x1, y1, x2, y2 = e.cx-e.w/2, e.cy-e.h/2, e.cx+e.w/2, e.cy+e.h/2
        boxes.append([x1, y1, x2, y2])
    piou = torchvision.ops.box_iou(torch.Tensor(boxes), torch.Tensor(boxes))
    for i in range(piou.shape[0]):
        for j in range(piou.shape[0]):
            if i == j:
                continue
            if piou[i][j] > 0:
                # keep the large
                ei, ej = cand_elements[i], cand_elements[j]
                if ei.h*ei.w < ej.h*ej.w:
                    abandon_idx.append(i)
                else:
                    abandon_idx.append(j)
    abandon_idx = list(set(abandon_idx))
    cand_elements = [cand_elements[idx] for idx in range(len(cand_elements)) if idx not in abandon_idx]
    return cand_elements