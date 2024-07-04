import os
import cv2
import pdb
import time
import math
import fitz
import copy
import json
import tqdm
import yaml
import pickle
import random
import argparse
import numpy as np
from PIL import Image
import multiprocessing
from matplotlib import pyplot as plt

from utils.util import *
from utils.render.text import *
from utils.render.title import *
from utils.render.image import *

def render_layout(config, layout, client, args):
    # fixed config settings
    # main text
    MAINTEXT_FONTSIZE_LOW, MAINTEXT_FONTSIZE_HIGH = config["maintext"]["fontsize"]
    # image
    LAION1M_BG_IMAGE_PATH = config["image"]["background"]["path"]
    LAION1M_IMAGE_PATH = config["image"]["laion1m"]["path"]
    CHART_IMAGE_PATH = config["image"]["chart"]["path"]
    IMAGE_RATIO_STEP = config["image"]["ratio_step"]
    IMAGE_AREA_STEP = config["image"]["area_step"]
    # table
    TABLE_RATIO_STEP = config['table']['ratio_step']
    TABLE_AREA_STEP = config['table']['area_step']
    TABLE_PATH = config['table']['path']
    # title (image)
    IMAGE_TITLE_PATH = config['title']['method_figure']['path']
    TITLE_RATIO_STEP = config['title']['method_figure']['ratio_step']
    TITLE_AREA_STEP = config['title']['method_figure']['area_step']
    # title (text)
    TITLE_FONTSIZE_LOW, TITLE_FONTSIZE_HIGH = config['title']['method_text']['fontsize']
    # others
    BG_PROB = config["others"]["bg_prob"]
    FIND_THR = config["others"]["find_thr"]
    LINE_MARGIN = config["others"]["find_thr"]
    ERASER_RATIO = config["others"]["line_margin"]
    BOUND_VISUALIZE = config["others"]["visualize"]
    
    doc = fitz.open()
    W, H = sample_hw(
        width_range=config['size']['width'],
        ratio_range=config['size']['ratio'],
        max_height=config['size']['max_height'],
    )
    annotation_json = {"bbox": [], "labels": [], "width":W, "height":H}

    page = doc.new_page(width=W, height=H)

    LANGUAGE = random.choice(config["languages"])
    TEXT_TITLE_PATH = config['title']['method_text'][f'{LANGUAGE}_path']
    MAINTEXT_PATH = config["maintext"][f"{LANGUAGE}_path"]
    page_type = random.choice(config["others"]["page_type"])
    
    for bbox, category in zip(layout["boxes"], layout["categories"]):
        bbox[0], bbox[2] = bbox[0]*W, bbox[2]*W
        bbox[1], bbox[3] = bbox[1]*H, bbox[3]*H
        rect = fitz.Rect([bbox[i] for i in range(4)])
        
        # text
        if category == 0 or category == 2:
            try:
                MAIN_TEXT_FONTSIZE = random.uniform(MAINTEXT_FONTSIZE_LOW,MAINTEXT_FONTSIZE_HIGH)
                text_bboxes = insert_text(
                    page=page,
                    rect=rect,
                    maintext_path=MAINTEXT_PATH, 
                    language=LANGUAGE, 
                    maintext_fontsize_low=MAINTEXT_FONTSIZE_LOW, 
                    maintext_fontsize_high=MAINTEXT_FONTSIZE_HIGH, 
                    main_text_fontsize=MAIN_TEXT_FONTSIZE, 
                    eraser_ratio=ERASER_RATIO, 
                    find_thr=FIND_THR, 
                    line_margin=LINE_MARGIN, 
                    bound_visualize=BOUND_VISUALIZE, 
                    page_type=page_type,
                )
            except BaseException:
                continue
            else:
                # add to annotations
                for rect in text_bboxes:
                    x0, y0, x1, y1 = rect.x0, rect.y0, rect.x1, rect.y1
                    annotation_json["bbox"].append([x0, y0, x1, y1])
                    annotation_json["labels"].append(0)  # text
        # title
        elif category == 1:
            if config["title"]["method"] == "figure":
                try:
                    insert_element_as_image(
                        page=page,
                        rect=rect,
                        base_path=IMAGE_TITLE_PATH,
                        search_map=title_search_map,
                        ratio_step=TITLE_RATIO_STEP,
                        area_step=TITLE_AREA_STEP,
                        bound_visualize=BOUND_VISUALIZE,
                        ratio_strict=True,
                    )
                except BaseException:
                    continue
                else:
                    annotation_json["bbox"].append(bbox)
                    annotation_json["labels"].append(1)  # title
            elif config["title"]["method"] == "text":
                try:
                    insert_title(
                        page=page,
                        rect=rect,
                        title_path=TEXT_TITLE_PATH,
                        language=LANGUAGE,
                        title_fontsize_low=TITLE_FONTSIZE_LOW,
                        title_fontsize_high=TITLE_FONTSIZE_HIGH,
                        find_thr=FIND_THR,
                        bound_visualize=BOUND_VISUALIZE,
                    )
                except BaseException:
                    continue
                else:
                    annotation_json["bbox"].append(bbox)
                    annotation_json["labels"].append(1)  # title
        # image
        elif category == 4:
            try:
                if random.choice(["chart", "laion"]) == "laion":
                    insert_element_as_image(
                        page=page,
                        rect=rect,
                        base_path=LAION1M_IMAGE_PATH,
                        search_map=laion1m_search_map,
                        ratio_step=IMAGE_RATIO_STEP,
                        area_step=IMAGE_AREA_STEP,
                        bound_visualize=BOUND_VISUALIZE,
                    )
                else:
                    insert_element_as_image(
                        page=page,
                        rect=rect,
                        base_path=CHART_IMAGE_PATH,
                        search_map=chart_search_map,
                        ratio_step=IMAGE_RATIO_STEP,
                        area_step=IMAGE_AREA_STEP,
                        bound_visualize=BOUND_VISUALIZE,
                    )
            except BaseException:
                continue
            else:
                annotation_json["bbox"].append(bbox)
                annotation_json["labels"].append(3)  # image
        # table
        elif category == 3:
            try:
                insert_element_as_image(
                    page=page,
                    rect=rect,
                    base_path=TABLE_PATH,
                    search_map=table_search_map,
                    ratio_step=TABLE_RATIO_STEP,
                    area_step=TABLE_AREA_STEP,
                    bound_visualize=BOUND_VISUALIZE,
                    ratio_strict=True,
                )
            except BaseException:
                continue
            else:
                annotation_json["bbox"].append(bbox)
                annotation_json["labels"].append(2)  # figure

    # insert background
    if random.uniform(0,1) < BG_PROB:
        try:
            rect = fitz.Rect([0, 0, W, H])
            insert_element_as_image(
                page=page,
                rect=rect,
                base_path=LAION1M_BG_IMAGE_PATH,
                search_map=laion1m_bg_search_map,
                ratio_step=IMAGE_RATIO_STEP,
                area_step=None,
                bound_visualize=BOUND_VISUALIZE,
                overlay=False,
            )
        except BaseException:
            pass
                
    # save image and annotation
    _id = str(time.time()).replace(".", "_")
    pix = page.get_pixmap()
    
    # save to local
    pix.save(os.path.join(image_path, f"{_id}.jpg"))
    anno_txt = open(os.path.join(anno_path, f"{_id}.txt"), "w")
    for bbox, category_id in zip(annotation_json["bbox"], annotation_json["labels"]):
        W, H = annotation_json["width"], annotation_json["height"]
        x0, y0, x1, y1 = bbox
        x0, y0 = x0/W, y0/H
        x1, y1 = x1/W, y1/H
        anno_txt.write(f"{category_id} {x0} {y0} {x1} {y0} {x1} {y1} {x0} {y1}\n")
    anno_txt.close()
        
    # save yolo(txt) annotations to ceph
    # str_bytes = ""
    # for bbox, category_id in zip(annotation_json["bbox"], annotation_json["labels"]):
    #     W, H = annotation_json["width"], annotation_json["height"]
    #     x0, y0, x1, y1 = bbox
    #     x0, y0 = x0/W, y0/H
    #     x1, y1 = x1/W, y1/H
    #     str_bytes += f"{category_id} {x0} {y0} {x1} {y0} {x1} {y1} {x0} {y1}\n"
    # str_bytes = str_bytes.encode("utf-8")
    # client.put(os.path.join(anno_path.replace(args.base_path, "s3://layout_pretrain_data_shdd"), f"{_id}.txt"), str_bytes)
        
    # save to ceph
    # img_array = np.frombuffer(buffer=pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, -1))
    # img_str = cv2.imencode('.jpg', img_array)[1].tobytes()
    # client.put(os.path.join(image_path.replace(args.base_path, "s3://layout_pretrain_data_shdd"), f"{_id}.jpg"), img_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-path', default="/mnt/hwfile/opendatalab/zhaozhiyuan/pretrain_layout", type=str, help='json file that contains all layout elements')
    parser.add_argument('--json-file', default=None, required=True, type=str, help='json file that contains all layout elements')
    parser.add_argument('--set', default=None, required=True, type=str, help='which set of data')
    args = parser.parse_args()
    
    # save path
    # assert args.set in ["random", "bestfit", "diffusion"]
    base_path = os.path.join(args.base_path, f"{args.set}_layout")
    image_path = os.path.join(args.base_path, f"{args.set}_layout", "images")
    anno_path = os.path.join(args.base_path, f"{args.set}_layout", "labels")
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(image_path, exist_ok=True)
    os.makedirs(anno_path, exist_ok=True)
    
    # load materials
    # title_search_map = pickle.load(open("title_material_map.pt", "rb"))
    # table_search_map = pickle.load(open("table_material_map.pt", "rb"))
    # image_search_map = pickle.load(open("laion1m_material_map.pt", "rb"))
    # table_search_map = pickle.load(open("pdftable_material_map.pt", "rb"))
    # image_search_map = pickle.load(open("chart_material_map.pt", "rb"))
    # bg_image_search_map = pickle.load(open("laion1m_bg_material_map.pt", "rb"))
    
    # load layout data
    layout_json = json.load(open(args.json_file))
    # layout_json = [l for l in layout_json if 1 in l["categories"]][:1000]
    # layout_json = layout_json[:100]
    
    # load config
    with open("config.yaml") as stream:
        try:
            config = yaml.safe_load(stream)
            print(json.dumps(config, indent=4))
        except yaml.YAMLError as exc:
            print(exc)
        
    title_search_map = pickle.load(open(config["title"]["method_figure"]["search_map"], "rb"))
    table_search_map = pickle.load(open(config["table"]["search_map"], "rb"))
    laion1m_search_map = pickle.load(open(config["image"]["laion1m"]["search_map"], "rb"))
    laion1m_bg_search_map = pickle.load(open(config["image"]["background"]["search_map"], "rb"))
    chart_search_map = pickle.load(open(config["image"]["chart"]["search_map"], "rb"))
        
    # ceph client
    from petrel_client.client import Client
    conf_path = '~/petreloss.conf'
    client = Client(conf_path)
        
    # render layout        
    start = time.time()
    n_jobs = 100
    with multiprocessing.Pool(n_jobs) as p:
        result = p.starmap(
            render_layout, 
            zip(
                [config for _ in range(len(layout_json))], 
                layout_json, 
                [client for _ in range(len(layout_json))],
                [args for _ in range(len(layout_json))],
            )
        )
    p.close()
    p.join()
    print(time.time() - start)
    
    # start = time.time()
    # for config, layout in tqdm.tqdm(zip([config for _ in range(len(layout_json))], layout_json)):
    #     # pdb.set_trace()
    #     render_layout(config, layout, client, args)
    # print(time.time() - start)