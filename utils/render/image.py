import os
import random

from .util import *

def select_image(rect, search_map, ratio_step, area_step, ratio_strict=False):
    ratio = (rect[2]-rect[0])/(rect[3]-rect[1])
    area = (rect[2]-rect[0])*(rect[3]-rect[1])
    ratio_range = [ratio for ratio in search_map.keys() 
        if not all([len(search_map[ratio][area])==0 for area in search_map[ratio].keys()])]
    min_ratio = min(ratio_range, key=lambda x:abs(x-ratio))
    if area_step is None:
        min_area = random.choice([area for area in list(search_map[min_ratio].keys()) if len(search_map[min_ratio][area])>0])
        return random.choice(search_map[min_ratio][min_area])
    else:
        area_range = [area for area in list(search_map[min_ratio].keys()) if len(search_map[min_ratio][area])!=0]
        min_area = min(area_range, key=lambda x:abs(x-area))
        if ratio_strict:
            assert abs(ratio - min_ratio) < ratio_step*3
        return random.choice(search_map[min_ratio][min_area])

def insert_element_as_image(page, rect, base_path, search_map, ratio_step, area_step, bound_visualize, ratio_strict=False, overlay=False):
    img_path = select_image(rect, search_map, ratio_step, area_step, ratio_strict)
    img_path = os.path.join(base_path, img_path)
    page.insert_image(rect, filename=img_path, keep_proportion=False, overlay=overlay)
    rect_visualize(page=page,type='figure',rect=rect,color_dic=COLOR_DIC,bound_visualize=bound_visualize)