import tqdm
import copy
import random
import argparse
import datetime

from utils.generator.process import *

random.seed(datetime.datetime.now().timestamp())

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', default=None, required=True, type=int, help='number of layouts to generate')
    parser.add_argument('--filter', default=None, required=False, type=float, help='whether filter low filled layout.')
    parser.add_argument('--json-file', default=None, required=True, type=str, help='json file that contains all layout elements')
    parser.add_argument('--output-file', default='output-random_layout.json', type=str, help='json file that contains all layout elements')
    args = parser.parse_args()
    
    print("Loading layout elements....")
    element_all = read_data(args.json_file)
    
    N = args.n
    json_layout = []
    for generate_idx in tqdm.tqdm(range(N)):
        # initial random generate
        n_elements = random.randint(1,25)
        cand_elemnts_idx = random.sample(list(range(len(element_all))), n_elements)
        cand_elements = [copy.deepcopy(element_all[idx]) for idx in cand_elemnts_idx]
        for e in cand_elements:
            e.cx = random.uniform(0,1)
            e.cy = random.uniform(0,1)

        # process
        cand_elements = clamp(cand_elements)
        cand_elements = iou_remove(cand_elements)
        cand_elements = handle_overlap(cand_elements)
        cand_elements = remove_overlap(cand_elements)

        # generate real bboxes
        for idx, e in enumerate(cand_elements):
            e.gen_real_bbox()

        # calculate metric
        # align = calculate_align_score(cand_elements)
        # fill = calculate_fill_score(cand_elements)
        # layout = Layout(cand_elements=cand_elements, align=align, fill=fill)
        
        layout = Layout(cand_elements=cand_elements)
        
        # filter low-filled layouts
        if args.filter is not None:
            fill = 0
            for element in layout.cand_elements:
                w, h = element.w, element.h
                fill += w*h
            if fill < args.filter:
                continue
        
        # convert to json file
        boxes, categories = [], []
        for element in layout.cand_elements:
            cx, cy, w, h = element.get_real_bbox()
            x1, y1, x2, y2 = cx-w/2, cy-h/2, cx+w/2, cy+h/2
            boxes.append([x1, y1, x2, y2])
            categories.append(element.category-1) 
        json_layout.append({
            "boxes": boxes,
            "categories": categories
        })
        
    # dump to file
    with open(args.output_file, "w") as f:
        json.dump(json_layout, f)