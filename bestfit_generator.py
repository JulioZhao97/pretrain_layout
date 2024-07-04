import time
import tqdm
import copy
import random
import argparse
import itertools
import multiprocessing
import datetime

from utils.generator.process import *

random.seed(datetime.datetime.now().timestamp())

def bestfit_generate(element_all):
    cand_elemnts_idx = random.sample(list(range(len(element_all))), 500)
    cand_elements = [copy.deepcopy(element_all[idx]) for idx in cand_elemnts_idx]
    put_elements = []
    e0 = random.choice(cand_elements)

    # initially, random put an element
    cx = random.uniform(min(e0.w/2, 1-e0.w/2), max(e0.w/2, 1-e0.w/2))
    cy = random.uniform(min(e0.h/2, 1-e0.h/2), max(e0.h/2, 1-e0.h/2))
    e0.cx, e0.cy = cx, cy
    put_elements = [e0]
    cand_elements.remove(e0)

    # iterativelly put element
    while True:
        # generate available bbox
        xticks, yticks = [0,1], [0,1]
        for e in put_elements:
            x1, y1, x2, y2 = e.cx-e.w/2, e.cy-e.h/2, e.cx+e.w/2, e.cy+e.h/2
            xticks.append(x1)
            xticks.append(x2)
            yticks.append(y1)
            yticks.append(y2)
        xticks, yticks = list(set(xticks)), list(set(yticks))
        pticks = list(itertools.product(xticks, yticks))
        meshgrid = list(itertools.product(pticks, pticks))

        # filter
        meshgrid = [grid for grid in meshgrid if grid[0][0] < grid[1][0] and grid[0][1] < grid[1][1]]
        put_element_boxes = []
        for e in put_elements:
            x1, y1, x2, y2 = e.cx-e.w/2, e.cy-e.h/2, e.cx+e.w/2, e.cy+e.h/2
            put_element_boxes.append([x1, y1, x2, y2])
        put_element_boxes = torch.Tensor(put_element_boxes)
        valid_grid_idx = []
        for idx, grid in enumerate(meshgrid):
            grid_bbox = torch.Tensor([grid[0][0], grid[0][1], grid[1][0], grid[1][1]]).unsqueeze(0)
            iou = torchvision.ops.box_iou(grid_bbox, put_element_boxes)
            if torch.sum(iou>0) == 0:
                valid_grid_idx.append(idx)
        meshgrid = [meshgrid[idx] for idx in valid_grid_idx]
        meshgrid = [[grid[0][0], grid[0][1], grid[1][0], grid[1][1]] for grid in meshgrid]

        # put most appropriate element
        max_fill, max_grid_idx, max_element_idx = 0, -1, -1
        for element_idx, e in enumerate(cand_elements):
            for grid_idx, grid in enumerate(meshgrid):
                if e.w > grid[2] - grid[0] or e.h > grid[3] - grid[1]:
                    continue
                element_area = e.w * e.h
                grid_area = (grid[2] - grid[0]) * (grid[3] - grid[1])
                if element_area/grid_area > max_fill:
                    max_fill = element_area/grid_area
                    max_grid_idx = grid_idx
                    max_element_idx = element_idx
        if max_element_idx == -1 or max_grid_idx == -1:
            break

        # put to top-left
        maxfit_element = cand_elements[max_element_idx]
        cand_elements.remove(maxfit_element)
        maxfit_element.cx = meshgrid[max_grid_idx][0] + maxfit_element.w/2
        maxfit_element.cy = meshgrid[max_grid_idx][1] + maxfit_element.h/2
        put_elements.append(maxfit_element)

    # get real bboxes
    for idx, e in enumerate(put_elements):
        e.gen_real_bbox()

    # calculate metric
    # align = calculate_align_score(put_elements)
    # fill = calculate_fill_score(put_elements)
    # layout = Layout(cand_elements=put_elements, align=align, fill=fill)
    layout = Layout(cand_elements=put_elements)

    # convert to json file
    boxes, categories = [], []
    for element in layout.cand_elements:
        cx, cy, w, h = element.get_real_bbox()
        x1, y1, x2, y2 = cx-w/2, cy-h/2, cx+w/2, cy+h/2
        boxes.append([x1, y1, x2, y2])
        categories.append(element.category-1) 
        
    return {
        "boxes": boxes,
        "categories": categories
    }

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', default=None, required=True, type=int, help='number of layouts to generate')
    parser.add_argument('--json-file', default=None, required=True, type=str, help='json file that contains all layout elements')
    parser.add_argument('--output-file', default='output-bestfit_layout.json', type=str, help='json file that contains all layout elements')
    args = parser.parse_args()
    
    print("Loading layout elements....")
    element_all = read_data(args.json_file)
    
    N = args.n
    # render layout        
    start = time.time()
    n_jobs = 100
    with multiprocessing.Pool(n_jobs) as p:
        generated_layout = p.starmap(
            bestfit_generate, [(element_all,) for _ in range(N)]
        )
    p.close()
    p.join()
    print(time.time() - start)
    
    # save generated layout to file
    with open(args.output_file, "w") as f:
        json.dump(generated_layout, f)