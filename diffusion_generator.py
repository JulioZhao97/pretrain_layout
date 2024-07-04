import pdb
import tqdm
import copy
import argparse

from utils.generator.process import *

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-file', default=None, required=True, type=str, help='json file that contains all layout elements')
    parser.add_argument('--output-file', default='output-diffusion_processed.json', type=str, help='json file that contains all layout elements')
    args = parser.parse_args()
    
    print("Loading layout elements....")
    duffusion_layout = read_diffusion_data(args.json_file)
    
    N = len(duffusion_layout)
    json_layout = []
    for generate_idx, cand_elements in enumerate(tqdm.tqdm(duffusion_layout)):
        if len(cand_elements) <= 0:
            continue
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
        
        # convert to json file
        boxes, categories = [], []
        for element in layout.cand_elements:
            cx, cy, w, h = element.get_real_bbox()
            x1, y1, x2, y2 = cx-w/2, cy-h/2, cx+w/2, cy+h/2
            boxes.append([x1, y1, x2, y2])
            categories.append(element.category) 
        json_layout.append({
            "boxes": boxes,
            "categories": categories
        })
        
    # dump to file
    with open(args.output_file, "w") as f:
        json.dump(json_layout, f)