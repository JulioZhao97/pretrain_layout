import random

def sample_hw(width_range, ratio_range, max_height):
    W = random.randint(width_range[0], width_range[1])
    ratio = random.uniform(ratio_range[0], ratio_range[1])
    H = min(max_height, int(W*ratio))
    return W, H