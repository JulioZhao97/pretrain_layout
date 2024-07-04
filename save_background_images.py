import os
import tqdm
import multiprocessing
from PIL import Image

def save_to_local(filename):
    try:
        image = Image.open(os.path.join("/mnt/hwfile/opendatalab/zhaozhiyuan/laion1m", filename))
        im_rgba = image.copy()
        im_rgba.putalpha(50)
        im_rgba.save(os.path.join("/mnt/hwfile/opendatalab/zhaozhiyuan/laion1m_background", filename.replace("jpg", "png")))
    except BaseException:
        pass

laion1m_images = os.listdir("/mnt/hwfile/opendatalab/zhaozhiyuan/laion1m")
n_jobs = 100
with multiprocessing.Pool(n_jobs) as p:
    result = p.map(
        save_to_local, 
        laion1m_images
    )
p.close()
p.join()
