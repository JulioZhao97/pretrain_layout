import os
import pdb
import cv2
import tqdm
import numpy as np
import pandas as pd
import multiprocessing
from os.path import splitext
from petrel_client.client import Client

def save_to_local(client, path):
    try:
        filename = splitext(path)[0].split("/")[-1]
        if os.path.exists(os.path.join("/mnt/hwfile/opendatalab/zhaozhiyuan/laion1m", f"{filename}.jpg")):
            return
        img_bytes = client.get(path)
        img_mem_view = memoryview(img_bytes)
        img_array = np.frombuffer(img_mem_view, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        cv2.imwrite(os.path.join("/mnt/hwfile/opendatalab/zhaozhiyuan/laion1m", f"{filename}.jpg"), img)
    except BaseException:
        pass

conf_path = '~/petreloss.conf'
client = Client(conf_path) # 若不指定 conf_path ，则从 '~/petreloss.conf' 读取配置文件

laion1m = pd.read_csv("laion_1m.csv")
laion1m_images = laion1m["path"].tolist()

n_jobs = 100
with multiprocessing.Pool(n_jobs) as p:
    result = p.starmap(
        save_to_local, 
        zip([client for _ in range(len(laion1m_images))], laion1m_images)
    )
p.close()
p.join()