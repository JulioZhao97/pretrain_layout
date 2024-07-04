# pretrain layout生成pipeline

## 使用

### 1. 生成layout

生成初始的layout，从公开数据集中采样元素（例如正文、标题、图片、表格等）然后组合成layout，参数如下:

```--json-file```: 采样元素的标注文件，COCO格式，例如publaynet的val

```--output-file```: 输出json文件

举例如下:

```bash
# 生成random类型的layout
python random_generator.py --n 1000000 --json-file /mnt/petrelfs/zhaozhiyuan/layout/LACE/datasets/publaynet-max25/raw/publaynet/val.json --filter 0.1 

# 生成bestfit类型的layout
python bestfit_generator.py --n 500000 --json-file /mnt/petrelfs/zhaozhiyuan/layout/LACE/datasets/publaynet-max25/raw/publaynet/val.json --output-file output-bestfit_layout-part1.json

python bestfit_generator.py --n 500000 --json-file /mnt/petrelfs/zhaozhiyuan/layout/LACE/datasets/publaynet-max25/raw/publaynet/val.json --output-file output-bestfit_layout-part2.json

# 合并两个layout json文件
jq -s 'flatten' output-bestfit_layout-part*.json > output-bestfit_layout.json
```

### 2. layout去重（暂未使用）

```--json-file```: 包含第一步生成的layout的json文件
```--output-file```: 去重之后的layout的json文件

```bash
python remove_duplicate.py --json-file output-bestfit_layout-part1.json --output-file output-bestfit_layout-part1-deduplicated.json
```

### 3. 渲染layout

渲染根据```config.yaml```中指定的参数来进行:

```yaml
size: # 指定尺寸
  width: [600, 1200] # 宽度的范围，从中随机选择
  ratio: [0.7, 1.5] # 长宽比范围，从中随机选择
  max_height: 3000 # 最大的高度，超过这个值会裁剪
languages: ['Ch', 'En'] # 可选的语言，从中随机选取
title: 
  method: 'figure' # 以何种方式插入，有两种分别是figure和text
  method_text:
    Ch_path: '/mnt/hwfile/opendatalab/kanghengrui/data/source/text/Chinese/title' # text方式插入text的路径
    En_path: '/mnt/hwfile/opendatalab/kanghengrui/data/source/text/English/title' # text方式插入text的路径
    fontsize: [18,36] # 标题的字体
  method_figure:
    path: '/mnt/petrelfs/zhaozhiyuan/layout/pretrain_layout/material/title' # figure方式插入，保存了所有图片的路径
    search_map: 'title_material_map.pt' # figure方式插入的mapping，从长宽比和面积映射到图片
    ratio_step: 0.2 # 隔多少step去搜索最优匹配
    area_step: 500 # 隔多少step去搜索最优匹配
maintext:
  fontsize: [10,12] # 正文字体
  Ch_path: '/mnt/hwfile/opendatalab/kanghengrui/data/source/text/Chinese/maintext' # 正文来源
  En_path: '/mnt/hwfile/opendatalab/kanghengrui/data/source/text/English/maintext' # 正文来源
image:
  laion1m:
    path: "/mnt/hwfile/opendatalab/zhaozhiyuan/laion1m" # laion1m图片路径
    search_map: "laion1m_material_map.pt" # laion1m插入的mapping，从长宽比和面积映射到图片
  chart: 
    path: "/mnt/hwfile/opendatalab/wangbin/project/pdf/chart/zhangbo/chart_data/png" # chart数据图片路径
    search_map: "chart_material_map.pt" # 图表数据插入的mapping，从长宽比和面积映射到图片
  background:
    path: "/mnt/hwfile/opendatalab/zhaozhiyuan/laion1m_background" # 背景图片路径
    search_map: "laion1m_bg_material_map.pt" # 背景图片插入的mapping，从长宽比和面积映射到图片
  ratio_step: 0.1 # 隔多少step去搜索最优匹配
  area_step: 1000 # 隔多少step去搜索最优匹配
table:
  # path: '/mnt/hwfile/opendatalab/kanghengrui/icdar_table'
  # search_map: "table_material_map.pt"
  path: '/mnt/hwfile/opendatalab/wangbin/share_data/pdf_table/train' # 表格图片路径
  search_map: "pdftable_material_map.pt" # 表格插入的mapping，从长宽比和面积映射到图片
  ratio_step: 0.2
  area_step: 4000
others:
  bg_prob: 0.5 # 插入背景的比例
  visualize: False
  find_thr: 3
  eraser_ratio: 1.2
  line_margin: 2
  page_type: ["newspaper"]

```

渲染命令参数以及示例如下：

```--json-file```: 上一步生成的layout
```--set```: 哪一部分数据，分别有```['random', 'bestfit', 'diffusion']```

```bash
python render.py --json-file output-bestfit_layout.json --set bestfit
python render.py --json-file output-diffusion_processed.json --set diffusion
python render.py --json-file output-random_layout.json --set random
```

渲染命令前加上以下命令，```--cpus-per-task=16```把CPU拉满，```-x```帮助排除一些CPU占用已经比较高的节点:
```bash
srun -N 1 --cpus-per-task=16 -p bigdata_alg -x SH-IDC1-10-140-24-[12,111,75,78,65,20] --gres=gpu:0 
```

### 4. 其他

1. sensesync快速进行删除、拷贝、统计等操作

```bash
# 删除，path为想删除的，必须/结尾
sensesync --dryrun --deleteSrc cp path/ ./srcSync/

# 统计文件夹下文件个数以及存储大小，path为想统计的，必须/结尾
sensesync --dryrun cp path/ ./srcSync/

# 拷贝数据，source_path为目标路径，target_path为目标路径
sensesync cp source_path/ target_path/
```