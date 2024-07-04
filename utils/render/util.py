import os
import random

# 初始化颜色字典
COLOR_DIC = {}
COLOR_DIC['title'] = [0,1,0]
COLOR_DIC['text'] = [0, 1, 1]
COLOR_DIC['figure'] = [139/255,26/255,26/255]

def rect_visualize(page,type,rect,color_dic,bound_visualize):
    if bound_visualize:
        annot = page.add_rect_annot(rect)  # 在矩形坐标处添加注释对象
        annot.set_colors(stroke=color_dic[type])  # 设置边框颜色
        annot.update()
    else:
        pass
    
def random_txt_load(text_dir,type):
    text_files = os.listdir(text_dir)
    text_name = random.choice(text_files)
    with open(os.path.join(text_dir,text_name), "r", encoding='utf-8') as f:  #打开文本
        text = f.read()   #读取文本
    if type != 'maintext':
        return text
    else:
        with open(os.path.join(text_dir,text_name), "r", encoding='utf-8') as f:  #打开文本
            text_list = f.readlines()   #逐行读取文本
        total_list = []
        # text = text.replace("\n", "").replace("\t", "").strip()
        # total_list.append(len(text))
        for num in range(len(text_list)):
            if num == 0:
                total_list.append(len(text_list[num]))
            else:
                total_list.append(total_list[-1] + len(text_list[num]))
        return text,total_list
    
def specify_font(language,text_type):
    if language.lower().find('c') != -1:
        return 'china-ss'
    else:
        return 'tiro'