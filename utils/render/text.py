import os
import pdb
import math
import fitz
import copy
import random

from .util import *

def insert_text(
    page, 
    rect, 
    maintext_path, 
    language, 
    maintext_fontsize_low, 
    maintext_fontsize_high, 
    main_text_fontsize, 
    eraser_ratio, 
    find_thr, 
    line_margin, 
    bound_visualize, 
    page_type,
    align=3,
):
    TEXT, TOTAL_LIST= random_txt_load(text_dir=maintext_path, type='maintext')
    maintext_font = specify_font(language=language, text_type='maintext')
    start, end = 0, 1
    text_bboxes = []
    put_para_text(
        page=page,
        start=start,
        end=end,
        text=TEXT,
        rect=rect, 
        fontname=maintext_font,
        fontsize=random.uniform(maintext_fontsize_low, maintext_fontsize_high),
        align=align,
        total_list=TOTAL_LIST,
        eraser_ratio=eraser_ratio,
        main_text_fontsize=main_text_fontsize,
        find_thr=find_thr,
        line_margin=line_margin,
        bound_visualize=bound_visualize,
        type=page_type,
        text_bboxes=text_bboxes,
    )
    return text_bboxes
    
def get_eraser_rect(rect, eraser_ratio, main_text_fontsize):
    eraser_rect = fitz.Rect(rect[0],rect[1],rect[2],rect[3] + eraser_ratio * main_text_fontsize) # 1.35 * ..
    return eraser_rect

def put_para_text(page,start,end,text,rect,fontname,fontsize,align,total_list,eraser_ratio,main_text_fontsize,find_thr,line_margin,bound_visualize,type,text_bboxes):
    # 确定起始位置的段落序号
    total_num_list = total_list.copy()
    total_num_list.append(start)
    total_num_list = sorted(total_num_list)
    start_row_num= total_num_list.index(start)
    # 插入起点的y坐标
    start_y = rect[1]
    # 该段的最后一个字符
    if start_row_num >= len(total_list):
        length = len(total_list)
        print(f'start_row_num:{start_row_num},length of total_list:{length}')
        print(f'start:{start},total_list:{total_list},total_num_list:{total_num_list}')
    if start_row_num != len(total_list) - 1:
        para_end = total_list[start_row_num] - 1 
    else:
        para_end = total_list[start_row_num]
    # 尝试插入当前段落
    #print(rect)
    rc = page.insert_textbox(rect,text[start:para_end],fontsize=fontsize,fontname=fontname,align=align)
    eraser_rect = get_eraser_rect(rect=rect, eraser_ratio=eraser_ratio, main_text_fontsize=main_text_fontsize)
    anno = page.add_redact_annot(eraser_rect)
    page.apply_redactions()
    rc_thr = get_rc_thr(rect=rect,type=type,fontname=fontname)
    if rc < rc_thr:
        # 当前段落都不能完全插入，直接二分寻找end字符
        end_low = start
        end_high = para_end
        start,end = binary_serach_end(page=page,end_low=end_low,end_high=end_high,text=text,rc_thr=rc_thr,rect=rect,fontname=fontname,fontsize=fontsize,align=align,main_text_fontsize=main_text_fontsize,eraser_ratio=eraser_ratio,find_thr=find_thr,line_margin=line_margin,bound_visualize=bound_visualize,type=type,text_bboxes=text_bboxes)
        return start,end
    else:
        # 能插入当前段落
        if start != para_end:
            start_y = find_endy(page=page,rect=rect,text=text[start:para_end],fontname=fontname,fontsize=fontsize,align=align,main_text_fontsize=main_text_fontsize,eraser_ratio=eraser_ratio,find_thr=find_thr,line_margin=line_margin,bound_visualize=bound_visualize,type=type,text_bboxes=text_bboxes)
        if start_row_num == len(total_list) - 1:
            # 最后一段都完美填充完，从头开始
            for row_idx in range(0,len(total_list)):
                rect = fitz.Rect(rect[0],start_y,rect[2],rect[3])
                if row_idx == 0:
                    para_start = 0
                else:
                    para_start = total_list[row_idx-1]
                if row_idx != len(total_list) - 1:
                    para_end = total_list[row_idx] - 1 
                else:
                    para_end = total_list[row_idx]
                rc = page.insert_textbox(rect,text[para_start:para_end],fontsize=fontsize,fontname=fontname,align=align)
                eraser_rect = get_eraser_rect(rect=rect, eraser_ratio=eraser_ratio, main_text_fontsize=main_text_fontsize)
                anno = page.add_redact_annot(eraser_rect)
                page.apply_redactions()
                rc_thr = get_rc_thr(rect=rect,type=type,fontname=fontname)
                if rc < rc_thr:
                    # 当前段落无法完全插入，转段内二分寻找end字符
                    end_low = para_start
                    end_high = para_end
                    start,end = binary_serach_end(page=page,end_low=end_low,end_high=end_high,text=text,rc_thr=rc_thr,rect=rect,fontname=fontname,fontsize=fontsize,align=align,main_text_fontsize=main_text_fontsize,eraser_ratio=eraser_ratio,find_thr=find_thr,line_margin=line_margin,bound_visualize=bound_visualize,type=type,text_bboxes=text_bboxes)
                    return start,end
                else:
                    # 依然能够插入整个该段落
                    start_y = find_endy(page=page,rect=rect,text=text[para_start:para_end],fontname=fontname,fontsize=fontsize,align=align,main_text_fontsize=main_text_fontsize,eraser_ratio=eraser_ratio,find_thr=find_thr,line_margin=line_margin,bound_visualize=bound_visualize,type=type,text_bboxes=text_bboxes)
                    if rect[3] - start_y < 1 * fontsize:
                        return start,min(para_end+2,len(text))
                    else:
                        continue
        elif rect[3] - start_y < 1.68 * fontsize:
            # 剩下部分不足一行，也直接return
            return start,min(para_end+2,len(text)) # 避开换行符
        else:
            # 利用循环继续考虑下面的段落
            for row_idx in range(start_row_num+1,len(total_list)):
                rect = fitz.Rect(rect[0],start_y,rect[2],rect[3])
                para_start = total_list[row_idx-1]
                if row_idx != len(total_list) - 1:
                    para_end = total_list[row_idx] - 1 
                else:
                    para_end = total_list[row_idx]
                rc = page.insert_textbox(rect,text[para_start:para_end],fontsize=fontsize,fontname=fontname,align=align)
                eraser_rect = get_eraser_rect(rect=rect, eraser_ratio=eraser_ratio, main_text_fontsize=main_text_fontsize)
                anno = page.add_redact_annot(eraser_rect)
                page.apply_redactions()
                rc_thr = get_rc_thr(rect=rect,type=type,fontname=fontname)
                if rc < rc_thr:
                    # 当前段落无法完全插入，转段内二分寻找end字符
                    end_low = para_start
                    end_high = para_end
                    start,end = binary_serach_end(page=page,end_low=end_low,end_high=end_high,text=text,rc_thr=rc_thr,rect=rect,fontname=fontname,fontsize=fontsize,align=align,main_text_fontsize=main_text_fontsize,eraser_ratio=eraser_ratio,find_thr=find_thr,line_margin=line_margin,bound_visualize=bound_visualize,type=type,text_bboxes=text_bboxes)
                    return start,end
                else:
                    # 依然能够插入整个该段落
                    start_y = find_endy(page=page,rect=rect,text=text[para_start:para_end],fontname=fontname,fontsize=fontsize,align=align,main_text_fontsize=main_text_fontsize,eraser_ratio=eraser_ratio,find_thr=find_thr,line_margin=line_margin,bound_visualize=bound_visualize,type=type,text_bboxes=text_bboxes)
                    if rect[3] - start_y < 1 * fontsize:
                        return start,min(para_end+2,len(text))
                    else:
                        continue
            for row_idx in range(0,len(total_list)):
                rect = fitz.Rect(rect[0],start_y,rect[2],rect[3])
                if row_idx == 0:
                    para_start = 0
                else:
                    para_start = total_list[row_idx-1]
                if row_idx != len(total_list) - 1:
                    para_end = total_list[row_idx] - 1 
                else:
                    para_end = total_list[row_idx]
                rc = page.insert_textbox(rect,text[para_start:para_end],fontsize=fontsize,fontname=fontname,align=align)
                eraser_rect = get_eraser_rect(rect=rect, eraser_ratio=eraser_ratio, main_text_fontsize=main_text_fontsize)
                anno = page.add_redact_annot(eraser_rect)
                page.apply_redactions()
                rc_thr = get_rc_thr(rect=rect,type=type,fontname=fontname)
                if rc < rc_thr:
                    # 当前段落无法完全插入，转段内二分寻找end字符
                    end_low = para_start
                    end_high = para_end
                    start,end = binary_serach_end(page=page,end_low=end_low,end_high=end_high,text=text,rc_thr=rc_thr,rect=rect,fontname=fontname,fontsize=fontsize,align=align,main_text_fontsize=main_text_fontsize,eraser_ratio=eraser_ratio,find_thr=find_thr,line_margin=line_margin,bound_visualize=bound_visualize,type=type,text_bboxes=text_bboxes)
                    return start,end
                else:
                    # 依然能够插入整个该段落
                    start_y = find_endy(page=page,rect=rect,text=text[para_start:para_end],fontname=fontname,fontsize=fontsize,align=align,main_text_fontsize=main_text_fontsize,eraser_ratio=eraser_ratio,find_thr=find_thr,line_margin=line_margin,bound_visualize=bound_visualize,type=type,text_bboxes=text_bboxes)
                    if rect[3] - start_y < 1 * fontsize:
                        return start,min(para_end+2,len(text))
                    else:
                        continue
                        

def get_rc_thr(rect,type,fontname):
    if type == 'newspaper':
        if fontname == 'tiro' or fontname == 'china-ss':
            if rect.height < 80:
                rc_thr = 0
            elif rect.height < 200:
                rc_thr = 2
            elif rect.height < 300:
                rc_thr = 4         # 11
            elif rect.height < 400:
                rc_thr = 14        # 37
            elif rect.height < 500:
                rc_thr = 17        # 39
            elif rect.height < 700:
                rc_thr = 23        # 49
            elif rect.height < 800:
                rc_thr = 30        # 52
            else:
                rc_thr = 35        # 55
        else:
            rc_thr = 0
    elif type == 'paper':
        if fontname == 'tiro' or fontname == 'china-ss':
            if rect.height < 200:
                rc_thr = 0
            elif rect.height < 350:
                rc_thr = 4              # 5
            elif rect.height < 420:
                rc_thr = 18             # 25
            elif rect.height < 520:
                rc_thr = 23             # 31
            elif rect.height < 600:
                rc_thr = 25             # 35
            else:
                rc_thr = 27             # 39
        else:
            rc_thr = 0
    else:
        if rect.height < 200:
            rc_thr = 0
        elif rect.height < 350:
            rc_thr = 2              # 5
        elif rect.height < 420:
            rc_thr = 12             # 25
        elif rect.height < 520:
            rc_thr = 16             # 31
        elif rect.height < 600:
            rc_thr = 18             # 35
        else:
            rc_thr = 20             # 39
    return rc_thr


def binary_serach_end(page,end_low,end_high,text,rc_thr,rect,fontname,fontsize,align,main_text_fontsize,eraser_ratio,find_thr,line_margin,bound_visualize,type,text_bboxes):
    eraser_rect = get_eraser_rect(rect=rect, eraser_ratio=eraser_ratio, main_text_fontsize=main_text_fontsize)
    start = end_low
    while True:
        mean = math.ceil((end_low + end_high) / 2)
        rc = page.insert_textbox(rect, text[start:mean], fontsize=fontsize, fontname=fontname,align = align)
        anno = page.add_redact_annot(eraser_rect)
        page.apply_redactions()
        if rc < rc_thr:
            rc1 = page.insert_textbox(rect, text[start:mean-1], fontsize=fontsize, fontname=fontname,align = align)
            if rc1 >= rc_thr:
                anno = page.add_redact_annot(eraser_rect)
                page.apply_redactions()
                endy = find_endy(page=page,rect=rect,text=text[start:mean-1],fontname=fontname,fontsize=fontsize,align=align,main_text_fontsize=main_text_fontsize,eraser_ratio=eraser_ratio,find_thr=find_thr,line_margin=line_margin,bound_visualize=bound_visualize,type=type,text_bboxes=text_bboxes)
                return start,mean
            else:
                anno = page.add_redact_annot(eraser_rect)
                page.apply_redactions()
                end_high = mean
        else:
            rc2 = page.insert_textbox(rect, text[start:mean+1], fontsize=fontsize, fontname=fontname,align = align)
            if rc2 < rc_thr:
                anno = page.add_redact_annot(eraser_rect)
                page.apply_redactions()
                endy = find_endy(page=page,rect=rect,text=text[start:mean],fontname=fontname,fontsize=fontsize,align=align,main_text_fontsize=main_text_fontsize,eraser_ratio=eraser_ratio,find_thr=find_thr,line_margin=line_margin,bound_visualize=bound_visualize,type=type,text_bboxes=text_bboxes)
                return start,mean+1
            else:
                anno = page.add_redact_annot(eraser_rect)
                page.apply_redactions()
                end_low = mean
                
                
def find_endy(page,rect,text,fontsize,fontname,align,main_text_fontsize,eraser_ratio,find_thr,line_margin,bound_visualize,type,text_bboxes):
    new_rect = fitz.Rect(rect[0],rect[1],rect[2],rect[3])
    endy_low = new_rect[1]
    endy_high = new_rect[3]
    # if rect[0]-711<2 and rect[1]-206<2:
    #     flag=True
    # if flag:
    #     pdb.set_trace()
    while True:
        # if flag:
        #     pdb.set_trace()
        meany = (endy_low + endy_high) / 2
        new_rect[3] = meany
        eraser_rect = get_eraser_rect(new_rect, eraser_ratio=eraser_ratio, main_text_fontsize=main_text_fontsize)
        rc_thr = get_rc_thr(rect=new_rect,type=type,fontname=fontname)
        rc = page.insert_textbox(new_rect, text, fontsize=fontsize, fontname=fontname,align=align)
        anno = page.add_redact_annot(eraser_rect)
        page.apply_redactions()
        if rc < rc_thr:
            endy_low = meany
        else:
            endy_high =meany
        if endy_high - endy_low < find_thr:
            new_rect[3] = min(endy_high + line_margin,rect[3])
            #new_rect[3] -= 1
            if new_rect.height <= max(find_thr + line_margin,5):
                return rect[1]
            else:
                # pdb.set_trace()
                rc = page.insert_textbox(new_rect, text, fontsize=fontsize, fontname=fontname,align=align)
                text_bboxes.append(new_rect)
                rect_visualize(page=page,type='text',rect=new_rect,color_dic=COLOR_DIC,bound_visualize=bound_visualize)
                return new_rect[3]
            
            
