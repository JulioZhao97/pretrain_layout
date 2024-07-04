import fitz
from .util import *

def insert_title(page, rect, title_path, language, title_fontsize_low, title_fontsize_high, find_thr, bound_visualize):
    title = random_txt_load(text_dir=title_path,type='title')
    title_font = specify_font(language=language,text_type='title')
    final_rect = put_title(
        text=title,
        page=page,
        fontname=title_font,
        title_fontsize_low=title_fontsize_low,
        title_fontsize_high=title_fontsize_high,
        find_thr=find_thr,
        rect=rect,
        align=1)
    rect_visualize(page=page,type='text',rect=final_rect,color_dic=COLOR_DIC,bound_visualize=bound_visualize)

def put_title(text, page,fontname,title_fontsize_low,title_fontsize_high,rect,find_thr,align=1):
    length = len(text)
    for fontsize in range(title_fontsize_low, title_fontsize_high+2):
        rc = page.insert_textbox(rect, text, fontsize=fontsize, fontname=fontname,align=align)
        if rc < 0:
            if fontsize == title_fontsize_low:
                text = text[:int(length/1.5)]
                anno = page.add_redact_annot(rect)
                page.apply_redactions()
                rc = page.insert_textbox(rect, text, fontsize=fontsize, fontname=fontname,align=align)
                if rc < 0:
                    raise Exception("最小的字号都无法填充下")
                else:
                    anno = page.add_redact_annot(rect)
                    page.apply_redactions()
                    break
            else:
                anno = page.add_redact_annot(rect)
                page.apply_redactions()
                break
        else:
            anno = page.add_redact_annot(rect)
            page.apply_redactions()
    fontsize -= 1
    post_rect = title_rect_center(page=page,text=text,fontname=fontname,fontsize=fontsize,find_thr=find_thr,rect=rect)
    final_rect = find_hori_bound(page=page,rect=post_rect,text=text,fontsize=fontsize,fontname=fontname,find_thr=find_thr,align=align)
    rc = page.insert_textbox(final_rect, text, fontsize=fontsize, fontname=fontname,align=align)
    return final_rect


def title_rect_center(page,text,fontname,fontsize,find_thr,rect,align=1):
    doc1 = fitz.open()
    page1 = doc1.new_page()
    new_rect = fitz.Rect(rect[0],rect[1],rect[2],rect[3])
    endy_low = new_rect[1]
    endy_high = new_rect[3]
    while True:
        meany = (endy_low + endy_high) / 2
        new_rect[3] = meany
        rc = page.insert_textbox(new_rect, text, fontsize=fontsize, fontname=fontname,align=align)
        anno = page.add_redact_annot(new_rect)
        page.apply_redactions()
        if rc < 0:
            endy_low = meany
        else:
            endy_high =meany
        if endy_high - endy_low < find_thr:
            new_rect[3] = endy_high
            height = new_rect[3] - new_rect[1]
            new_rect[1] = (rect[1]+rect[3])/2 - height/2
            new_rect[3] = (rect[1]+rect[3])/2 + height/2
            return new_rect
        
def find_hori_bound(page,rect,text,fontsize,fontname,find_thr,align):
    new_rect = fitz.Rect(rect[0],rect[1],rect[2],rect[3])
    if align == 1:
        half = (new_rect[0] + new_rect[2])/2
        endx_low = 0
        endx_high = (new_rect[2] - new_rect[0])/2
        while True:
            meanx = (endx_low + endx_high) / 2
            new_rect[0] = half - meanx
            new_rect[2] = half + meanx
            rc = page.insert_textbox(new_rect, text, fontsize=fontsize, fontname=fontname,align=align)
            anno = page.add_redact_annot(new_rect)
            page.apply_redactions()
            if rc < 0:
                endx_low = meanx
            else:
                endx_high = meanx
            if endx_high - endx_low < find_thr:
                new_rect[0] = half - endx_high
                new_rect[2] = half + endx_high
                return new_rect
    else:
        endx_low = new_rect[0]
        endx_high = new_rect[2]
        while True:
            meanx = (endx_low + endx_high) / 2
            new_rect[2] = meanx
            rc = page.insert_textbox(new_rect, text, fontsize=fontsize, fontname=fontname,align=align)
            anno = page.add_redact_annot(new_rect)
            page.apply_redactions()
            if rc < 0:
                endx_low = meanx
            else:
                endx_high = meanx
            if endx_high - endx_low < find_thr:
                new_rect[2] = endx_high
                return new_rect