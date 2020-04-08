import numpy as np
from PIL import Image, ImageDraw
from PIL import ImageFont
from enum import Enum, auto
from pathlib import Path

class FontFamily(Enum):
    Ming = auto()
    Kai = auto()
    Yen = auto()
    FangSong = auto()

    def get_path(self):
        base_dir = Path(__file__).parent
        path_map = {
            "Ming": base_dir/"../../data/fonts/cwTeXMing-zhonly.ttf",
            "Kai": base_dir/"../../data/fonts/cwTeXKai-zhonly.ttf",
            "Yen": base_dir/"../../data/fonts/cwTeXYen-zhonly.ttf",
            "FangSong": base_dir/"../../data/fonts/cwTeXFangSong-zhonly.ttf",            
        }
        return str(path_map.get(self.name))

def step_char_wise(im, **kwargs):    
    nchar = kwargs.get("nchar", 1)

    char_width = kwargs.get("font_size", 64)
    xoffset = round(kwargs.get("xoffset", 0) * char_width)
    step_size = char_width * nchar
    step_overlap = kwargs.get("step_overlap", False)
    if step_overlap:
        steps = list(range(xoffset, im.size[0], char_width))
    else:
        steps = list(range(xoffset, im.size[0], step_size))    
    return steps, step_size
    
def measure_text(text, font_size=64, font_family=FontFamily.Ming):        
    font = ImageFont.truetype(font_family.get_path(), font_size)    
    # measure text    
    txtw, txth = font.getsize(text)
    return txtw, txth

def text2bitmap(text, im_dim=None, font_size=64, font_family=FontFamily.Ming):        
    font = ImageFont.truetype(font_family.get_path(), font_size)    

    m_dim = measure_text(text)
    im_dim = (font_size, font_size)
    # "L" for a 8bit bitmap    
    im = Image.new("L", im_dim)
    draw = ImageDraw.Draw(im) 
    h_offset = int((font_size-m_dim[0])/2)
    v_offset = int((font_size-m_dim[1])/2)    
    draw.text((h_offset, v_offset), text, (255,), font=font)    
    return im

def get_filtered_shape(img, coord):
    w, h = img.shape
    x, y, sw, sh = coord 
    
    if x > w or y > h:
        raise ValueError(f"{(x, y)} not in img dim {(w, h)}")
    if x + sw > w:
        sw = w - x
        
    if y + sh > h:
        sh = h - y
    
    return x, y, sw, sh

def spotlight_rectangle(img, coord, **kwarg):
    x, y, sw, sh = get_filtered_shape(img, coord)
    padw = 0; padh = 0
    if sw < coord[2]:
        padw = coord[2] - sw
    if sh < coord[3]:
        padh = coord[3] - sh
    
    simg = img[x:(x+sw), y:(y+sh)]
    fimg = np.pad(simg, ((0, padw), (0, padh)), "constant")
    
    return fimg