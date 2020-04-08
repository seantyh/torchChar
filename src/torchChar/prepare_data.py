import re
import json
import logging
import numpy as np
from dataclasses import dataclass
from .utils import get_data_dir
from .draw_text import FontFamily, text2bitmap
from zhon import zhuyin as z
from tqdm.autonotebook import tqdm

logger = logging.getLogger(__name__)

@dataclass
class Character:
    title: str
    strokes: int
    radical: str
    zhuyin: str

class InputExample:
    def __init__(self, character, font_family=FontFamily.Ming):
        self.title = character.title
        self.strokes = character.strokes
        self.radical = character.radical
        self.zhuyin = character.zhuyin
        syllable = self.parse_cv(character.zhuyin)
        self.consonant, self.vowel, self.tone = syllable
        self.font_family = font_family.name
        self.bitmap = self.generate_bitmap(self.title, font_family)
    
    def __repr__(self):
        return f"<InputExample[{self.font_family}]: {self.title}({self.zhuyin})>"

    def parse_cv(self, zhuyin):
        constant = re.findall("[ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙ]", zhuyin)
        vowel = re.findall("[ㄚㄛㄝㄜㄞㄟㄠㄡㄢㄣㄤㄥㄦㄧㄨㄩ]+", zhuyin)
        tone = re.findall(f"[{z.marks}]", zhuyin)
        consonant = constant[0] if constant else None
        vowel = vowel[0] if vowel else None
        tone = tone[0] if tone else '-'        
        return consonant, vowel, tone
    
    def generate_bitmap(self, title, font_family):
        im = text2bitmap(self.title, im_dim=(64,64), font_family=font_family)
        return np.array(im)

def build_examples(lexicon, font_family):    
    excl_glyph = "㑳" # this character is not in cwTeX font
    excl_char = [x for x in lexicon if x.title == excl_glyph][0]
    excl_ex = InputExample(excl_char)
    excl_bitmap = excl_ex.bitmap
    
    examples = []
    for char_x in tqdm(lexicon):
        ex = InputExample(char_x, font_family)
        if np.array_equal(ex.bitmap, excl_bitmap):
            continue
        examples.append(ex)
    return examples


def build_lexicon(moe_data):
    moe_json = get_data_dir() / "dict-revised.json"
    if not moe_json.exists():
        logger.error("Be sure to put MOE-dict json in the data directory")
        raise FileNotFoundError()

    # with open(moe_json, "r", encoding="UTF-8") as fin:
    #     moe_data = json.load(fin)
    lexicon = []
    n_homophone = 0    
    bopomofo_pat = z.characters
    tone_pat = z.marks
    zhuyin_pat = re.compile(rf"[{bopomofo_pat}{tone_pat} ]+")
    
    for entry in moe_data:        
        title = entry.get("title")
        if "{" in title or len(title) > 1:
            continue        
        strokes = entry.get("stroke_count")
        radical = entry.get("radical")
        heteronyms = entry.get("heteronyms", [])        

        if len([x for x in heteronyms if "bopomofo" in x]) > 1:
            n_homophone += 1

        for heter_x in heteronyms:            
            if 'bopomofo' not in heter_x: continue
            m_zhuyin = zhuyin_pat.findall(heter_x.get("bopomofo"))
            if not m_zhuyin:
                continue
            zhuyin = m_zhuyin[0]
            lexicon.append(Character(title, strokes, radical, zhuyin))

    logger.info("%d homophones in lexicon", n_homophone)
    return lexicon, n_homophone

    