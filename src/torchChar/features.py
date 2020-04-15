import json
from typing import List
from pathlib import Path
from itertools import chain
import pickle
import torch
from tqdm.autonotebook import tqdm
from torch.utils.data import TensorDataset
from .utils import get_data_dir
from .prepare_data import InputExample, FontFamily
import logging

logger = logging.getLogger("features")

class InputFeature:
    def __init__(self, bitmap, radical, consonant, vowel, tone):
        self.bitmap = bitmap
        self.radical = radical
        self.consonant = consonant
        self.vowel = vowel
        self.tone = tone

    def __repr__(self):
        return self.to_json()
    
    def to_json(self):
        return json.dumps(self.__dict__, indent=2, sort_keys=True)

class Tones:
    def __init__(self):
        data_path = get_data_dir() / "train_examples/labels.json"
        with data_path.open("r", encoding="UTF-8") as fin:
            data = json.load(fin)
            self.tones = data.get("tones")
    
    def __len__(self):
        return len(self.tones)+1

    def decode(self, index):
        if index == 0:
            return "<NA>"
        else:
            return self.tones[index-1]
    
    def encode(self, value):
        try:
            return self.tones.index(value)+1
        except ValueError:
            return 0

class Vowels:
    def __init__(self):
        data_path = get_data_dir() / "train_examples/labels.json"
        with data_path.open("r", encoding="UTF-8") as fin:
            data = json.load(fin)
            self.vowels = data.get("vowels")
            
    def __len__(self):
        return len(self.vowels)+1

    def decode(self, index):
        if index == 0:
            return "<NA>"
        else:
            return self.vowels[index-1]
    
    def encode(self, value):
        try:
            return self.vowels.index(value)+1
        except ValueError:
            return 0

class Consonants:
    def __init__(self):
        data_path = get_data_dir() / "train_examples/labels.json"
        with data_path.open("r", encoding="UTF-8") as fin:
            data = json.load(fin)
            self.consonants = data.get("consonants")
            
    def __len__(self):
        return len(self.consonants)+1

    def decode(self, index):
        if index == 0:
            return "<NA>"
        else:
            return self.consonants[index-1]
    
    def encode(self, value):
        try:
            return self.consonants.index(value)+1
        except ValueError:
            return 0

class Radicals:
    def __init__(self):
        data_path = get_data_dir() / "train_examples/labels.json"
        with data_path.open("r", encoding="UTF-8") as fin:
            data = json.load(fin)
            self.radicals = data.get("radicals")
            
    def __len__(self):
        return len(self.radicals)+1

    def decode(self, index):
        if index == 0:
            return "<UNK>"
        else:
            return self.radicals[index-1]
    
    def encode(self, radical):
        try:
            return self.radicals.index(radical)+1
        except ValueError:
            return 0

def split_dataset(examples, mode):
    for ex_i, ex in enumerate(examples):
        if mode == "train" and ex_i % 10 < 8:
            yield ex
        elif mode == "test" and ex_i % 10 == 8:
            yield ex
        elif mode == "dev" and ex_i % 10 == 9:
            yield ex
        elif mode == "debug" and ex_i % 2000 == 0:
            yield ex
        elif mode == "debug_dev" and ex_i % 2003 == 0:
            yield ex
        else:
            pass

def convert_examples_to_features(examples: List[InputExample]):
    rvocab = Radicals()
    vvocab = Vowels()
    cvocab = Consonants()
    tvocab = Tones()
    
    features = []
    for ex in tqdm(examples, desc="converting features"):
        bitmap = ex.bitmap
        radical = rvocab.encode(ex.radical)
        consonant = cvocab.encode(ex.consonant)
        vowel = vvocab.encode(ex.vowel)
        tone = tvocab.encode(ex.tone)

        feature = InputFeature(bitmap, radical, consonant, vowel, tone)
        features.append(feature)
    return features

def load_and_cache_features(mode):
    cache_dir = get_data_dir() / "training_data"
    cache_dir.mkdir(exist_ok=True, parents=True)
    cache_file_path = cache_dir / f"cache_features_{mode}.pkl"

    if cache_file_path.exists():
        with cache_file_path.open("rb") as fin:
            features = pickle.load(fin)
    else:
        examples = []
        for font in FontFamily:            
            fname = f"train_examples/char_examples_{font.name}.pkl"
            logging.info(f"load {fname}")
            fpath = get_data_dir() / fname
            with fpath.open("rb") as fin:
                examples.append(pickle.load(fin))
        
        example_iter = chain.from_iterable(examples)
        dataset_iter = split_dataset(example_iter, mode)
        features = convert_examples_to_features(dataset_iter)
        with cache_file_path.open("wb") as fout:
            pickle.dump(features, fout)
    
    #pylint: disable=E1102
    bitmaps = torch.tensor([x.bitmap for x in features], dtype=torch.long)
    radicals = torch.tensor([x.radical for x in features], dtype=torch.long)
    consonants = torch.tensor([x.consonant for x in features], dtype=torch.long)
    vowels = torch.tensor([x.vowel for x in features], dtype=torch.long)
    tones = torch.tensor([x.tone for x in features], dtype=torch.long)
    dataset = TensorDataset(bitmaps, radicals, 
                            consonants, vowels, tones)

    return dataset
    

