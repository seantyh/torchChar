from argparse import ArgumentParser
from import_pkg import torchChar
import logging
import json
import pandas as pd
import pickle

logging.basicConfig(level="DEBUG", format="[%(levelname)s] %(name)s: %(message)s")
parser = ArgumentParser(description="Generate torchChar training data")

if __name__ == "__main__":
    args = parser.parse_args()    
    moe_json = torchChar.get_data_dir() / "dict-revised.json"
    
    logging.info("Loading MOE-Dict: %s", moe_json)
    with open(moe_json, "r", encoding="UTF-8") as fin:
        moe_data = json.load(fin)
    out_dir = torchChar.get_data_dir() / "train_examples"
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Building lexicon")
    lexicon, n_homophone = torchChar.build_lexicon(moe_data)
    logging.info("Lexicon contains %d characters and %d homophones", len(lexicon), n_homophone)

    radicals = set()
    consonants = set()
    vowels = set()
    tones = set()
    
    for font in torchChar.FontFamily:
        logging.info("processing %s", font.name)
        examples = torchChar.build_examples(lexicon, font)

        frame =  pd.DataFrame.from_records(
                    [(x.title, x.strokes, x.radical, x.zhuyin, x.font_family) \
                    for x in examples], 
                    columns = ["title", "strokes", "radical", "zhuyin", "font_family"])            
        frame.to_csv(out_dir/f"char_list_{font.name}.csv")

        radicals.update(x.radical for x in examples)
        consonants.update(x.consonant for x in examples)
        vowels.update(x.vowel for x in examples)
        tones.update(x.tone for x in examples)

        with (out_dir/f"char_examples_{font.name}.pkl").open("wb") as fout:
            pickle.dump(examples, fout)
        logging.info("Generated %d examples in %s", len(examples), font.name) 

    logging.info("Generating labels.json")
    with (out_dir / "labels.json").open("w", encoding="UTF-8") as fout:
        json.dump({
            "radicals": list(radicals), "consonants": list(consonants),
            "vowels": list(vowels), "tones": list(tones)
            }, fout, ensure_ascii=False, indent=2)
