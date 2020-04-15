from import_pkg import torchChar
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import logging

logger = logging.getLogger()
logging.basicConfig(level="DEBUG", format="[%(levelname)s] %(name)s: %(message)s")

parser = ArgumentParser(description="training entry point")

parser.add_argument("--lr", default="1e-4")

if __name__ == "__main__":
    n_radicals = len(torchChar.Radicals())
    n_tones = len(torchChar.Tones())
    n_consonants = len(torchChar.Consonants())
    n_vowels = len(torchChar.Vowels())

    args = parser.parse_args()
    model = torchChar.AliceModel(n_radicals, 
                n_consonants, n_vowels, n_tones, **vars(args))
    trainer = Trainer(max_epochs=10)
    trainer.fit(model)

