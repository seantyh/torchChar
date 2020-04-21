from import_pkg import torchChar
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import logging

logger = logging.getLogger()
logging.basicConfig(level="DEBUG", format="[%(levelname)s] %(name)s: %(message)s")

parser = ArgumentParser(description="training entry point")
parser = torchChar.AliceModel.add_model_args(parser)

parser.add_argument("--lr", default="1e-4")
parser.add_argument("--use-cuda", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--subset", default="as")
parser.add_argument("--epochs", type=int, default=5)


if __name__ == "__main__":
    args = parser.parse_args()
    args.n_radicals = len(torchChar.Radicals())
    args.n_tones = len(torchChar.Tones())
    args.n_consonants = len(torchChar.Consonants())
    args.n_vowels = len(torchChar.Vowels())
    
    model = torchChar.AliceModel(args)
    
    use_cuda = 1 if args.use_cuda else None
    trainer = Trainer(max_epochs=args.epochs, gpus=use_cuda, fast_dev_run=args.debug)
    trainer.fit(model)

