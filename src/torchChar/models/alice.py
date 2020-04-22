import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from pprint import pformat
import logging
from argparse import ArgumentParser
from pytorch_lightning import LightningModule
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from ..features import load_and_cache_features


class AliceModel(LightningModule):
    def __init__(self, hparams):
        super(AliceModel, self).__init__()
        self.lr = float(hparams.lr)
        self.clogger = logging.getLogger("AliceModel")
        self.hparams = hparams
        self.hooks = {}
        self.layer_outputs = {}

        # define convolution layers
        self.conv1 = nn.Conv2d(1, hparams.n_c1, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(hparams.n_c1, hparams.n_c2, 5, padding=2)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(hparams.n_c2, hparams.n_c3, 5, padding=2)

        # define fully-connected

        self.fc1 = nn.Linear(hparams.n_c3 * 16 * 16, hparams.n_fc1)
        self.fc2 = nn.Linear(hparams.n_fc1, hparams.n_fc2)
        self.fc3 = nn.Linear(hparams.n_fc2, hparams.n_fc3)
        self.fc_radicals = nn.Linear(hparams.n_fc2, hparams.n_radicals)
        self.fc_consonants = nn.Linear(hparams.n_fc3, hparams.n_consonants)
        self.fc_vowels = nn.Linear(hparams.n_fc3, hparams.n_vowels)
        self.fc_tones = nn.Linear(hparams.n_fc3, hparams.n_tones)

    @staticmethod
    def add_model_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--n_c1', type=int, default=20, help='num of channel in conv1')
        parser.add_argument('--n_c2', type=int, default=50, help='num of channel in conv2')
        parser.add_argument('--n_c3', type=int, default=100, help='num of channel in conv3')
        parser.add_argument('--n_fc1', type=int, default=1000, help='dimension of fc1')
        parser.add_argument('--n_fc2', type=int, default=500, help='dimension of fc2')
        parser.add_argument('--n_fc3', type=int, default=200, help='dimension of fc3')
        return parser

    def register_output_hook(self, layer_name):
        if not hasattr(self, layer_name):
            return

        if layer_name in self.hooks:
            self.hooks[layer_name].remove()

        def output_hook(module, input, output):
            self.layer_outputs[layer_name] = output

        layer = getattr(self, layer_name)
        handle = layer.register_forward_hook(output_hook)
        self.hooks[layer_name] = handle

    def forward(self, bitmaps):
        assert tuple(bitmaps.shape[-2:]) == (64, 64)
        x = bitmaps.unsqueeze(1).float()
        h = self.conv1(x)
        h = F.relu(self.pool1(h))
        h = F.relu(self.conv2(h))
        h = self.pool2(h)
        h = F.relu(self.conv3(h))

        v1 = self.fc1(h.view(-1, self.fc1.in_features))
        v1 = self.fc2(F.relu(v1))
        o_radicals = self.fc_radicals(F.relu(v1))
        o_radicals = F.relu(o_radicals)

        v2 = self.fc3(F.relu(v1))
        o_tones = self.fc_tones(F.relu(v2))
        o_tones = F.relu(o_tones)
        o_consonants = self.fc_consonants(F.relu(v2))
        o_consonants = F.relu(o_consonants)
        o_vowels = self.fc_vowels(F.relu(v2))
        o_vowels = F.relu(o_vowels)

        return o_radicals, o_consonants, o_vowels, o_tones

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def train_dataloader(self):
        dataset = load_and_cache_features("train", self.hparams.subset)
        return DataLoader(dataset, batch_size=8, shuffle=True)

    def training_step(self, batch, batch_idx):
        bitmaps = batch[0]
        radicals = batch[1]
        consonants = batch[2]
        vowels = batch[3]
        tones = batch[4]

        pred_rs, pred_cs, pred_vs, pred_ts = self(bitmaps)
        r_loss = F.cross_entropy(pred_rs, radicals, reduction='sum')
        c_loss = F.cross_entropy(pred_cs, consonants, reduction='sum')
        v_loss = F.cross_entropy(pred_vs, vowels, reduction='sum')
        t_loss = F.cross_entropy(pred_ts, tones, reduction='sum')

        loss = r_loss + c_loss + t_loss + v_loss

        return {'loss': loss}

    def training_epoch_end(self, outputs):
        if len(outputs) == 0:
            return {}
        training_loss = sum(x['loss'].item() for x in outputs) / len(outputs)

        return {"log": {"training_loss": training_loss}}

    def val_dataloader(self):
        dataset = load_and_cache_features("dev", subset=self.hparams.subset)
        return DataLoader(dataset, batch_size=8, shuffle=True)

    def validation_step(self, batch, batch_idx):
        bitmaps = batch[0]
        radicals = batch[1]
        consonants = batch[2]
        vowels = batch[3]
        tones = batch[4]

        pred_rs, pred_cs, pred_vs, pred_ts = self(bitmaps)
        r_loss = F.cross_entropy(pred_rs, radicals, reduction='sum')
        c_loss = F.cross_entropy(pred_cs, consonants, reduction='sum')
        v_loss = F.cross_entropy(pred_vs, vowels, reduction='sum')
        t_loss = F.cross_entropy(pred_ts, tones, reduction='sum')

        loss = r_loss + c_loss + t_loss + v_loss

        results = {'val_loss': loss}
        results.update(self.compute_metric(radicals, pred_rs, "radicals"))
        results.update(self.compute_metric(consonants, pred_cs, "consonants"))
        results.update(self.compute_metric(vowels, pred_vs, "vowels"))
        results.update(self.compute_metric(tones, pred_ts, "tones"))

        return results

    def validation_epoch_end(self, outputs):
        if len(outputs) == 0:
            return {'progress_bar': {}, 'log': {}}

        fields = list(outputs[0].keys())
        results = {}
        for field in fields:
            results[field] = sum(x[field].item() for x in outputs) / len(outputs)

        self.clogger.info("validation results: %s\n", pformat(results))
        out_dict = {
            'val_loss': results['val_loss'],
            'progress_bar': {'val_loss': results["val_loss"]},
            'log': {'val_loss': results["val_loss"],
                    'radicals_accuracy': results["radicals_accuracy"],
                    'consonants_accuracy': results["consonants_accuracy"],
                    'vowels_accuracy': results["vowels_accuracy"],
                    'tones_accuracy': results["tones_accuracy"]}
        }
        return out_dict

    def compute_metric(self, y_true, y_score, prefix):
        y_pred = y_score.argmax(axis=1).cpu()
        y_true = y_true.cpu()
        acc = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred, average='micro')
        precision = precision_score(y_true, y_pred, average='micro')
        f1 = f1_score(y_true, y_pred, average='micro')

        return {
            f"{prefix}_accuracy": torch.tensor(acc),
            f"{prefix}_recall": torch.tensor(recall),
            f"{prefix}_precision": torch.tensor(precision),
            f"{prefix}_f1": torch.tensor(f1)
        }
