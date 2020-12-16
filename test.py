import os

import pytorch_lightning as pl
import yaml
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.dataloader import DataLoader

from utils.data_loader import get_data_word2vec
from dataset.bias_news import BiasNews
from hparams import HParams
from models.title_body_head_att import TitleBodyHeadAtt

if __name__ == "__main__":
    yaml_file = "configs/title_body_head_att.yaml"
    config = yaml.full_load(open(yaml_file))
    hparams = HParams(**config)
    model_name = config["model_name"]
    tb_logger = pl_loggers.TensorBoardLogger(f"logs/{model_name}/{config['head_num']}")
    x_data, y_data = get_data_word2vec(config["article_size"], config["title_size"])
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    accuracy = []
    for i, (train, test) in enumerate(kfold.split(x_data, y_data)):
        train_loader = DataLoader(BiasNews(x_data[train], y_data[train]), batch_size=32)
        test_loader = DataLoader(BiasNews(x_data[test], y_data[test]), batch_size=1000)
        model_dir = f"checkpoint/{model_name}/{i}/{config['head_num']}"
        if not os.path.exists(model_dir):
            break
        model_path = os.path.join(model_dir, os.listdir(model_dir)[0])
        # sk-learn provides 10-fold CV wrapper.
        # init model
        # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
        trainer = pl.Trainer(gpus=-1)
        model = TitleBodyHeadAtt.load_from_checkpoint(model_path, hparam=hparams)
        acc = trainer.test(model=model, test_dataloaders=test_loader)
        accuracy.append(acc[0]["test_acc"])
    print(f"Average test accuracy of {model_name}: {sum(accuracy) / len(accuracy)}")
    print(f"The accuracy of each fold: {accuracy}")
