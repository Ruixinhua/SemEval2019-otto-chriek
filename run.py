import pytorch_lightning as pl
import yaml
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.dataloader import DataLoader

from utils.data_loader import get_training_data
from dataset.bias_news import BiasNews
from hparams import HParams
from models.title_body_head_att import TitleBodyHeadAtt

if __name__ == "__main__":
    yaml_file = "configs/title_body_head_att.yaml"
    config = yaml.full_load(open(yaml_file))
    hparams = HParams(**config)
    model_name = config["model_name"]

    x_data, y_data = get_training_data(config["article_size"], config["title_size"])
    pl.seed_everything(42)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    accuracy = []
    for i, (train, test) in enumerate(kfold.split(x_data, y_data)):
        train_loader = DataLoader(BiasNews(x_data[train], y_data[train]), batch_size=32)
        test_loader = DataLoader(BiasNews(x_data[test], y_data[test]), batch_size=1000)
        # sk-learn provides 10-fold CV wrapper.
        # init model
        model = TitleBodyHeadAtt(hparam=hparams)
        checkpoint_callback = ModelCheckpoint(
            monitor="valid_acc",
            dirpath=f"checkpoint/{model_name}/{i}/{config['head_num']}",
            filename="{valid_acc:.3f}",
            save_top_k=1,
            mode="max",
        )
        tb_logger = pl_loggers.TensorBoardLogger(f"logs/{model_name}/{config['head_num']}/{i}")
        # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
        trainer = pl.Trainer(gpus=1, min_epochs=20, max_epochs=30, check_val_every_n_epoch=1, logger=tb_logger,
                             callbacks=[checkpoint_callback], deterministic=True)
        trainer.fit(model, train_loader, test_loader)
        model = TitleBodyHeadAtt.load_from_checkpoint(checkpoint_callback.best_model_path, hparam=hparams)
        model.freeze()
        acc = trainer.test(model=model, test_dataloaders=test_loader)
        accuracy.append(acc[0]["test_acc"])
    print(f"Average test accuracy of {model_name}: {sum(accuracy) / len(accuracy)}")
    print(f"The accuracy of each fold: {accuracy}")
