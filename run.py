from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.dataloader import DataLoader

from data_loader import get_data_word2vec
from dataset.bias_news import BiasNews
from hparams import HParams
from models.title_body_head_att import TitleBodyHeadAtt

if __name__ == '__main__':
    embed_dim, article_size, title_size = 300, 50, 20
    model_name = "title_body_attention"
    x_data, y_data = get_data_word2vec(article_size, title_size)
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    accuracy = []
    for i, (train, test) in enumerate(kfold.split(x_data, y_data)):
        train_loader = DataLoader(BiasNews(x_data[train], y_data[train]), batch_size=32)
        test_loader = DataLoader(BiasNews(x_data[test], y_data[test]), batch_size=1000)
        # sk-learn provides 10-fold CV wrapper.
        # init model
        hparams = HParams(word_emb_dim=300, head_num=20, article_size=article_size, title_size=title_size,
                          dropout=0.2, attention_hidden_dim=200, word_emb_file="utils/embedding.npy")
        model = TitleBodyHeadAtt(hparam=hparams)
        checkpoint_callback = ModelCheckpoint(
            monitor='valid_acc',
            dirpath=f'checkpoint/{model_name}/{i}',
            filename='{valid_acc:.3f}',
            save_top_k=1,
            mode='max',
        )
        tb_logger = pl_loggers.TensorBoardLogger(f'logs/{model_name}')
        # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
        trainer = pl.Trainer(gpus=-1, min_epochs=20, max_epochs=30, check_val_every_n_epoch=1, logger=tb_logger,
                             callbacks=[checkpoint_callback])
        trainer.fit(model, train_loader, test_loader)
        model = TitleBodyHeadAtt.load_from_checkpoint(checkpoint_callback.best_model_path, hparam=hparams)
        acc = trainer.test(model=model, test_dataloaders=test_loader)
        accuracy.append(acc[0]["test_acc"])
    print(f"Average test accuracy of {model_name}: {sum(accuracy) / len(accuracy)}")
    print(f"The accuracy of each fold: {accuracy}")
