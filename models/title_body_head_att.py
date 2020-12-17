import torch.nn as nn
import torch
import torch.nn.functional as F
from torchnlp.nn.attention import Attention
import pytorch_lightning as pl
import copy
import numpy as np

from models.layers import AttLayer
from transformers import BertTokenizer, BertForSequenceClassification


class TitleBodyHeadAtt(pl.LightningModule):

    def __init__(self, hparam):
        super().__init__()
        self.__dict__.update(hparam.params)
        self.accuracy = pl.metrics.Accuracy()
        self.word2vec_embedding = np.load(self.word_emb_file)
        self.embedding_layer = nn.Embedding(self.word2vec_embedding.shape[0], self.word_emb_dim).from_pretrained(
            torch.FloatTensor(self.word2vec_embedding), freeze=False)
        self.news_self_att = nn.MultiheadAttention(self.word_emb_dim, self.head_num)
        self.head_attentions = self.clones(Attention(self.word_emb_dim // self.head_num), self.head_num)
        self.attentions = self.clones(AttLayer(self.word_emb_dim // self.head_num, self.attention_hidden_dim),
                                      self.head_num)
        self.fc1 = nn.Linear(self.word_emb_dim, 128)
        self.fc2 = nn.Linear(128, 2)

    def clones(self, module, N):
        "Produce N identical layers."
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

    def reshape_head(self, y):
        return y.reshape(y.shape[0], self.head_num, -1).transpose(0, 1)

    def sentence_encoder(self, y):
        # shape of y: [N, S, E]     N: batch size, S: sequence length, E: embedded size
        y = y.transpose(0, 1)
        y = self.news_self_att(y, y, y)[0].transpose(0, 1)
        y = F.dropout(y, p=self.dropout)
        # shape of y: [N, S, E]
        y = y.view(y.shape[0], y.shape[1], self.head_num, -1)
        # shape of y: [N, S, H, D]     E is head_num * head_dim
        y = y.transpose(1, 2).transpose(0, 1)
        y = torch.stack([self_att(h) for h, self_att in zip(y, self.attentions)])
        y = y.transpose(0, 1)
        y = y.reshape(y.shape[0], self.word_emb_dim)
        # shape of y: [N, E]
        return y

    def forward(self, sequences):
        # Embedding
        sequences = F.dropout(self.embedding_layer(sequences), p=self.dropout)
        q, y = sequences[:, :self.title_size, :], sequences[:, self.title_size:, :]

        y = y.reshape((y.shape[0], self.article_size, self.title_size, self.word_emb_dim))
        # shape of y: [N, A, S, E]      A: article size
        y = y.transpose(0, 1)

        q = self.reshape_head(self.sentence_encoder(q))
        # shape of q: [N, H, D]      H: head number, D: head dimension
        y = torch.stack([self.reshape_head(self.sentence_encoder(s)) for s in y]).transpose(0, 1).transpose(1, 2)
        # shape of y: [H, N, A, D]      A: article size
        y = torch.stack([torch.squeeze(attention(torch.unsqueeze(t, 1), b)[0])
                         for t, b, attention in zip(q, y, self.head_attentions)])
        # shape of y: [H, N, D]      A: article size
        y = y.transpose(0, 1)
        y = y.reshape(y.shape[0], -1)
        # y = F.relu(self.fc2(F.relu(self.fc1(y))))
        y = torch.sigmoid(self.fc2(F.relu(self.fc1(y))))
        return y

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        sequences, y = batch
        y_hat = self(sequences)
        loss = F.cross_entropy(y_hat, y)
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        return loss

    def inference(self, batch):
        prob = torch.softmax(self(batch), dim=1)
        predicted = torch.argmax(prob, 1)
        return prob, predicted

    def validation_step(self, batch, batch_idx):
        sequences, y = batch
        labels_hat = torch.argmax(self(sequences), dim=1)
        acc = self.accuracy(labels_hat, y)
        self.log('valid_acc', acc, on_step=True, on_epoch=True)
        return acc

    def test_step(self, batch, batch_idx):
        sequences, y = batch
        labels_hat = torch.argmax(self(sequences), dim=1)
        acc = self.accuracy(labels_hat, y)
        self.log('test_acc', acc, on_step=True, on_epoch=True)
        return acc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
