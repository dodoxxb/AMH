# coding = utf-8

# imports
import numpy as np
import pandas as pd
from data_util import TextDataset
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from transformers import BertTokenizer, BertModel
from ft_config import *
from tqdm import trange, tqdm


class SentenceEmbeddingGenerator:
    """ """
    def __init__(self, pretrained_path):
        super(SentenceEmbeddingGenerator, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_path)
        self.bert.to(DEVICE)
        self.bert.train(False)

    def gen_embedding(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None):
        sentence_embedding = self.bert(input_ids, position_ids=position_ids,
                                       token_type_ids=token_type_ids,
                                       attention_mask=attention_mask,
                                       head_mask=head_mask)
        return sentence_embedding[0][:, 0, :]  # 第一个token CLS的Embedding可以用来分类


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained(BERT_PRETRAINED_PATH)
    dataset = TextDataset(csv_files=TEXT_CSV_FILES, tokenizer=tokenizer, train=False)

    model = SentenceEmbeddingGenerator(BERT_PRETRAINED_PATH)
    dataset_size = len(dataset)
    train_indices = list(range(dataset_size))[:dataset.num_neg]
    train_sampler = SubsetRandomSampler(train_indices)
    train_loader = DataLoader(dataset, batch_size=1, sampler=train_sampler)
    train_iterator = tqdm(train_loader, desc="Train Iteration")

    unique_ids, labels, embeddings = [], [], []
    for step, batch in enumerate(train_iterator):
        inputs = {
            "input_ids": batch[0].to(DEVICE),
            "token_type_ids": batch[1].to(DEVICE),
            "attention_mask": batch[2].to(DEVICE)
        }
        label = batch[3].to(DEVICE)
        unique_id = batch[4]
        embedding = model.gen_embedding(**inputs)

        for item in label.data:
            labels.append(item.cpu().numpy())
        for item in unique_id:
            unique_ids.append(item)
        for item in embedding.data:
            embeddings.append(item.cpu().numpy())

    embeddings = np.array(embeddings)
    np.savetxt("./text_feature_healthy_20211219.csv", embeddings, delimiter=",")
    # df = {"unique_id": unique_ids, "label": labels}
    # df = pd.DataFrame(df)
    # df.to_csv("./text_id_label_20211219.csv")

    dataset_size = len(dataset)
    train_indices = list(range(dataset_size))[dataset.num_neg:]
    train_sampler = SubsetRandomSampler(train_indices)
    train_loader = DataLoader(dataset, batch_size=1, sampler=train_sampler)
    train_iterator = tqdm(train_loader, desc="Train Iteration")

    unique_ids, labels, embeddings = [], [], []
    for step, batch in enumerate(train_iterator):
        inputs = {
            "input_ids": batch[0].to(DEVICE),
            "token_type_ids": batch[1].to(DEVICE),
            "attention_mask": batch[2].to(DEVICE)
        }
        label = batch[3].to(DEVICE)
        unique_id = batch[4]
        embedding = model.gen_embedding(**inputs)

        for item in label.data:
            labels.append(item.cpu().numpy())
        for item in unique_id:
            unique_ids.append(item)
        for item in embedding.data:
            embeddings.append(item.cpu().numpy())

    embeddings = np.array(embeddings)
    np.savetxt("./text_feature_unhealthy_20211219.csv", embeddings, delimiter=",")
    print("finish")
