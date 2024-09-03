# coding = utf-8

# imports
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler, WeightedRandomSampler
from data_util import TextDataset
from transformers import BertTokenizer, BertModel, BertConfig, get_linear_schedule_with_warmup
# from pytorch_transformers import WarmupLinearSchedule
from ft_config import *
from tqdm import trange, tqdm


class FnClassifier(nn.Module):
    """ """
    def __init__(self, dims):
        super(FnClassifier, self).__init__()
        self.fn1 = nn.Linear(dims[0], dims[1])
        self.fn2 = nn.Linear(dims[1], dims[2])
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        [torch.nn.init.xavier_normal_(item.weight) for item in [self.fn1, self.fn2]]

    def forward(self, x):
        x = self.fn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fn2(x)
        return x


class SentenceCls(nn.Module):
    """ """
    def __init__(self, pretrained_path):
        super(SentenceCls, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_path)
        embedding_dim = self.bert.config.hidden_size
        self.dropout = nn.Dropout(0.3)
        self.classifier = FnClassifier([embedding_dim, 1024, 2])

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, head_mask=None):
        sentence_embedding = self.bert(input_ids, position_ids=position_ids,
                                       token_type_ids=token_type_ids,
                                       attention_mask=attention_mask,
                                       head_mask=head_mask)
        x = sentence_embedding[-1]
        x = self.dropout(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained(BERT_PRETRAINED_PATH)
    dataset = TextDataset(csv_files=TEXT_CSV_FILES, tokenizer=tokenizer, train=True)

    model = SentenceCls(BERT_PRETRAINED_PATH)
    model.to(DEVICE)

    validation_split = 0.2
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    shuffle_dataset = True

    if shuffle_dataset:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=validation_sampler)

    print("Training Set Size{}, Valid Set Size{}".format(len(train_indices), len(val_indices)))

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam([
        {"params": model.bert.parameters(), "lr": 1e-5},
        {"params": model.classifier.parameters(), "lr": 4e-5}
    ])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS,
                                                num_training_steps=len(train_loader)//GRADIENT_ACCUMULATION_STEPS)
    model.zero_grad()
    epoch_iterator = trange(int(NUM_EPOCHS), desc="Epoch")
    training_acc_list, val_acc_list = [], []
    best_train_acc = 0.0
    best_val_acc = 0.0

    for epoch in epoch_iterator:
        epoch_loss = 0.0
        train_correct_total = 0

        train_iterator = tqdm(train_loader, desc="Train Iteration")
        for step, batch in enumerate(train_iterator):
            model.train(True)
            inputs = {
                "input_ids": batch[0].to(DEVICE),
                "token_type_ids": batch[1].to(DEVICE),
                "attention_mask": batch[2].to(DEVICE)
            }

            labels = batch[3].to(DEVICE)
            result = model(**inputs)

            loss = criterion(result, labels) / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            epoch_loss += loss.item()

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            _, predicted = torch.max(result.data, 1)
            correct_reviews_in_batch = (predicted == labels).sum().item()
            train_correct_total += correct_reviews_in_batch

        print("Epoch {} - Loss {}".format(epoch + 1, epoch_loss / len(train_indices)))

        with torch.no_grad():
            val_correct_total = 0
            model.train(False)
            val_iterator = tqdm(val_loader, desc="Validation Iteration")
            for step, batch in enumerate(val_iterator):
                inputs = {
                    "input_ids": batch[0].to(DEVICE),
                    "token_type_ids": batch[1].to(DEVICE),
                    "attention_mask": batch[2].to(DEVICE)
                }

                labels = batch[3].to(DEVICE)
                result = model(**inputs)

                _, predicted = torch.max(result.data, 1)
                correct_reviews_in_batch = (predicted == labels).sum().item()
                val_correct_total += correct_reviews_in_batch

            current_train_acc = train_correct_total * 100 / len(train_indices)
            current_val_acc = val_correct_total * 100 / len(val_indices)
            training_acc_list.append(current_train_acc)
            val_acc_list.append(current_val_acc)
            if current_train_acc >= best_train_acc and current_val_acc >= best_val_acc:
                best_val_acc = current_val_acc
                best_train_acc = current_train_acc
                torch.save(model, "text_ft_bert.pth")
            print("Training Acc {:.4f} - Val Acc {:.4f}".format(current_train_acc, current_val_acc))
    print("The best acc of training is {}, the best validation acc is {}".format(best_train_acc, best_val_acc))