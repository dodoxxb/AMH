# coding = utf-8

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BERT_PRETRAINED_PATH = "hfl/chinese-roberta-wwm-ext-large"
TEXT_CSV_FILES = ["data_preprocessed/text_healthy.csv", "data_preprocessed/text_unhealthy.csv"]
MAX_LEN = 160
BATCH_SIZE = 4
NUM_EPOCHS = 10000
GRADIENT_ACCUMULATION_STEPS = 8
WARMUP_STEPS = 3