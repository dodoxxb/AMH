
# imports
import torch
from transformers import BertModel, BertTokenizer, BertConfig
from torch import nn

class BertTextNet(nn.Module):
    def __init__(self, pretrained_path):
        super(BertTextNet, self).__init__()

        self.textExtractor = BertModel.from_pretrained(pretrained_path)
        embedding_dim = self.textExtractor.config.hidden_size

    def forward(self, tokens, segments, input_masks):
        output = self.textExtractor(tokens, token_type_ids=segments, attention_mask=input_masks)
        text_embeddings = output[0][:, 0, :]

        return text_embeddings


def get_embeddings(texts, pretrained_path = "../pretrained/bert/chinese_wwm_ext_pytorch"):
    textnet = BertTextNet(pretrained_path)
    tokenizer = BertTokenizer.from_pretrained(pretrained_path)
    tokens, segments, input_masks = [], [], []
    for text in texts:
        tokenized_text = tokenizer.tokenize(text)
        print("tokenized text:", tokenized_text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens.append(indexed_tokens)
        segments.append([0] * len(indexed_tokens))
        input_masks.append([1] * len(indexed_tokens))

    max_len = max([len(single) for single in tokens])
    for j in range(len(tokens)):
        padding = [0] * (max_len - len(tokens[j]))
        tokens[j] += padding
        segments[j] += padding
        input_masks[j] += padding

    tokens_tensor = torch.tensor(tokens)
    segments_tensor = torch.tensor(segments)
    input_mask_tensor = torch.tensor(input_masks)

    text_hashCodes = textnet(tokens_tensor, segments_tensor, input_mask_tensor)

    return text_hashCodes


if __name__ == "__main__":

    texts = ["[CLS]你好。[SEP]你好。[SEP]再见。[SEP]",
             "[CLS]你好。再见。[SEP]"
             ]
    # 前后要加上记号，表示是一句话。

    vecs = get_embeddings(texts)
    print(vecs[0].shape)
