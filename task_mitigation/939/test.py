import numpy as np 
import pandas as pd 
import random
import os
import torch
import torch.nn as nn
from tqdm import  tqdm

from transformers import BertTokenizer, BertForSequenceClassification,BertModel, Trainer, TrainingArguments,AdamW
from sklearn.model_selection import train_test_split
from sklearn import metrics
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import Sampler
from transformers import get_linear_schedule_with_warmup


TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'
SUB_PATH = 'sub/bert_base__submission.csv'
#BERT_PATH = 'model/uncased_L-4_H-256_A-4'
BERT_PATH = 'model/bert-base-uncase'
output_model_file = 'bert_base_1.bin'


target_column = 'target'
text_column = 'comment_text'

identity_columns = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

aux_columns = ['severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat','sexual_explicit']

label_columns = [target_column,'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat','sexual_explicit']


def preprocess(data):
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    def clean_special_chars(text, punct):
        for p in punct:
            text = text.replace(p, ' ')
        return text

    data = data.astype(str).apply(lambda x: clean_special_chars(x, punct))
    data = data.str.lower() 
    return data


tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
model = BertForSequenceClassification.from_pretrained(BERT_PATH, num_labels=len(label_columns))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

model.load_state_dict(torch.load(output_model_file)) 


MAX_LEN = 200
def encode_data(texts, tokenizer,max_len):
    input_ids = []
    attention_masks = []

    for text in tqdm(texts, desc="Tokenizing", unit="text"):
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks


test_df = pd.read_csv(TEST_PATH)
test_df[text_column] = preprocess(test_df[text_column])

test_inputs, test_masks = encode_data(test_df[text_column], tokenizer, MAX_LEN)
test_dataset = TensorDataset(test_inputs, test_masks)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print("start prediction")

predictions  = []
model.eval()

for step, batch in enumerate(tqdm(test_loader, desc="validation", unit="batch")):

    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)

  # 不让模型计算或存储梯度，节省内存和加速预测
    with torch.no_grad():
        outputs = model(b_input_ids,
            token_type_ids=None,
            attention_mask=b_input_mask,
            )

    logits = outputs[0]
    sig = torch.sigmoid(logits)
    #result = torch.mean(sig,-1) + sig[:,0]
    result = torch.max(sig,1)[0]
    logits = result.detach().cpu().numpy()


    predictions.extend(logits)

submission = pd.DataFrame.from_dict({
    'id': test_df['id'],
    'prediction': predictions
})

submission.to_csv(SUB_PATH, index=False)
