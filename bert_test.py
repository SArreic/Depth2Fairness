import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from torch.utils.data.sampler import Sampler
from tqdm import tqdm
# import tensorrt as trt
from transformers import BertTokenizer, BertForSequenceClassification, AdamW


def seed_everything(seed=3125):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything()

target_column = 'target'
text_column = 'comment_text'

identity_columns = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

aux_columns = ['severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat', 'sexual_explicit']

label_columns = [target_column, 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat', 'sexual_explicit']

TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'
SUB_PATH = 'sub/bert_base_submission.csv'
# BERT_PATH = 'model/uncased_L-4_H-256_A-4'
BERT_PATH = 'model/bert-base-uncase'
output_model_file = 'bert_base_1.bin'
ENCODE_PATH = 'encoded_train_data_base.pkl'
# ENCODE_PATH = 'encoded_train_data.pkl'

encode = False

train_df = pd.read_csv(TRAIN_PATH)
train_df.fillna(0, inplace=True)

print("loaded % rrecords " % len(train_df))


def preprocess(data):
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

    def clean_special_chars(text, punct):
        for p in punct:
            text = text.replace(p, ' ')
        return text

    data = data.astype(str).apply(lambda x: clean_special_chars(x, punct))
    data = data.str.lower()
    return data


train_df[text_column] = preprocess(train_df[text_column])
train_texts = train_df[text_column]
train_labels = train_df[label_columns].values

weights = np.ones((len(train_df),)) * 0.25
num_identity = len(identity_columns)
num_df = len(train_df)

# Positive and negative examples get balanced weights in each part

hasToxicity = train_df['target'] >= 0.5
hasIdentity = train_df[identity_columns] >= 0.5

# weights[hasIdentity] = hasIdentity.mean()
weights[hasToxicity] = hasToxicity.mean() * 4

for col in identity_columns:
    hasIdentity = train_df[col] >= 0.5
    # These samples participate in the subgroup AUC and BNSP terms    
    weights[hasIdentity & ~hasToxicity] += ((hasIdentity & ~hasToxicity).sum() * num_identity) / num_df * 2
    # These samples participate in the BPSN term
    weights[~hasIdentity & hasToxicity] += ((~hasIdentity & hasToxicity).sum() * num_identity) / num_df * 2

weights = weights / weights.max()
loss_weight = weights.mean()
print("loss weight is", loss_weight)

tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
model = BertForSequenceClassification.from_pretrained(BERT_PATH, num_labels=len(label_columns))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

MAX_LEN = 172


def encode_data(texts, tokenizer, max_len):
    input_ids = []
    attention_masks = []

    for text in tqdm(texts, desc="Tokenizing", unit="text"):
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return input_ids, attention_masks


if encode is True:
    train_inputs, train_masks = encode_data(train_texts, tokenizer, MAX_LEN)

    with open(ENCODE_PATH, 'wb') as f:
        pickle.dump((train_inputs, train_masks), f)
else:
    with open(ENCODE_PATH, 'rb') as f:
        train_inputs, train_masks = pickle.load(f)

train_labels = train_df[label_columns].values
train_labels = torch.tensor(train_labels)

weights = weights.reshape((-1, 1))
weights = torch.tensor(weights)

print("data preparation done")

VALIDATION_PATH = 'data/all_data.csv'
validation_df = pd.read_csv(VALIDATION_PATH)
validation_df = validation_df[~validation_df['id'].isin(train_df['id'])]
validation_df[text_column] = preprocess(validation_df[text_column])
validation_texts = validation_df[text_column]
validation_labels = validation_df[label_columns].values
validation_inputs, validation_masks = encode_data(validation_texts, tokenizer, MAX_LEN)
validation_labels = torch.tensor(validation_labels)
validation_dataset = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_loader = DataLoader(validation_dataset, batch_size=16, shuffle=False)


class CustomSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        indices_0, indices_1 = [], []
        for i, label in enumerate(self.data_source):
            if label[2][0] < 0.5:
                indices_0.append(i)
            else:
                indices_1.append(i)
        random.shuffle(indices_0)
        random.shuffle(indices_1)

        sampled_indices = []
        for i in range(0, len(indices_0), 4):
            sampled_indices.extend(indices_0[i:i + 4])
            sampled_indices.extend(indices_1[i // 4:i // 4 + 1])

        return iter(sampled_indices)

    def __len__(self):
        return len(self.data_source)


batch_size = 48
train_dataset = TensorDataset(train_inputs, train_masks, train_labels, weights)

train_loader = DataLoader(
    train_dataset,
    # sampler = CustomSampler(train_dataset),
    sampler=RandomSampler(train_dataset),
    # shuffle=True,
    batch_size=batch_size)

# Number of training epochs (authors recommend between 2 and 4)
epochs = 2


def custom_loss(inputs, targets, weights):
    bce_loss_1 = nn.BCEWithLogitsLoss(weight=weights)(inputs, targets)
    bce_loss_2 = nn.BCEWithLogitsLoss()(inputs, targets)
    return (bce_loss_1 * 0.8) + (bce_loss_2 * 1)


def validate():
    model.eval()
    total_loss = 0
    for batch in validation_loader:
        b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]
        total_loss += loss.item()
    avg_loss = total_loss / len(validation_loader)
    return avg_loss


from torch.optim.lr_scheduler import ReduceLROnPlateau

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8, correct_bias=True)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1, verbose=True)

for epoch_i in range(0, epochs):
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    model.train()
    total_train_loss = 0

    for step, batch in enumerate(tqdm(train_loader, desc="Training", unit="batch")):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        b_weight = batch[3].to(device)

        model.zero_grad()
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        logits: object = outputs[1]
        loss = custom_loss(logits, b_labels, b_weight)
        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    avg_train_loss = total_train_loss / len(train_loader)
    print("Average training loss: {0:.2f}".format(avg_train_loss))

    # Validation step
    validation_loss = validate()
    print("Validation loss: {0:.2f}".format(validation_loss))

    # Update the learning rate
    scheduler.step(validation_loss)

torch.save(model.state_dict(), output_model_file)

test_df = pd.read_csv(TEST_PATH)
test_df[text_column] = preprocess(test_df[text_column])

test_inputs, test_masks = encode_data(test_df[text_column], tokenizer, MAX_LEN)
test_dataset = TensorDataset(test_inputs, test_masks)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

predictions = []
model.eval()

for step, batch in enumerate(tqdm(test_loader, desc="Predicting", unit="batch")):
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)

    with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        logits = outputs[0]
        sig = torch.sigmoid(logits)
        preds = sig.detach().cpu().numpy()
        predictions.append(preds)

predictions = np.concatenate(predictions, axis=0)

submission = pd.DataFrame.from_dict({
    'id': test_df['id'],
    'prediction': predictions
})

submission.to_csv(SUB_PATH, index=False)
