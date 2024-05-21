import numpy as np 
import pandas as pd 
import random
import os
import torch
import torch.nn as nn

from tqdm import tqdm,trange,tnrange
import pickle
#import tensorrt as trt
from transformers import BertTokenizer, BertForSequenceClassification,BertModel, Trainer, TrainingArguments,AdamW
from sklearn.model_selection import train_test_split
from sklearn import metrics
from torch.utils.data import DataLoader, TensorDataset,RandomSampler
from torch.utils.data.sampler import Sampler
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForMaskedLM

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

aux_columns = ['severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat','sexual_explicit']

label_columns = [target_column,'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat','sexual_explicit']

TRAIN_PATH = 'data/train.csv'
TEST_PATH = 'data/test.csv'
SUB_PATH = 'sub/bert_base_submission.csv'
#BERT_PATH = 'model/uncased_L-4_H-256_A-4'
BERT_PATH = 'model/bert-base-uncase'
output_model_file = 'bert_base_1.bin'
ENCODE_PATH = 'encoded_train_data_base.pkl'
#ENCODE_PATH = 'encoded_train_data.pkl'

encode = False


train_df = pd.read_csv(TRAIN_PATH)
train_df.fillna(0, inplace = True)

print("loaded % records " % len(train_df) )


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

#weights[hasIdentity] = hasIdentity.mean()
weights[hasToxicity ] = hasToxicity.mean()* 4               


for col in identity_columns:
    hasIdentity = train_df[col] >=0.5 
    # These samples participate in the subgroup AUC and BNSP terms    
    weights[hasIdentity & ~hasToxicity]  +=   (( hasIdentity &  ~hasToxicity).sum() * num_identity) /num_df *2
    # These samples participate in the BPSN term
    weights[~hasIdentity & hasToxicity]  +=   ((~hasIdentity &  hasToxicity).sum() * num_identity) /num_df *2
    
weights = weights / weights.max()
loss_weight =  weights.mean()
print("loss weight is",loss_weight)

tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
model = BertForSequenceClassification.from_pretrained(BERT_PATH, num_labels=len(label_columns))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

MAX_LEN = 172
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

if encode is True:
    train_inputs, train_masks = encode_data(train_texts, tokenizer, MAX_LEN)

    with open(ENCODE_PATH,'wb') as f:
        pickle.dump((train_inputs,train_masks),f)
else:
    with open(ENCODE_PATH,'rb') as f:
        train_inputs, train_masks = pickle.load(f)


train_labels = train_df[label_columns].values
train_labels = torch.tensor(train_labels) 


weights = weights.reshape((-1, 1))
weights = torch.tensor(weights)

print("data preparation done")

class CustomSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        indices_0,indices_1 = [],[]
        for i, label in enumerate(self.data_source):
            if label[2][0] < 0.5:
                indices_0.append(i)
            else:
                indices_1.append(i)
        random.shuffle(indices_0)
        random.shuffle(indices_1)

        sampled_indices = []
        for i in range(0, len(indices_0), 4):
            sampled_indices.extend(indices_0[i:i+4])
            sampled_indices.extend(indices_1[i//4:i//4+1])

        return iter(sampled_indices)

    def __len__(self):
        return len(self.data_source)


batch_size = 48
train_dataset = TensorDataset(train_inputs, train_masks, train_labels, weights)

train_loader = DataLoader(
                    train_dataset, 
                   # sampler = CustomSampler(train_dataset),
                    sampler = RandomSampler(train_dataset),
                    # shuffle=True,
                    batch_size=batch_size)

optimizer = AdamW(model.parameters(),
                  lr = 2e-5, 
                  eps = 1e-8,
                  correct_bias = True 
                )

# Number of training epochs (authors recommend between 2 and 4)
epochs = 2

# Total number of training steps is number of batches * number of epochs.
total_steps = int(len(train_dataset) / batch_size * epochs)

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

def custom_loss(inputs, targets,weights):    
    bce_loss_1 = nn.BCEWithLogitsLoss(weight=weights)(inputs,targets)
    bce_loss_2 = nn.BCEWithLogitsLoss()(inputs,targets)
    return (bce_loss_1 * 0.8) + (bce_loss_2 * 1)




for epoch_i in range(0, epochs):
    
    # ========================================
    #               Training
    # ========================================
    
    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    model.train()

    for step, batch in enumerate(tqdm(train_loader, desc="Training", unit="batch")):
        
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        b_weight = batch[3].to(device)

        model.zero_grad()        
        outputs = model(
            b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask, 
                    labels=b_labels)
        
        logits = outputs[1]
    # sig = torch.sigmoid(logits)
    # logits = torch.mean(sig,-1)+sig[:,0]

        loss = custom_loss(logits,b_labels, b_weight)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()



torch.save(model.state_dict(), output_model_file)

test_df = pd.read_csv(TEST_PATH)
test_df[text_column] = preprocess(test_df[text_column])

test_inputs, test_masks = encode_data(test_df[text_column], tokenizer, MAX_LEN)
test_dataset = TensorDataset(test_inputs, test_masks)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


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
    result = torch.mean(sig,-1) + sig[:,0]
    logits = result.detach().cpu().numpy()
    

    predictions.extend(logits)

submission = pd.DataFrame.from_dict({
    'id': test_df['id'],
    'prediction': predictions
})

submission.to_csv(SUB_PATH, index=False)    
