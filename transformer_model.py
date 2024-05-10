import warnings

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import roc_auc_score
from tensorflow.keras.layers import Input, GlobalMaxPooling1D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, TFAutoModel

warnings.filterwarnings("ignore", category=DeprecationWarning)

MAX_NUM_WORDS = 10000
MAX_SEQUENCE_LENGTH = 250
EMBEDDINGS_PATH = 'Embedding_file/glove.6B.100d.txt'
EMBEDDINGS_DIMENSION = 100
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.00005
NUM_EPOCHS = 10
BATCH_SIZE = 128
TOXICITY_COLUMN = 'target'
TEXT_COLUMN = 'comment_text'

pretrained_model_name = 'bert-base-uncased'


def pad_text(texts, tokenizer):
    return pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=MAX_SEQUENCE_LENGTH)


def create_tokenizer(pretrained_model_name):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    return tokenizer


def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    print('loaded %d records' % len(data))

    data['comment_text'] = data['comment_text'].astype(str)

    identity_columns = [
        'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
        'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

    def convert_to_bool(df, col_name):
        df[col_name] = np.where(df[col_name] >= 0.5, True, False)

    for col in ['target'] + identity_columns:
        convert_to_bool(data, col)

    return data


def split_data(data):
    train_df, validate_df = model_selection.train_test_split(data, test_size=0.2)
    print('%d train comments, %d validate comments' % (len(train_df), len(validate_df)))
    return train_df, validate_df


def create_transformer_model(pretrained_model_name, max_length=MAX_SEQUENCE_LENGTH):
    transformer_model = TFAutoModel.from_pretrained(pretrained_model_name)

    input_ids = Input(shape=(max_length,), dtype='int32', name="input_ids")
    attention_mask = Input(shape=(max_length,), dtype='int32', name="attention_mask")

    embeddings = transformer_model(input_ids, attention_mask=attention_mask)[0]

    x = GlobalMaxPooling1D()(embeddings)
    x = Dense(128, activation='relu')(x)
    x = Dropout(DROPOUT_RATE)(x)
    preds = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[input_ids, attention_mask], outputs=preds)
    return model


# 准备Transformer模型的数据
def prepare_data_for_transformer(texts, tokenizer, max_length=MAX_SEQUENCE_LENGTH):
    # 确保文本数据是字符串列表，并且去除了NaN值
    texts = texts.fillna("NA").tolist()
    encoded_dict = tokenizer(texts, padding='max_length', truncation=True, max_length=max_length, return_tensors='tf')
    return [encoded_dict['input_ids'], encoded_dict['attention_mask']]


# 编译和训练模型
def compile_and_train_model(model, train_text, train_labels, validate_text, validate_labels):
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(learning_rate=LEARNING_RATE),
                  metrics=['acc'])

    history = model.fit(train_text, train_labels,
                        batch_size=BATCH_SIZE,
                        epochs=NUM_EPOCHS,
                        validation_data=(validate_text, validate_labels),
                        verbose=1)
    return history


# 创建和训练模型
def create_and_train_model(train_df, validate_df, tokenizer):
    # 准备数据
    train_text = prepare_data_for_transformer(train_df[TEXT_COLUMN], tokenizer)
    train_labels = train_df[TOXICITY_COLUMN].values
    validate_text = prepare_data_for_transformer(validate_df[TEXT_COLUMN], tokenizer)
    validate_labels = validate_df[TOXICITY_COLUMN].values

    # 创建模型
    model = create_transformer_model(pretrained_model_name)

    # 编译和训练模型
    history = compile_and_train_model(model, train_text, train_labels, validate_text, validate_labels)

    return model, history


def calculate_bias_metrics_for_model(dataset, model, tokenizer, identity_columns, TOXICITY_COLUMN):
    bias_metrics = {}

    for col in identity_columns:
        subgroup_mask = dataset[col]
        subgroup_examples = dataset[subgroup_mask]
        subgroup_text = prepare_data_for_transformer(subgroup_examples[TEXT_COLUMN], tokenizer)
        subgroup_labels = subgroup_examples[TOXICITY_COLUMN].values
        subgroup_scores = model.predict(subgroup_text)[:, 0]  # 修改索引，因为输出现在是(n_samples, 1)
        bias_metrics[col] = roc_auc_score(subgroup_labels, subgroup_scores)

    return bias_metrics


def power_mean(series, p):
    """计算序列的p次幂均值"""
    total = sum(np.power(series, p))
    return np.power(total / len(series), 1 / p)


def calculate_final_score(overall_auc, bias_auc_dict, POWER=-5, OVERALL_MODEL_WEIGHT=0.25):
    bias_auc_list = list(bias_auc_dict.values())
    bias_power_mean = power_mean(bias_auc_list, POWER)
    final_score = (OVERALL_MODEL_WEIGHT * overall_auc) + ((1 - OVERALL_MODEL_WEIGHT) * bias_power_mean)
    return final_score


# 预测和提交
def predict_and_submit(model, tokenizer):
    test = pd.read_csv('Data/test.csv')
    submission = pd.read_csv('Data/sample_submission.csv', index_col='id')
    test_text = prepare_data_for_transformer(test[TEXT_COLUMN], tokenizer)
    submission['prediction'] = model.predict(test_text)[:, 0]  # 修改索引，因为输出现在是(n_samples, 1)
    submission.to_csv('final_submission.csv')


# 主函数
def main():
    train_data_filepath = 'Data/train.csv'
    data = load_and_preprocess_data(train_data_filepath)

    train_df, validate_df = split_data(data)

    tokenizer = create_tokenizer(pretrained_model_name)

    model, history = create_and_train_model(train_df, validate_df, tokenizer)

    # 评估模型
    validate_text = prepare_data_for_transformer(validate_df[TEXT_COLUMN], tokenizer)
    validate_labels = validate_df[TOXICITY_COLUMN].values
    predictions = model.predict(validate_text)
    overall_auc = roc_auc_score(validate_labels, predictions)
    print(f'Validation ROC AUC: {overall_auc}')

    identity_columns = [
        'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
        'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
    ]
    bias_metrics = calculate_bias_metrics_for_model(validate_df, model, tokenizer, identity_columns, TOXICITY_COLUMN)

    final_score = calculate_final_score(overall_auc, bias_metrics)
    print(f'Final Score: {final_score}')

    predict_and_submit(model, tokenizer)  # 使用修改后的预测和提交功能


if __name__ == '__main__':
    main()
