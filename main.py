# 导入必要的库
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import model_selection
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Embedding, Input, Conv1D, GlobalMaxPooling1D, Flatten, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.initializers import Constant
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# 定义一些常量
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


def pad_text(texts, tokenizer):
    return pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=MAX_SEQUENCE_LENGTH)


# 数据加载与预处理
def load_and_preprocess_data(filepath):
    # 加载数据
    data = pd.read_csv(filepath)
    print('loaded %d records' % len(data))

    # 确保所有 comment_text 值都是字符串
    data['comment_text'] = data['comment_text'].astype(str)

    # 转换目标列和身份列为布尔值
    identity_columns = [
        'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
        'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

    def convert_to_bool(df, col_name):
        df[col_name] = np.where(df[col_name] >= 0.5, True, False)

    for col in ['target'] + identity_columns:
        convert_to_bool(data, col)

    return data


# 分割数据集
def split_data(data):
    train_df, validate_df = model_selection.train_test_split(data, test_size=0.2)
    print('%d train comments, %d validate comments' % (len(train_df), len(validate_df)))
    return train_df, validate_df


# 加载嵌入层
def load_embedding_matrix(word_index, embedding_path=EMBEDDINGS_PATH):
    # 加载预训练的词向量
    embeddings_index = {}
    with open(embedding_path, 'r', encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    # 准备嵌入矩阵
    num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, EMBEDDINGS_DIMENSION))
    for word, i in word_index.items():
        if i >= MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # 单词不在嵌入索引中的单词将为全零。
            embedding_matrix[i] = embedding_vector

    # 加载预训练的词向量作为嵌入层，设置 trainable = False 以保持嵌入层不变
    embedding_layer = Embedding(
        num_words,
        EMBEDDINGS_DIMENSION,
        embeddings_initializer=Constant(embedding_matrix),
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=False
    )
    return embedding_layer


# 定义模型结构
def create_model(word_index):
    embedding_layer = load_embedding_matrix(word_index)
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(DROPOUT_RATE)(x)
    preds = Dense(2, activation='softmax')(x)

    model = Model(sequence_input, preds)
    return model


# 编译和训练模型
def compile_and_train_model(model, train_text, train_labels, validate_text, validate_labels):
    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=LEARNING_RATE),
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
    train_text = pad_text(train_df[TEXT_COLUMN], tokenizer)
    train_labels = to_categorical(train_df[TOXICITY_COLUMN])
    validate_text = pad_text(validate_df[TEXT_COLUMN], tokenizer)
    validate_labels = to_categorical(validate_df[TOXICITY_COLUMN])

    # 创建模型
    model = create_model(tokenizer.word_index)

    # 编译和训练模型
    history = compile_and_train_model(model, train_text, train_labels, validate_text, validate_labels)

    return model, history


# 主函数
def main():
    # 加载和预处理数据
    train_data_filepath = 'Data/train.csv'
    data = load_and_preprocess_data(train_data_filepath)

    # 分割数据集
    train_df, validate_df = split_data(data)

    # 创建文本分词器
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(train_df[TEXT_COLUMN])

    # 创建和训练模型
    model, history = create_and_train_model(train_df, validate_df, tokenizer)

    # 评估模型
    validate_text = pad_sequences(tokenizer.texts_to_sequences(validate_df[TEXT_COLUMN]), maxlen=MAX_SEQUENCE_LENGTH)
    validate_labels = to_categorical(validate_df[TOXICITY_COLUMN])
    predictions = model.predict(validate_text, batch_size=BATCH_SIZE)
    roc_auc = roc_auc_score(validate_df[TOXICITY_COLUMN], predictions[:, 1])
    print(f'Validation ROC AUC: {roc_auc}')

    # 预测和提交
    test = pd.read_csv('Data/test.csv')
    submission = pd.read_csv('Data/sample_submission.csv', index_col='id')
    submission['prediction'] = model.predict(pad_text(test[TEXT_COLUMN], tokenizer))[:, 1]
    submission.to_csv('submission.csv')


if __name__ == '__main__':
    main()
