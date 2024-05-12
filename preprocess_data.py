import re
import numpy as np
import pandas as pd
from keras.src.utils import pad_sequences
from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import stopwords
import nlpaug.augmenter.word as naw
import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Constants
MAX_NUM_WORDS = 10000
MAX_SEQUENCE_LENGTH = 250
TOXICITY_COLUMN = 'target'
TEXT_COLUMN = 'comment_text'

# 初始化词形还原器和停用词列表
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def clean_text(text):
    # 去除标点和数字
    text = re.sub(r'[\d\W]+', ' ', text)
    # 分词
    words = word_tokenize(text)
    # 去除停用词并进行词形还原
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)


# 初始化同义词替换增强器
synonym_aug = naw.SynonymAug(aug_src='wordnet')


def augment_text(texts, augmenter, num_augmented=1):
    augmented_texts = []
    for text in texts:
        augmented = augmenter.augment(text, n=num_augmented)
        augmented_texts.extend(augmented if isinstance(augmented, list) else [augmented])
    return augmented_texts


def pad_text(texts, tokenizer):
    # Pads sequences to the same length
    return pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=MAX_SEQUENCE_LENGTH)


# Data loading and preprocessing
def load_and_preprocess_data(filepath):
    # Load the data
    data = pd.read_csv(filepath)
    print(f'Loaded {len(data)} records')

    # Ensure all comment_text values are strings
    data[TEXT_COLUMN] = data[TEXT_COLUMN].astype(str)

    # 文本清洗和数据增强
    data[TEXT_COLUMN] = data[TEXT_COLUMN].apply(clean_text)
    # 假设我们只对训练集进行数据增强
    augmented_texts = augment_text(data[TEXT_COLUMN], synonym_aug)
    augmented_data = pd.DataFrame({TEXT_COLUMN: augmented_texts, TOXICITY_COLUMN: 0})  # 增强的文本暂时标记为非有害
    data = pd.concat([data, augmented_data], ignore_index=True)

    # Convert target column and identity columns to boolean
    identity_columns = [
        'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
        'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

    def convert_to_bool(df, col_name):
        df[col_name] = np.where(df[col_name] >= 0.5, True, False)

    for col in [TOXICITY_COLUMN] + identity_columns:
        convert_to_bool(data, col)

    return data


# Main function
def main():
    # Load and preprocess data
    train_data_filepath = 'Data/train.csv'
    data = load_and_preprocess_data(train_data_filepath)

    # Save the preprocessed data to a new CSV file
    preprocessed_data_filepath = 'Data/preprocessed_train.csv'
    data.to_csv(preprocessed_data_filepath, index=False)
    print(f'Preprocessed data saved to {preprocessed_data_filepath}')


if __name__ == '__main__':
    main()
