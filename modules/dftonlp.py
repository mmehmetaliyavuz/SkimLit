
import torch
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence

class nlpPudMed():
    """takes dataframes and returns data is available for nlp models"""
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        nltk.download("stopwords")
        self.STOPWORDS = stopwords.words("english")
        print(self.STOPWORDS[:5])
        self.porter = PorterStemmer()

    def encode_labels(self):
        """
        Encodes labels using LabelEncoder.

        Parameters:
        train_data (DataFrame): Training data.
        test_data (DataFrame): Test data.

        Returns:
        tuple: Tuple containing encoded labels for training and test data.
        """
        label_encoder = LabelEncoder()
        train_labels = label_encoder.fit_transform(self.train_data['type'].to_numpy())
        test_labels = label_encoder.transform(self.test_data['type'].to_numpy())
        return train_labels, test_labels

    def preprocess(self, text, stopwords=None):
        """Conditional preprocessing on our text unique to our task."""
        
        stopwords = self.STOPWORDS

        text = text.lower()

        pattern = re.compile(r"\b(" + r"|".join(stopwords) + r")\b\s*")
        text = pattern.sub("", text)

        text = re.sub(r"[)]", "", text)

        text = re.sub(r"([-;;.,!?<=>])", r" \1 ", text)
        text = re.sub("[^A-Za-z0-9]+", " ", text)
        text = re.sub(" +", " ", text)
        text = text.strip()

        return text

    def preprocess_data(self):
        """Preprocesses the text data in the dataframes."""
        preprocessed_train_data = self.train_data.copy()
        preprocessed_train_data['sentence'] = preprocessed_train_data['sentence'].apply(self.preprocess)

        preprocessed_test_data = self.test_data.copy()
        preprocessed_test_data['sentence'] = preprocessed_test_data['sentence'].apply(self.preprocess)
        train_sen = preprocessed_train_data.sentence.tolist()
        test_sen = preprocessed_test_data.sentence.tolist()
        tokenizer = get_tokenizer("spacy")
        tokenized_train = [tokenizer(sentence) for sentence in train_sen]
        vocab_builder = build_vocab_from_iterator(tokenized_train, specials=["<unk>"])
        tokenized_train = [[vocab_builder[token] for token in sentence] for sentence in tokenized_train]
        tensor_train_text = [torch.tensor(sublist) for sublist in tokenized_train]
        padded_train = pad_sequence(tensor_train_text, batch_first=True, padding_value=0)

        # Pad test data to match the length of padded_train
        max_train_length = padded_train.size(1)
        tokenized_test = [tokenizer(sentence) for sentence in test_sen]
        tokenized_test = [[vocab_builder[token] if token in vocab_builder else vocab_builder["<unk>"] for token in sentence] for sentence in tokenized_test]
        padded_test = []
        for sentence_tokens in tokenized_test:
            if len(sentence_tokens) < max_train_length:
                sentence_tokens += [0] * (max_train_length - len(sentence_tokens))
            padded_test.append(torch.tensor(sentence_tokens))
        padded_test = torch.stack(padded_test)

        return padded_train, padded_test
