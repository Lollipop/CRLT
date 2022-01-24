from transformers import FSMTForConditionalGeneration, FSMTTokenizer
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from typing import List

import copy
import torch
import string
import numpy as np

from nltk import data
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
data.path.append("data/nltk")


def remove_stop_word(words: List[str]) -> List[str]:
    return [word for word in words if word not in stopwords.words("english")]


def get_synonyms(word: str):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join(
                [char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    if len(synonyms) != 0:
        return list(synonyms)[np.random.randint(0, len(synonyms))]
    return None


class BaseAugmentor:
    def __init__(
        self,
        granularity: str = None,
        number: int = None,
        probability: float = None
    ):
        self.g = granularity
        self.n = number
        self.p = probability

    def preprocess(self, sentences: List[str]) -> List[str]:
        return sentences

    def postprocess(self, features):
        return features

    def dynamic_process(self, embeddings):
        pass


class DeleteAugmentor(BaseAugmentor):
    def __init__(
        self,
        granularity: str,
        probability: float
    ):
        super().__init__(
            granularity=granularity,
            probability=probability
        )

    def preprocess(self, sentences: List[str]) -> List[str]:
        if self.g == "feature" or self.g == "semantic":
            return sentences

        da_sentences = []

        for sentence in sentences:
            words = word_tokenize(sentence)
            if self.g == "word":
                reserved_words = [
                    word for word in words if np.random.uniform() > self.p]
                if len(reserved_words) == 0:
                    reserved_words = [words[np.random.randint(0, len(words))]]
                da_sentences.append(" ".join(reserved_words))
            elif self.g == "span":
                span_length = int(len(words) * self.p)
                span_length = min(span_length, len(words) - 1)
                start_idx = np.random.randint(0, len(words) - span_length + 1)
                del words[start_idx: start_idx + span_length]
                da_sentences.append(" ".join(words))

        return da_sentences

    def dynamic_process(self, embeddings):
        if self.g == "feature":
            feature_size = embeddings.shape[-1]
            cut_off_idx = [idx for idx in range(
                feature_size) if np.random.uniform() <= self.p]
            embeddings[:, :, cut_off_idx] *= 0


class InsertAugmentor(BaseAugmentor):
    def __init__(
        self,
        granularity: str,
        number: int
    ):
        super().__init__(granularity=granularity, number=number)

    def preprocess(self, sentences: List[str]) -> List[str]:
        da_sentences = []

        table = str.maketrans({key: None for key in string.punctuation})
        for sentence in sentences:
            if self.g == "word":
                words = word_tokenize(sentence)
                candidates = remove_stop_word(
                    word_tokenize(sentence.translate(table))
                )
                n = min(self.n, len(candidates))
                for _ in range(n):
                    synonyms = None
                    retry = 2 * n
                    while synonyms is None and retry > 0:
                        random_idx = np.random.randint(0, len(candidates))
                        synonyms = get_synonyms(candidates[random_idx])
                        if synonyms == None:
                            retry -= 1
                    if synonyms is not None:
                        insert_idx = np.random.randint(0, len(words))
                        words.insert(insert_idx, synonyms)
                da_sentences.append(" ".join(words))
            else:
                raise NotImplementedError(
                    f"Insert method doesn't support `{self.g}` granularity."
                )

        return da_sentences


class ReplaceAugmentor(BaseAugmentor):
    def __init__(
        self,
        granularity: str,
        number: int
    ):
        super().__init__(granularity=granularity, number=number)

    def preprocess(self, sentences: List[str]) -> List[str]:
        da_sentences = []

        for sentence in sentences:
            if self.g == "word":
                words = word_tokenize(sentence)
                candidates = remove_stop_word(word_tokenize(sentence))
                n = min(self.n, len(candidates))
                for _ in range(n):
                    random_idx = np.random.randint(0, len(candidates))
                    synonyms = get_synonyms(candidates[random_idx])
                    if synonyms is not None:
                        words[words.index(
                            candidates[random_idx])] = synonyms
                        candidates.pop(random_idx)
                da_sentences.append(" ".join(words))
            else:
                raise NotImplementedError(
                    f"Replace method doesn't support `{self.g}` granularity."
                )

        return da_sentences


class BackTranslationAugmentor(BaseAugmentor):
    def __init__(
        self,
        granularity: str,
        en_de_model_name: str = "facebook/wmt19-en-de",
        de_en_model_name: str = "facebook/wmt19-de-en",
        num_beams: int = 5,
        no_repeat_ngram_size: int = 2,
        num_return_sequences: int = 3,
        max_length: int = 128,
        device: int = None
    ):
        super().__init__(granularity)

        self.en_de_tokenizer = FSMTTokenizer.from_pretrained(
            en_de_model_name, cache_dir=f"cache/models/{en_de_model_name}"
        )
        self.en_de_model = FSMTForConditionalGeneration.from_pretrained(
            en_de_model_name, cache_dir=f"cache/models/{en_de_model_name}"
        )
        self.de_en_tokenizer = FSMTTokenizer.from_pretrained(
            de_en_model_name, cache_dir=f"cache/models/{de_en_model_name}"
        )
        self.de_en_model = FSMTForConditionalGeneration.from_pretrained(
            de_en_model_name, cache_dir=f"cache/models/{de_en_model_name}"
        )
        if device is not None:
            self.en_de_model.cuda(device)
            self.de_en_model.cuda(device)

        self.num_beams = num_beams
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.num_return_sequences = num_return_sequences
        self.max_length = max_length
        self.device = device

    def preprocess(self, sentences: List[str]) -> List[str]:
        if self.g != "semantic":
            raise NotImplementedError(
                f"Back Translation method doesn't support `{self.g}` granularity."
            )

        inputs = self.en_de_tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        if self.device is not None:
            for k, v in inputs.items():
                inputs[k] = v.cuda(self.device)
        outputs = self.en_de_model.generate(
            **inputs,
            num_beams=self.num_beams,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            num_return_sequences=self.num_return_sequences,
            max_length=self.max_length,
            early_stopping=True
        )
        decoded = self.en_de_tokenizer.batch_decode(
            outputs, skip_special_tokens=True)

        inputs = self.de_en_tokenizer(
            decoded,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        if self.device is not None:
            for k, v in inputs.items():
                inputs[k] = v.cuda(self.device)
        outputs = self.de_en_model.generate(
            **inputs,
            num_beams=self.num_beams,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            num_return_sequences=self.num_return_sequences,
            max_length=self.max_length,
            early_stopping=True
        )
        paraphrased = self.de_en_tokenizer.batch_decode(
            outputs, skip_special_tokens=True)

        da_sentences = []
        skip_size = self.num_return_sequences * self.num_return_sequences
        for i in range(0, skip_size * len(sentences), skip_size):
            da_sentences.append(
                paraphrased[i + np.random.randint(0, skip_size)]
            )

        return da_sentences


class ParaphraseAugmentor(BaseAugmentor):
    def __init__(
        self,
        granularity: str,
        model_name: str = "tuner007/pegasus_paraphrase",
        num_beams: int = 5,
        num_return_sequences: int = 5,
        max_length: int = 60,
        device: int = None
    ):
        super().__init__(granularity)

        cache_dir = f"cache/models/{model_name}"
        self.paraphrase_tokenizer = PegasusTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir)
        self.paraphrase_model = PegasusForConditionalGeneration.from_pretrained(
            model_name, cache_dir=cache_dir)
        if device is not None:
            self.paraphrase_model.cuda(device)

        self.num_beams = num_beams
        self.num_return_sequences = num_return_sequences
        self.max_length = max_length
        self.device = device

    def preprocess(self, sentences: List[str], ) -> List[str]:
        if self.g != "semantic":
            raise NotImplementedError(
                f"Paraphrase method doesn't support `{self.g}` granularity."
            )

        inputs = self.paraphrase_tokenizer(
            sentences,
            truncation=True,
            padding="longest",
            max_length=self.max_length,
            return_tensors="pt"
        )
        if self.device is not None:
            for k, v in inputs.items():
                inputs[k] = v.cuda(self.device)
        paraphrased = self.paraphrase_model.generate(
            **inputs,
            max_length=self.max_length,
            num_beams=self.num_beams,
            num_return_sequences=self.num_return_sequences,
            temperature=1.5
        )
        paraphrased = self.paraphrase_tokenizer.batch_decode(
            paraphrased, skip_special_tokens=True)

        da_sentences = []
        for i in range(len(sentences)):
            da_sentences.append(paraphrased[np.random.randint(
                i * self.num_return_sequences, (i + 1) * self.num_return_sequences)])

        return da_sentences


class MaskAugmentor(BaseAugmentor):
    def __init__(
        self,
        granularity,
        probability
    ):
        super().__init__(granularity=granularity, probability=probability)
        import torch.nn as nn
        self.dropout = nn.Dropout(probability, inplace=True)

    def postprocess(self, features):
        features["attention_mask"] = self.mask(features["attention_mask"])
        return features

    def mask(self, attention_mask: List[List[int]]) -> List[List[int]]:
        if self.g == "feature" or self.g == "semantic":
            return attention_mask

        da_attention_mask = []
        for mask in attention_mask:
            length = (np.array(mask) == 1).sum()
            if self.g == "word":
                da_attention_mask.append(
                    [1] +
                    list((np.random.uniform(size=(length-2)) > self.p).astype(int)) +
                    [1] +
                    [0] * (len(mask) - length)
                )
            elif self.g == "span":
                span_length = max(2, int(length * self.p))
                span_length = min(span_length, length - 2)
                start_idx = np.random.randint(1, length - span_length)
                mask[start_idx: start_idx+span_length] = [0] * span_length
                da_attention_mask.append(mask)

        return da_attention_mask

    def dynamic_process(self, embeddings):
        if self.g == "feature":
            self.dropout(embeddings)


class ShuffleAugmentor(BaseAugmentor):
    def __init__(
        self,
        granularity,
        number
    ):
        super().__init__(granularity=granularity, number=number)

    def postprocess(self, features):
        features["input_ids"] = self.shuffle(features["input_ids"])
        return features

    def shuffle(self, input_ids: List[List[int]]) -> List[List[int]]:
        if self.g == "feature" or self.g == "semantic":
            return input_ids

        da_input_ids = copy.deepcopy(input_ids)

        # TODO: when the granularity is `word`, we may shuffle the sub-word beacause we use input id
        #  in this method.
        for input_id in da_input_ids:
            length = (np.array(input_id) != 0).sum()
            for _ in range(self.n):
                if self.g == "word":
                    # skip specitial token.
                    random_idx_1 = np.random.randint(1, length-1)
                    random_idx_2 = random_idx_1
                    while random_idx_2 == random_idx_1:
                        random_idx_2 = np.random.randint(1, length-1)
                    input_id[random_idx_1], input_id[random_idx_2] = input_id[random_idx_2], input_id[random_idx_1]
                elif self.g == "span":
                    indexes = []
                    while len(indexes) < 4:
                        idx = np.random.randint(1, length - 1)
                        if idx not in indexes:
                            indexes.append(idx)
                    indexes.sort()

                    input_id[indexes[2]: indexes[3]], \
                        input_id[indexes[0]: indexes[1]] = \
                        input_id[indexes[0]: indexes[1]], \
                        input_id[indexes[2]: indexes[3]]

        return da_input_ids

    def dynamic_process(self, embeddings):
        feature_size = embeddings.shape[-1]
        if self.g == "feature":
            for _ in range(self.n):
                random_idx_1 = np.random.randint(feature_size)
                random_idx_2 = random_idx_1
                while random_idx_2 == random_idx_1:
                    random_idx_2 = np.random.randint(feature_size)
                embeddings[:, :, [random_idx_1, random_idx_2]] = \
                    embeddings[:, :, [random_idx_2, random_idx_1]]
