from .data_augmentor import (
    BaseAugmentor,
    DeleteAugmentor,
    InsertAugmentor,
    ReplaceAugmentor,
    BackTranslationAugmentor,
    ParaphraseAugmentor,
    MaskAugmentor,
    ShuffleAugmentor
)
from .custom_generator import CustomGenerator

import copy
from typing import List, Dict


def get_funcs(params: Dict):
    funcs = []

    if params is None:
        return [BaseAugmentor()]

    for k, v in params.items():
        if k == "dropout":
            assert v is None, "Dropout parameters should be setted by dropout_prob in training arguments."
            funcs.append(BaseAugmentor())
        elif k == "delete" or k == "cut_off":
            assert "granularity" in v and "probability" in v, \
                "Delete need set granularity and probability."
            funcs.append(
                DeleteAugmentor(
                    granularity=v["granularity"],
                    probability=v["probability"]
                )
            )
        elif k == "insert":
            assert "granularity" in v and "number" in v, \
                "Insert need set granularity and number."
            funcs.append(
                InsertAugmentor(
                    granularity=v["granularity"],
                    number=v["number"]
                )
            )
        elif k == "replace":
            assert "granularity" in v and "number" in v, \
                "Replace need set granularity and number."
            funcs.append(
                ReplaceAugmentor(
                    granularity=v["granularity"],
                    number=v["number"]
                )
            )
        elif k == "back-translation":
            assert "granularity" in v and "device" in v, \
                "Back-translation need set granularity and device."
            funcs.append(
                BackTranslationAugmentor(
                    granularity=v["granularity"],
                    device=v["device"]
                )
            )
        elif k == "paraphrase":
            assert "granularity" in v and "device" in v, \
                "Paraphrase need set granularity and device."
            funcs.append(
                ParaphraseAugmentor(
                    granularity=v["granularity"],
                    device=v["device"]
                )
            )
        elif k == "mask":
            assert "granularity" in v and "probability" in v, \
                "Mask need set granularity and probability."
            funcs.append(
                MaskAugmentor(
                    granularity=v["granularity"],
                    probability=v["probability"]
                )
            )
        elif k == "shuffle":
            assert "granularity" in v and "number" in v, \
                "Shuffle need set granularity and number."
            funcs.append(
                ShuffleAugmentor(
                    granularity=v["granularity"],
                    number=v["number"]
                )
            )
        elif k == "custom":
            funcs.append(
                CustomGenerator(v)
            )

    return funcs


class Generator:

    def __init__(self, params: Dict):
        self.generator = get_funcs(params)[0]

    def preprocess(self, sentences: List[str]) -> List[str]:
        return self.generator.preprocess(copy.deepcopy(sentences))

    def postprocess(self, features):
        return self.generator.postprocess(features)

    def dynamic_process(self, embeddings):
        self.generator.dynamic_process(embeddings)


if __name__ == "__main__":
    import torch
    import copy
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "/share/model/transformers/bert/uncased_L-12_H-768_A-12")

    query_params = {
        "shuffle": {
            "granularity": "feature",
            "number": 5
        }
        # pass
        # "mask": {
        #     "granularity": "feature",
        #     "probability": 0.1
        # }
        # "paraphrase": {
        #     "granularity": "semantic",
        #     "device": 0
        # },
        # pass
        # "back-translation": {
        #     "granularity": "semantic",
        #     "device": 0
        # },
        # pass
        # "replace": {
        #     "granularity": "word",
        #     "number": 3
        # },
        # pass
        # "insert": {
        #     "granularity": "word",
        #     "number": 3
        # },
        # pass
        # "delete": {
        #     "granularity": "feature",
        #     "probability": 0.5
        # }
    }
    # query_params = None
    sentences = [
        "In 2018, Cain was elected to the Board of Directors of the National Rifle Association.",
        "On June 19, 2018, Cain was sworn in as a reserve police officer for the St. Anthony Police Department in St. Anthony, Idaho."
    ]
    features = tokenizer(sentences, padding=True)
    embeddings = torch.randn((5, 10, 12))
    flag_embeddings = copy.deepcopy(embeddings)

    generator = Generator(query_params)

    print(generator.preprocess(sentences))
    print(sentences)
    print(features)
    print(generator.postprocess(features))
    generator.dynamic_process(embeddings)
    # print(embeddings)
    # print("=" * 50)
    # print(flag_embeddings)
    print(embeddings != flag_embeddings)

    print("pass!")
