from typing import Dict, List

import copy


class CustomGenerator:
    def __init__(self, params: Dict):
        self.params = params

    def preprocess(self, sentences: List[str]) -> List[str]:
        sentences = copy.deepcopy(sentences)
        # TODO
        return sentences

    def postprocess(self, features):
        # TODO
        return features

    def dynamic_process(self, embeddings):
        # TODO: Some inplace operation for embeddings.
        pass
