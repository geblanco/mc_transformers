import numpy as np

from dataclasses import dataclass
from typing import List, Optional, Union, NamedTuple


@dataclass(frozen=True)
class InputExample:
    """
    A single training/test example for multiple choice

    Args:
        example_id: Unique id for the example.
        question: string. The untokenized text of the second sequence (question).
        contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
        endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    example_id: Union[str, int]
    question: str
    contexts: List[str]
    endings: List[str]
    label: Optional[str]

    def todict(self):
        return dict(
            example_id=self.example_id,
            question=self.question,
            contexts=self.contexts,
            endings=self.endings,
            label=self.label,
        )


@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    example_id: Union[str, int]
    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    label: Optional[int]


class WindowPrediction(NamedTuple):
    predictions: np.ndarray
    window_ids: List[int]
    labels: List[int]
    label: Optional[int]
    example: Optional[InputExample]

    def todict(self):
        return dict(
            predictions=self.predictions.tolist(),
            window_ids=self.window_ids,
            labels=self.labels,
            label=self.label,
            example=(
                self.example.todict() if self.example is not None else None
            )
        )
