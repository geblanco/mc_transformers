import torch
import numpy as np

from dataclasses import dataclass
from typing import List, Optional, Union, NamedTuple, Dict


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


class DataCollatorWithIds():
    def __init__(self):
        self.example_ids = None

    def collate(self, features: List) -> Dict[str, torch.Tensor]:
        """
        Very simple data collator that:
        - simply collates batches of dict-like objects
        - Performs special handling for potential keys named:
            - ``label``: handles a single value (int or float) per object
            - ``label_ids``: handles a list of values per object
        - does not do any additional preprocessing

        i.e., Property names of the input object will be used as corresponding inputs to the model.
        See glue and ner for example of how it's useful.
        """

        # In this function we'll make the assumption that all `features` in the batch
        # have the same attributes.
        # So we will look at the first element as a proxy for what attributes exist
        # on the whole batch.
        if not isinstance(features[0], dict):
            features = [vars(f) for f in features]

        first = features[0]
        batch = {}

        # Special handling for labels.
        # Ensure that tensor is created with the correct type
        # (it should be automatically the case, but let's make sure of it.)
        if "label" in first and first["label"] is not None:
            label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
            dtype = torch.long if isinstance(label, int) else torch.float
            batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
        elif "label_ids" in first and first["label_ids"] is not None:
            if isinstance(first["label_ids"], torch.Tensor):
                batch["labels"] = torch.stack([f["label_ids"] for f in features])
            else:
                dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
                batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

        # Skip example_ids
            # if "example_id" in first and first["example_id"] is not None:
            #     example_id = first["example_id"].item() if isinstance(first["example_id"], torch.Tensor) else first["example_id"]
            #     dtype = torch.long if isinstance(example_id, int) else torch.float
            #     batch["example_ids"] = torch.tensor([f["example_id"] for f in features], dtype=dtype)
        example_ids = [f["example_id"] for f in features]
        if self.example_ids is None:
            self.example_ids = np.array(example_ids)
        else:
            self.example_ids = np.hstack([self.example_ids, example_ids])
        # Handling of all other possible keys.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        for k, v in first.items():
            if k not in ("label", "label_ids", "example_id") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                else:
                    batch[k] = torch.tensor([f[k] for f in features], dtype=torch.long)

        return batch

    def drop_ids(self):
        self.example_ids = None


class PredictionOutputWithIds(NamedTuple):
    predictions: np.ndarray
    label_ids: Optional[np.ndarray]
    example_ids: Optional[np.ndarray]
    metrics: Optional[Dict[str, float]]
