# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Multiple choice fine-tuning: utilities to work with multiple choice tasks of reading comprehension """


import re
import os
import csv
import json
import glob
import tqdm
import logging


from enum import Enum
from typing import List, Optional, Callable

from filelock import FileLock
from transformers import PreTrainedTokenizer, is_tf_available, is_torch_available

from mc_transformers.featuring import convert_examples_to_features
from mc_transformers.data_classes import InputFeatures, InputExample

logger = logging.getLogger(__name__)


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


if is_torch_available():
    import torch
    from torch.utils.data.dataset import Dataset

    class MultipleChoiceDataset(Dataset):
        """
        This will be superseded by a framework-agnostic approach
        soon.
        """

        features: List[InputFeatures]

        def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            task: str,
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,
            enable_windowing: bool = False,
            stride: int = None,
            no_answer_text: str = None,
            window_fn: Callable = None,
        ):
            processor = processors[task]()

            cached_features_file = os.path.join(
                data_dir,
                "cached_{}_{}_{}_{}".format(mode.value, tokenizer.__class__.__name__, str(max_seq_length), task,),
            )

            # Make sure only the first process in distributed training processes the dataset,
            # and the others will use the cache.
            lock_path = cached_features_file + ".lock"
            with FileLock(lock_path):

                if os.path.exists(cached_features_file) and not overwrite_cache:
                    logger.info(f"Loading features from cached file {cached_features_file}")
                    self.features = torch.load(cached_features_file)
                else:
                    logger.info(f"Creating features from dataset file at {data_dir}")
                    if mode == Split.dev:
                        examples = processor.get_dev_examples(data_dir)
                    elif mode == Split.test:
                        examples = processor.get_test_examples(data_dir)
                    else:
                        examples = processor.get_train_examples(data_dir)
                    label_list = processor.get_labels()
                    logger.info("Training examples: %s", len(examples))

                    self.features = convert_examples_to_features(
                        examples,
                        label_list,
                        max_seq_length,
                        tokenizer,
                        enable_windowing=enable_windowing,
                        stride=stride,
                        no_answer_text=no_answer_text,
                        window_fn=window_fn,
                    )
                    logger.info("Saving features into cached file %s", cached_features_file)
                    torch.save(self.features, cached_features_file)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            return self.features[i]


if is_tf_available():
    import tensorflow as tf

    class TFMultipleChoiceDataset:
        """
        This will be superseded by a framework-agnostic approach
        soon.
        """

        features: List[InputFeatures]

        def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            task: str,
            max_seq_length: Optional[int] = 128,
            overwrite_cache=False,
            mode: Split = Split.train,
            enable_windowing: bool = False,
            stride: int = None,
            no_answer_text: str = None,
            window_fn: Callable = None,
        ):
            processor = processors[task]()

            logger.info(f"Creating features from dataset file at {data_dir}")
            if mode == Split.dev:
                examples = processor.get_dev_examples(data_dir)
            elif mode == Split.test:
                examples = processor.get_test_examples(data_dir)
            else:
                examples = processor.get_train_examples(data_dir)
            label_list = processor.get_labels()
            logger.info("Training examples: %s", len(examples))

            self.features = convert_examples_to_features(
                examples,
                label_list,
                max_seq_length,
                tokenizer,
                enable_windowing=enable_windowing,
                stride=stride,
                no_answer_text=no_answer_text,
                window_fn=window_fn,
            )

            def gen():
                for (ex_index, ex) in tqdm.tqdm(enumerate(self.features), desc="convert examples to features"):
                    if ex_index % 10000 == 0:
                        logger.info("Writing example %d of %d" % (ex_index, len(examples)))

                    yield (
                        {
                            "example_id": ex.example_id,
                            "input_ids": ex.input_ids,
                            "attention_mask": ex.attention_mask,
                            "token_type_ids": ex.token_type_ids,
                        },
                        ex.label,
                    )

            self.dataset = tf.data.Dataset.from_generator(
                gen,
                (
                    {
                        "example_id": tf.int32,
                        "input_ids": tf.int32,
                        "attention_mask": tf.int32,
                        "token_type_ids": tf.int32,
                    },
                    tf.int64,
                ),
                (
                    {
                        "example_id": tf.TensorShape([]),
                        "input_ids": tf.TensorShape([None, None]),
                        "attention_mask": tf.TensorShape([None, None]),
                        "token_type_ids": tf.TensorShape([None, None]),
                    },
                    tf.TensorShape([]),
                ),
            )

        def get_dataset(self):
            return self.dataset

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            return self.features[i]


class DataProcessor:
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class RaceProcessor(DataProcessor):
    """Processor for the RACE data set."""
    reg = re.compile(r'^.*/RACE/(.*)\.txt$')
    replacements = [
        ('train', '01'), ('dev', '02'), ('test', '03'),
        ('high', '04'), ('middle', '05'),
        ('/', '00')
    ]

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        high = os.path.join(data_dir, "train/high")
        middle = os.path.join(data_dir, "train/middle")
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        high = os.path.join(data_dir, "dev/high")
        middle = os.path.join(data_dir, "dev/middle")
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        high = os.path.join(data_dir, "test/high")
        middle = os.path.join(data_dir, "test/middle")
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _encode_id(self, race_id, example_id):
        """
        train/dev/test := 01, 02, 03
        high/middle    := 04, 02
        <int>.txt      := int
        /              := 00
        """
        race_id = self.reg.findall(race_id)[0]
        for repl in self.replacements:
            race_id = race_id.replace(repl[0], repl[1])
        str_example_id = str(example_id)
        if example_id < 10:
            str_example_id = '0' + str_example_id
        race_id += str_example_id
        return int(race_id)

    def _decode_id(self, race_id):
        # 0 stripped from the beggining in numbers
        race_id = '0' + str(race_id)
        example_id = race_id[-2:]
        id = race_id[8:-2] + '.txt'
        race_id = race_id[:8]
        for repl in self.replacements:
            race_id = race_id.replace(repl[1], repl[0])
        race_id += id
        return race_id, example_id

    def _read_txt(self, input_dir):
        lines = []
        files = glob.glob(input_dir + "/*txt")
        for file in tqdm.tqdm(files, desc="read files"):
            with open(file, "r", encoding="utf-8") as fin:
                data_raw = json.load(fin)
                data_raw["race_id"] = file
                lines.append(data_raw)
        return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (_, data_raw) in enumerate(lines):
            article = data_raw["article"]
            for i in range(len(data_raw["answers"])):
                race_id = self._encode_id(data_raw['race_id'], i)
                truth = str(ord(data_raw["answers"][i]) - ord("A"))
                question = data_raw["questions"][i]
                options = data_raw["options"][i]

                examples.append(
                    InputExample(
                        example_id=race_id,
                        question=question,
                        contexts=[article, article, article, article],  # this is not efficient but convenient
                        endings=[options[0], options[1], options[2], options[3]],
                        label=truth,
                    )
                )
        return examples


class SynonymProcessor(DataProcessor):
    """Processor for the Synonym data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "mctrain.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "mchp.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))

        return self._create_examples(self._read_csv(os.path.join(data_dir, "mctest.csv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3", "4"]

    def _read_csv(self, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            return list(csv.reader(f))

    def _create_examples(self, lines: List[List[str]], type: str):
        """Creates examples for the training and dev sets."""

        examples = [
            InputExample(
                example_id=line[0],
                question="",  # in the swag dataset, the
                # common beginning of each
                # choice is stored in "sent2".
                contexts=[line[1], line[1], line[1], line[1], line[1]],
                endings=[line[2], line[3], line[4], line[5], line[6]],
                label=line[7],
            )
            for line in lines  # we skip the line with the column names
        ]

        return examples


class SwagProcessor(DataProcessor):
    """Processor for the SWAG data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_csv(os.path.join(data_dir, "val.csv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        raise ValueError(
            "For swag testing, the input file does not contain a label column. It can not be tested in current code"
            "setting!"
        )
        return self._create_examples(self._read_csv(os.path.join(data_dir, "test.csv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_csv(self, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            return list(csv.reader(f))

    def _create_examples(self, lines: List[List[str]], type: str):
        """Creates examples for the training and dev sets."""
        if type == "train" and lines[0][-1] != "label":
            raise ValueError("For training, the input file must contain a label column.")

        examples = [
            InputExample(
                example_id=line[2],
                question=line[5],  # in the swag dataset, the
                # common beginning of each
                # choice is stored in "sent2".
                contexts=[line[4], line[4], line[4], line[4]],
                endings=[line[7], line[8], line[9], line[10]],
                label=line[11],
            )
            for line in lines[1:]  # we skip the line with the column names
        ]

        return examples


class ArcProcessor(DataProcessor):
    """Processor for the ARC data set (request from allennlp)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.jsonl")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "dev.jsonl")), "dev")

    def get_test_examples(self, data_dir):
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.jsonl")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_json(self, input_file):
        with open(input_file, "r", encoding="utf-8") as fin:
            lines = fin.readlines()
            return lines

    def _create_examples(self, lines, type):
        """Creates examples for the training and dev sets."""

        # There are two types of labels. They should be normalized
        def normalize(truth):
            if truth in "ABCD":
                return ord(truth) - ord("A")
            elif truth in "1234":
                return int(truth) - 1
            else:
                logger.info("truth ERROR! %s", str(truth))
                return None

        examples = []
        three_choice = 0
        four_choice = 0
        five_choice = 0
        other_choices = 0
        # we deleted example which has more than or less than four choices
        for line in tqdm.tqdm(lines, desc="read arc data"):
            data_raw = json.loads(line.strip("\n"))
            if len(data_raw["question"]["choices"]) == 3:
                three_choice += 1
                continue
            elif len(data_raw["question"]["choices"]) == 5:
                five_choice += 1
                continue
            elif len(data_raw["question"]["choices"]) != 4:
                other_choices += 1
                continue
            four_choice += 1
            truth = str(normalize(data_raw["answerKey"]))
            assert truth != "None"
            question_choices = data_raw["question"]
            question = question_choices["stem"]
            id = data_raw["id"]
            options = question_choices["choices"]
            if len(options) == 4:
                examples.append(
                    InputExample(
                        example_id=id,
                        question=question,
                        contexts=[
                            options[0]["para"].replace("_", ""),
                            options[1]["para"].replace("_", ""),
                            options[2]["para"].replace("_", ""),
                            options[3]["para"].replace("_", ""),
                        ],
                        endings=[options[0]["text"], options[1]["text"], options[2]["text"], options[3]["text"]],
                        label=truth,
                    )
                )

        if type == "train":
            assert len(examples) > 1
            assert examples[0].label is not None
        logger.info("len examples: %s}", str(len(examples)))
        logger.info("Three choices: %s", str(three_choice))
        logger.info("Five choices: %s", str(five_choice))
        logger.info("Other choices: %s", str(other_choices))
        logger.info("four choices: %s", str(four_choice))

        return examples


class GenericProcessor(DataProcessor):

    nof_labels = 4

    def _read_examples(self, file_path, set_type):
        """See base class."""
        logger.info("LOOKING AT {}".format(file_path))
        data = self._read_json(file_path)
        return self._create_examples(data, set_type)

    """Processor for the Generic data sets."""
    def get_train_examples(self, data_dir):
        data_file_path = os.path.join(data_dir, "train.json")
        return self._read_examples(data_file_path, "train")

    def get_dev_examples(self, data_dir):
        data_file_path = os.path.join(data_dir, "dev.json")
        return self._read_examples(data_file_path, "dev")

    def get_test_examples(self, data_dir):
        data_file_path = os.path.join(data_dir, "test.json")
        return self._read_examples(data_file_path, "test")

    def get_labels(self):
        """See base class."""
        return [str(lab) for lab in range(self.nof_labels)]

    def _encode_id(self, str_id, example_id):
        str_id = str(str_id)
        id = ''
        # ascii representation of an alphabet char is always a number
        # of two/three ciphers.
        for char in str_id:
            int_char = ord(char)
            # ensure 3 chars representation for each character
            str_int_char = str(int_char) if int_char > 99 else ('0' + str(int_char))
            id = '{}{}'.format(id, str_int_char)
        str_example_id = str(example_id)
        if example_id < 10:
            str_example_id = '0' + str_example_id
        id += str_example_id
        return int(id)

    def _decode_id(self, int_id):
        id = ''
        str_int_id = str(int_id)
        example_id = str_int_id[-2:]
        str_int_id = str_int_id[:-2]
        # ascii representation of an alphabet char is always a number
        # of two/three ciphers, stick to three digits
        if len(str_int_id) % 3 != 0:
            str_int_id = '0' + str_int_id

        for idx in range(0, len(str_int_id), 3):
            char = chr(int(str_int_id[idx:(idx + 3)]))
            id = '{}{}'.format(id, str(char))
        return id, example_id

    def _read_json(self, input_file):
        with open(input_file, "r", encoding="utf-8") as fin:
            data_raw = json.load(fin)
        return data_raw

    def _create_examples(self, data, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for data_raw in data['data']:
            article = data_raw['article']
            # encode by default. When input_batching with ids, no string tensor is
            # allowed, ensure that example ids are always numeric.
            for i in range(len(data_raw["answers"])):
                example_id = self._encode_id(data_raw['id'], i)
                truth = str(ord(data_raw["answers"][i]) - ord("A"))
                question = data_raw["questions"][i]
                options = data_raw["options"][i]
                self.nof_labels = len(options)

                examples.append(
                    InputExample(
                        example_id=example_id,
                        question=question,
                        contexts=[article] * len(options),
                        endings=options,
                        label=truth,
                    )
                )
        return examples


processors = {
    "race": RaceProcessor,
    "swag": SwagProcessor,
    "arc": ArcProcessor,
    "syn": SynonymProcessor,
    "generic": GenericProcessor
}


MULTIPLE_CHOICE_TASKS_NUM_LABELS = {"race", 4, "swag", 4, "arc", 4, "syn", 5, "generic", 4}
