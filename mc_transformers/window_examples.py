import os
import sys
import tqdm
import json

from pathlib import Path
from typing import List, Callable, Optional
from dataclasses import dataclass, field
from transformers import PreTrainedTokenizer
from mc_transformers.data_classes import InputExample
from mc_transformers.utils_mc import Split, processors
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)
from mc_transformers.featuring import (
    should_window,
    create_windows,
    should_correct_label,
    correct_label,
    TextTokenizer,
)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(processors.keys())})
    data_dir: str = field(metadata={"help": "Should contain the data files for the task."})
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


@dataclass
class WindowArguments:
    """
    Arguments pertaining to output directories for metrics, results and predictions
    """
    enable_windowing: bool = field(
        default=False,
        metadata={
            'help': 'Enable windowing system alltogether'
        }
    )
    stride: Optional[int] = field(
        default=None,
        metadata={
            "help": "Stride to use when windowing features"
        }
    )
    no_answer_text: Optional[str] = field(
        default=None,
        metadata={
            "help": "Text of an unanswerable question option (Triggers "
            "label correction mechanism on windowed features)"
        }
    )


def windowed_tokenization(
    example: InputExample,
    label_map: dict,
    max_window_length: int,
    max_length: int,
    stride: int,
    no_answer_text: str,
    tokenizer: PreTrainedTokenizer,
    text_tokenizer: TextTokenizer,
    window_fn: Callable = None
) -> List[InputExample]:
    # ToDo := Different amount of windows will trigger an error because of
    # different size in input features? sequences should be grouped by
    # size and chopped, padded accordingly

    # ToDo := no_answer_text is not used by now, no label corrected
    able_to_correct_label = no_answer_text is not None and no_answer_text != ""
    window_fn = window_fn if window_fn is not None else create_windows
    window_texts = window_fn(
        example.contexts[0], tokenizer, max_window_length, stride
    )
    windowed_examples = []
    for win_idx, win_text in enumerate(window_texts):
        str_win_idx = str(win_idx)
        if len(str_win_idx) % 2 != 0:
            str_win_idx = '0' + str_win_idx

        if able_to_correct_label and should_correct_label(
            win_text, example.endings, no_answer_text, text_tokenizer
        ):
            label, endings = correct_label(
                win_text,
                example.endings.copy(),
                no_answer_text,
                text_tokenizer
            )
        else:
            label = label_map[example.label]
            endings = example.endings

        # maximum 100 windows
        example_id = int(str(example.example_id) + str_win_idx)
        windowed_examples.append(InputExample(
            example_id=example_id,
            question=example.question,
            contexts=[win_text] * len(endings),
            endings=endings,
            label=label,
        ))

    return windowed_examples


def window_examples(
    examples: List[InputExample],
    label_list: List[str],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
    enable_windowing: bool = False,
    stride: int = None,
    no_answer_text: str = None,
    window_fn: Callable = None,
) -> List[InputExample]:
    """
    Loads a data file into a list of `InputFeatures`
    """
    if enable_windowing and stride is None:
        raise ValueError(
            'Windowing mechanism is activated, but no "stride" or '
            'was provided, please provide it or disable'
            'the mechanism altogether with `enable_windowing=False`'
        )

    windowed_examples = []
    label_map = {label: i for i, label in enumerate(label_list)}
    text_tokenizer = TextTokenizer()
    # three special tokens will be added, remove them from the count
    windowing_max_length = max_length - 6
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="windowing examples"):
        if enable_windowing:
            trigger_windowing, max_ending_length = should_window(
                example=example,
                tokenizer=tokenizer,
                max_length=windowing_max_length,
                no_answer_text=no_answer_text,
            )
        else:
            trigger_windowing, max_ending_length = False, None

        if trigger_windowing:
            windowed_examples.extend(windowed_tokenization(
                example=example,
                label_map=label_map,
                max_window_length=windowing_max_length - max_ending_length,
                max_length=max_length,
                stride=stride,
                no_answer_text=no_answer_text,
                tokenizer=tokenizer,
                text_tokenizer=text_tokenizer,
                window_fn=window_fn
            ))
        else:
            # Append 00 window idx to conform to the rest of windowed features
            example_id = example.example_id
            if enable_windowing:
                example_id = int(str(example_id) + '00')

            windowed_examples.append(InputExample(
                example_id=example_id,
                question=example.question,
                contexts=example.contexts,
                endings=example.endings,
                label=example.label,
            ))

    return windowed_examples


def process_examples(
    data_dir: str,
    tokenizer: PreTrainedTokenizer,
    task: str,
    max_seq_length: Optional[int] = None,
    mode: Split = Split.train,
    enable_windowing: bool = False,
    stride: int = None,
    no_answer_text: str = None,
    window_fn: Callable = None,
) -> List[InputExample]:
    processor = processors[task]()

    label_list = processor.get_labels()
    if mode == Split.dev:
        examples = processor.get_dev_examples(data_dir)
    elif mode == Split.test:
        examples = processor.get_test_examples(data_dir)
    else:
        examples = processor.get_train_examples(data_dir)

    return window_examples(
        examples,
        label_list,
        max_seq_length,
        tokenizer,
        enable_windowing=enable_windowing,
        stride=stride,
        no_answer_text=no_answer_text,
        window_fn=window_fn,
    )


def serialize_examples(examples: List[InputExample]):
    return [ex.__dict__ for ex in examples]


def save_examples(output_file: str, examples: List[InputExample]):
    str_examples = json.dumps(
        serialize_examples(examples),
        ensure_ascii=False
    ) + '\n'
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as fout:
        fout.write(str_examples)


def main():
    parser = HfArgumentParser((
        ModelArguments, DataTrainingArguments,
        TrainingArguments, WindowArguments
    ))
    if len(sys.argv) > 1 and sys.argv[1].endswith('.json'):
        model_args, data_args, training_args, window_args = (
            parser.parse_json_file(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, window_args = (
            parser.parse_args_into_dataclasses()
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    splits = []
    file_names = []
    if training_args.do_train:
        splits.append(Split.train)
        file_names.append('train')
    if training_args.do_eval:
        splits.append(Split.dev)
        file_names.append('dev')
    if training_args.do_predict:
        splits.append(Split.test)
        file_names.append('test')

    for split, f_name in zip(splits, file_names):
        examples = process_examples(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            task=data_args.task_name,
            max_seq_length=data_args.max_seq_length,
            mode=split,
            enable_windowing=window_args.enable_windowing,
            stride=window_args.stride,
            no_answer_text=window_args.no_answer_text,
        )
        print(f'Processed {len(examples)} for split {split}')
        file_path = os.path.join(training_args.output_dir, f_name) + '.json'
        save_examples(file_path, examples)
