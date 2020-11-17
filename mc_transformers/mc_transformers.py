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
""" Finetuning the library models for multiple choice (Bert, Roberta, XLNet)."""


import os
import sys
import json
import logging

from pathlib import Path
from typing import Dict, Optional
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers import is_tf_available
from mc_transformers.utils_mc import MultipleChoiceDataset, Split, processors
from mc_transformers.data_classes import (
    WindowPrediction,
    PredictionOutputWithIds,
    DataCollatorWithIds,
)


if is_tf_available():
    # Force no unnecessary allocation
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


logger = logging.getLogger(__name__)
os.environ.update(**{"WANDB_DISABLED": "true"})


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def compute_metrics(p: EvalPrediction) -> Dict:
    preds = np.argmax(p.predictions, axis=1)
    return {"acc": simple_accuracy(preds, p.label_ids)}


def softmax(preds, axis=None):
    # Taken from: https://nolanbconaway.github.io/blog/2017/softmax-numpy.html
    if axis is None:
        raise ValueError("Softmax function needs an axis to work!")
    # make preds at least 2d
    y = np.atleast_2d(preds)
    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)
    y = np.exp(y)
    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)
    p = y / ax_sum
    # flatten if preds was 1D
    if len(preds.shape) == 1:
        p = p.flatten()

    return p


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
class DirArguments:
    """
    Arguments pertaining to output directories for metrics, results and predictions
    """
    metrics_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Output directory for metrics (loss/accuracy)"
        }
    )
    results_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Output directory for predictions"
        }
    )
    save_logits: bool = field(
        default=False,
        metadata={
            "help": "Whether to store logits along with predictions"
        }
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
    windows_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Output directory for windowed predictions"
        }
    )


def vote_windowed_predictions(windowed_predictions):
    predictions = []
    # strategies:
    #   max along each column
    #   voting?
    for win_pred in windowed_predictions:
        predictions.append(np.argmax(win_pred.predictions, axis=0))

    return np.array(predictions)


def parse_windowed_predictions(args, processor, results, split):
    data_args = args['data_args']
    window_args = args['window_args']
    if split == Split.dev:
        file_name = "dev_windowed_predictions.json"
        examples = processor.get_dev_examples(data_args.data_dir)
    elif split == Split.test:
        file_name = "test_windowed_predictions.json"
        examples = processor.get_test_examples(data_args.data_dir)
    else:
        file_name = "train_windowed_predictions.json"
        examples = processor.get_train_examples(data_args.data_dir)

    predictions = []
    id_to_example_map = {example.example_id: example for example in examples}
    label_map = {label: i for i, label in enumerate(processor.get_labels())}

    example_labels = defaultdict(list)
    example_window_ids = defaultdict(list)
    example_predictions = defaultdict(list)

    zipped = zip(results.example_ids, results.label_ids, results.predictions)
    for feat_id, feat_label, feat_preds in zipped:
        str_feat_id = str(feat_id)
        example_id = int(str_feat_id[:-2])
        win_id = int(str_feat_id[-2:])
        example_labels[example_id].append(int(feat_label))
        example_window_ids[example_id].append(win_id)
        example_predictions[example_id].append(feat_preds)

    for example_id, example in id_to_example_map.items():
        windowed_predictions = WindowPrediction(
            predictions=np.vstack(example_predictions[example_id]),
            window_ids=example_window_ids[example_id],
            labels=example_labels[example_id],
            label=label_map[example.label],
            example=example,
        )
        predictions.append(windowed_predictions)

    if window_args.windows_dir is not None:
        Path(window_args.windows_dir).mkdir(parents=True, exist_ok=True)
        file_path = os.path.join(window_args.windows_dir, file_name)
        window_preds_str = json.dumps([win.todict() for win in predictions])
        with open(file_path, 'w') as fout:
            fout.write(window_preds_str + '\n')

    reduced_predictions = vote_windowed_predictions(predictions)
    example_ids, label_ids = zip(*[
        (win_pred.example.example_id, win_pred.example.label)
        for win_pred in predictions
    ])
    return parse_default_predictions(
        args=args,
        processor=processor,
        example_ids=example_ids,
        label_ids=label_ids,
        predictions=reduced_predictions,
    )


def parse_default_predictions(args, processor, example_ids, label_ids, predictions):
    # ToDo := Test predictions should not have true label
    # cast to avoid json serialization issues
    example_ids = [processor._decode_id(int(ex_id)) for ex_id in example_ids]
    label_ids = [int(lab) for lab in label_ids]
    label_id_map = {i: chr(ord('A') + int(label)) for i, label in enumerate(processor.get_labels())}

    pred_logits = predictions.tolist()
    predictions = softmax(predictions, axis=1)
    predictions_dict = defaultdict(list)

    for (ex_id, q_id), true_label, preds, logits in zip(example_ids, label_ids, predictions, pred_logits):
        pred_dict = {
            "probs": preds.tolist(),
            "pred_label": label_id_map[np.argmax(preds)],
            "label": label_id_map[true_label],
        }

        if args['dir_args'].save_logits:
            pred_dict.update(**{"logits": logits})

        predictions_dict[ex_id].append(pred_dict)

    full_ids = ['-'.join([c_id, qa_id]) for c_id, qa_id in example_ids]
    predictions = np.argmax(predictions, axis=1)
    predicted_labels = [label_id_map[id] for id in predictions]
    predictions_list = dict(zip(full_ids, predicted_labels))

    return predictions_dict, predictions_list


def save_metrics(metrics, args, split):
    dir_args = args['dir_args']
    prefix = "eval"
    if split == Split.test:
        prefix = "test"
    elif split == Split.train:
        prefix = "train"

    metrics_dict = {}
    output_metrics_file = os.path.join(
        dir_args.metrics_dir,
        f"{prefix}_metrics.json"
    )
    for key in ["eval_loss", "eval_acc"]:
        if metrics.get(key) is not None:
            metrics_dict[key] = metrics.get(key)

    if len(metrics_dict.keys()) == 0:
        logger.info("Neither loss or accuracy found on result dict!")
    else:
        with open(output_metrics_file, "w") as writer:
            writer.write(json.dumps(metrics_dict) + '\n')
        for key, value in metrics_dict.items():
            logger.info("  %s = %s", key, value)


def save_predictions(processor, results, args, split):
    dir_args, window_args = args['dir_args'], args['window_args']
    prefix = "eval"
    if split == Split.test:
        prefix = "test"
    elif split == Split.train:
        prefix = "train"

    if window_args.enable_windowing:
        predictions_dict, predictions_list = parse_windowed_predictions(
            args=args,
            processor=processor,
            results=results,
            split=split,
        )
    else:
        predictions_dict, predictions_list = parse_default_predictions(
            args=args,
            processor=processor,
            example_ids=results.example_ids,
            label_ids=results.label_ids,
            predictions=results.predictions,
        )

    output_nbest_file = os.path.join(
        dir_args.results_dir,
        f"{prefix}_nbest_predictions.json"
    )
    output_predictions_file = os.path.join(
        dir_args.results_dir,
        f"{prefix}_predictions.json"
    )

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(predictions_dict) + '\n')

    with open(output_predictions_file, "w") as writer:
        writer.write(json.dumps(predictions_list) + '\n')


def save_results(processor, results, args, split):
    # only predict method returns prediction outputs,
    # evaluate and train only return the metrics
    if isinstance(results, PredictionOutputWithIds):
        save_metrics(results.metrics, args, split)
        save_predictions(processor, results, args, split)
    else:
        save_metrics(results, args, split)


def pair_predictions_with_ids(results, data_collator):
    return PredictionOutputWithIds(
        predictions=results.predictions,
        label_ids=results.label_ids,
        example_ids=data_collator.example_ids,
        metrics=results.metrics,
    )


def setup(argc=None, **kwargs):
    if argc is None:
        argc = sys.argv[1:]
    parser = HfArgumentParser((
        ModelArguments, DataTrainingArguments,
        DirArguments, TrainingArguments, WindowArguments
    ))
    if (
        isinstance(argc, list) and
        len(argc) == 1 and
        argc[0].endswith('.json')
    ):
        model_args, data_args, dir_args, training_args, window_args = (
            parser.parse_json_file(argc[0])
        )
    elif isinstance(argc, dict):
        model_args, data_args, dir_args, training_args, window_args = (
            parser.parse_dict(argc)
        )
    else:
        model_args, data_args, dir_args, training_args, window_args = (
            parser.parse_args_into_dataclasses()
        )

    if (
        os.path.exists(training_args.output_dir)
        and [f for f in os.listdir(training_args.output_dir) if f != '.gitignore']
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    all_args = {
        'model_args': model_args,
        'data_args': data_args,
        'dir_args': dir_args,
        'training_args': training_args,
        'window_args': window_args,
    }
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    try:
        processor = processors[data_args.task_name]()
        label_list = processor.get_labels()
        num_labels = len(label_list)
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    config_kwargs = kwargs.pop('config_kwargs', {})
    tokenizer_kwargs = kwargs.pop('tokenizer_kwargs', {})
    model_kwargs = kwargs.pop('model_kwargs', {})

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        **config_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        **tokenizer_kwargs,
    )
    model = AutoModelForMultipleChoice.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        **model_kwargs,
    )

    return all_args, processor, config, tokenizer, model


def main():
    all_args, processor, config, tokenizer, model = setup()
    model_args, data_args, dir_args, training_args, window_args = (
        all_args.values()
    )

    train_dataset = (
        MultipleChoiceDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            task=data_args.task_name,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.train,
            enable_windowing=window_args.enable_windowing,
            stride=window_args.stride,
            no_answer_text=window_args.no_answer_text,
        )
        if training_args.do_train
        else None
    )
    eval_dataset = (
        MultipleChoiceDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            task=data_args.task_name,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.dev,
            enable_windowing=window_args.enable_windowing,
            stride=window_args.stride,
            no_answer_text=window_args.no_answer_text,
        )
        if training_args.do_eval
        else None
    )

    test_dataset = (
        MultipleChoiceDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            task=data_args.task_name,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.test,
            enable_windowing=window_args.enable_windowing,
            stride=window_args.stride,
            no_answer_text=window_args.no_answer_text,
        )
        if training_args.do_predict
        else None
    )

    data_collator = DataCollatorWithIds()

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator.collate,
    )

    # Training
    results = {}
    if training_args.do_train:
        trainer.train(
            model_path=(
                model_args.model_name_or_path
                if os.path.isdir(model_args.model_name_or_path)
                else None
            )
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

        logger.info("*** Evaluate (train set)***")
        result = trainer.predict(train_dataset)
        if trainer.is_world_master():
            result = pair_predictions_with_ids(result, data_collator)
            save_results(
                processor, result, all_args, split=Split.train
            )
            results['train'] = result
            data_collator.drop_ids()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        result = trainer.predict(eval_dataset)
        if trainer.is_world_master():
            result = pair_predictions_with_ids(result, data_collator)
            save_results(
                processor, result, all_args, split=Split.dev
            )
            results['eval'] = result
            data_collator.drop_ids()

    if training_args.do_predict:
        logger.info("*** Test ***")
        result = trainer.predict(test_dataset)
        if trainer.is_world_master():
            result = pair_predictions_with_ids(result, data_collator)
            save_results(
                processor, result, all_args, split=Split.test
            )
            results['test'] = result
            data_collator.drop_ids()

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
