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
import json
import logging


from dataclasses import dataclass, field
from typing import Dict, Optional

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
from transformers.trainer_utils import PredictionOutput
from mc_transformers.utils_mc import MultipleChoiceDataset, Split, processors


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


def save_metrics(metrics, dir_args, prefix="eval"):
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


def save_predictions(results, dir_args, prefix="eval"):
    model_predictions = results.predictions
    # cast to avoid json serialization issues
    example_ids = [int(id) for id in results.example_ids]
    label_ids = [int(lab) for lab in results.label_ids]

    output_nbest_file = os.path.join(
        dir_args.results_dir,
        f"{prefix}_nbest_predictions.json"
    )
    output_predictions_file = os.path.join(
        dir_args.results_dir,
        f"{prefix}_predictions.json"
    )

    predictions = softmax(model_predictions, axis=1)
    predictions_dict = {}
    for ex_id, true_label, preds in zip(example_ids, label_ids, predictions):
        pred_dict = {
            "probs": preds.tolist(),
            "pred_label": chr(ord('A') + np.argmax(preds)),
            "label": chr(ord('A') + true_label),
        }
        predictions_dict[ex_id] = pred_dict

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(predictions_dict) + '\n')

    predictions = np.argmax(predictions, axis=1)
    predicted_labels = [chr(ord('A') + id) for id in predictions]
    predictions_list = dict(zip(example_ids, predicted_labels))
    with open(output_predictions_file, "w") as writer:
        writer.write(json.dumps(predictions_list) + '\n')


def save_results(results, dir_args, prefix="eval"):
    # only predict method returns prediction outputs,
    # evaluate and train only return the metrics
    if isinstance(results, PredictionOutput):
        save_metrics(results.metrics, dir_args, prefix)
        save_predictions(results, dir_args, prefix)
    else:
        save_metrics(results, dir_args, prefix)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((
        ModelArguments, DataTrainingArguments,
        DirArguments, TrainingArguments
    ))
    model_args, data_args, dir_args, training_args = (
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

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = AutoModelForMultipleChoice.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    train_dataset = (
        MultipleChoiceDataset(
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            task=data_args.task_name,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.train,
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
        )
        if training_args.do_predict
        else None
    )

    def compute_metrics(p: EvalPrediction) -> Dict:
        preds = np.argmax(p.predictions, axis=1)
        return {"acc": simple_accuracy(preds, p.label_ids)}

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Training
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

    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        result = trainer.predict(eval_dataset)
        if trainer.is_world_master():
            save_results(result, dir_args, prefix="eval")
            results['eval'] = result

    if training_args.do_predict:
        logger.info("*** Test ***")
        result = trainer.predict(test_dataset)
        if trainer.is_world_master():
            save_results(result, dir_args, prefix="test")
            results['test'] = result

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
