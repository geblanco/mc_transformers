import tqdm
import random
import logging

from typing import List, Callable, Tuple
from transformers import PreTrainedTokenizer
from mc_transformers.data_classes import InputFeatures, InputExample


logger = logging.getLogger(__name__)


try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import SnowballStemmer, WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    # trigger download if necessary
    stopwords.words('english')
    is_nltk_available = True
except ImportError:
    is_nltk_available = False
except LookupError:
    # download extra data
    nltk.download(['stopwords', 'punkt', 'wordnet'])
    is_nltk_available = True


class TextTokenizer(object):
    stop_words = stopwords.words("english") if is_nltk_available else None
    lemmatizer = WordNetLemmatizer() if is_nltk_available else None
    stemmer = SnowballStemmer("english") if is_nltk_available else None

    def __call__(self, text):
        if is_nltk_available:
            text_tokens = word_tokenize(text.lower().strip())
            return [
                self.stemmer.stem(self.lemmatizer.lemmatize(w))
                for w in text_tokens if w not in self.stop_words
            ]
        else:
            logger.warning(
                "You are using a `TextTokenizer` for windowing but NLTK module"
                " is not installed, you can install with "
                "`pip install mc_transformers[windowing]`"
            )
            return text.lower().strip().split(' ')


def argmax(arr: List) -> int:
    return max(enumerate(arr), key=lambda elem: len(elem[1]))[0]


def match_text_by_tokenizer(
    text_a: str,
    text_b: str,
    text_tokenizer: TextTokenizer,
) -> bool:
    text_a_tokens = text_tokenizer(text_a)
    text_b_tokens = text_tokenizer(text_b)
    threshold = 0.5
    nof_correct = sum([
        1 if token in text_a_tokens else 0
        for token in text_b_tokens
    ])
    if nof_correct > 0:
        nof_correct /= len(text_b_tokens)
    return (nof_correct > threshold)


def should_correct_label(
    context: str,
    endings: str,
    no_answer_text: str,
    text_tokenizer: TextTokenizer,
) -> bool:
    # find if the answer is contained in the context,
    # which possibly indicates that the current window
    # context is enough to guess the answer
    # ToDo := This should be revised, if inference is to be made,
    # words do not necessarily appear in the options
    ctx = context.lower()
    endings = [end.lower() for end in endings]
    correct = False
    for ans in endings:
        correct = (
            (no_answer_text == ans or ctx.find(ans) != -1) or
            match_text_by_tokenizer(
                text_a=ctx, text_b=ans, text_tokenizer=text_tokenizer
            )
        )

        if correct:
            break
    return correct


def should_window(
    example: InputExample,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    no_answer_text: str,
) -> bool:
    context = example.contexts[0]
    context_tokens = tokenizer.encode(context, add_special_tokens=False)
    concats = concat_question_and_endings(
        example.question, example.endings + (
            [no_answer_text]
            if no_answer_text is not None and no_answer_text != ""
            else []
        )
    )
    longest_concat = concats[argmax(concats)]
    # get the longest span to test max length
    text_b_tokens = tokenizer.encode(longest_concat, add_special_tokens=False)
    enable_windowing = len(context_tokens) + len(text_b_tokens) > max_length
    return enable_windowing, len(text_b_tokens)


def correct_label(
    context: str,
    endings: List[str],
    no_answer_text: str,
    text_tokenizer: TextTokenizer,
) -> Tuple[int, List[str]]:
    # find `no answer text` in endings, if not found, replace a random ending
    # with the provided one and mark it as the correct answer
    found = False
    label_index = -1
    endings = [end.lower() for end in endings]
    for idx, ans in enumerate(endings):
        found = (
            (no_answer_text == ans or ans.find(no_answer_text) != -1) or
            match_text_by_tokenizer(
                text_a=no_answer_text, text_b=ans, text_tokenizer=text_tokenizer,
            )
        )
        if found:
            label_index = idx
            break

    if not found:
        label_index = random.choice(range(len(endings)))
        endings[label_index] = no_answer_text

    return label_index, endings


def create_windows(
    context: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    stride: int
) -> List[int]:
    context_tokens = tokenizer.encode(context, add_special_tokens=False)
    windows = []
    win_start = 0
    win_end = max_length
    total_size = len(context_tokens)
    nof_windows = round(total_size / (max_length - stride))
    for _ in range(nof_windows):
        windows.append(context_tokens[win_start:win_end])
        win_start = win_end - stride
        win_end = min(win_start + max_length, total_size)

    return [
        tokenizer.decode(
            tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        for tokens in windows
    ]


def concat_question_and_endings(question: str, endings: List[str]) -> List[str]:
    concats = []
    for end in endings:
        if question.find("_") != -1:
            # this is for cloze question
            text_b = question.replace("_", end)
        else:
            text_b = question + " " + end
        concats.append(text_b)

    return concats


def create_input_features(
    contexts: List[str],
    endings: List[str],
    example_id: int,
    label: int,
    max_length: int,
    tokenizer: PreTrainedTokenizer,
) -> InputFeatures:
    choices_inputs = []
    for text_a, text_b in zip(contexts, endings):
        inputs = tokenizer(
            text_a,
            text_b,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation='only_first',
            return_overflowing_tokens=True,
        )
        if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
            logger.info(
                "Attention! you are cropping tokens (swag task is ok). "
                "If you are training ARC and RACE and you are poping question + options,"
                "you need to try to use a bigger max seq length! "
                f"Cropped {inputs['num_truncated_tokens']} tokens"
            )

        choices_inputs.append(inputs)

    input_ids = [x["input_ids"] for x in choices_inputs]
    attention_mask = (
        [x["attention_mask"] for x in choices_inputs] if "attention_mask" in choices_inputs[0] else None
    )
    token_type_ids = (
        [x["token_type_ids"] for x in choices_inputs] if "token_type_ids" in choices_inputs[0] else None
    )
    return InputFeatures(
        example_id=example_id,
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        label=label,
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
) -> List[InputFeatures]:
    # ToDo := Different amount of windows will trigger an error because of
    # different size in input features? sequences should be grouped by
    # size and chopped, padded accordingly

    # ToDo := no_answer_text is not used by now, no label corrected
    able_to_correct_label = no_answer_text is not None and no_answer_text != ""
    window_fn = window_fn if window_fn is not None else create_windows
    window_texts = window_fn(
        example.contexts[0], tokenizer, max_window_length, stride
    )
    # logger.info(
    #     f"Created {len(window_texts)} windows for example "
    #     f"`{example.example_id}` with max window size {max_window_length}"
    # )
    features = []
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

        texts_a = [win_text] * len(endings)
        texts_b = concat_question_and_endings(example.question, endings)
        # maximum 100 windows
        example_id = int(str(example.example_id) + str_win_idx)
        features.append(
            create_input_features(
                contexts=texts_a,
                endings=texts_b,
                example_id=example_id,
                label=label,
                max_length=max_length,
                tokenizer=tokenizer,
            )
        )

    return features


def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
    enable_windowing: bool = False,
    stride: int = None,
    no_answer_text: str = None,
    window_fn: Callable = None,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """
    if enable_windowing and stride is None:
        raise ValueError(
            'Windowing mechanism is activated, but no "stride" or '
            'was provided, please provide it or disable'
            'the mechanism altogether with `enable_windowing=False`'
        )

    features = []
    label_map = {label: i for i, label in enumerate(label_list)}
    text_tokenizer = TextTokenizer()
    # three special tokens will be added, remove them from the count
    windowing_max_length = max_length - 6
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

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
            features.extend(windowed_tokenization(
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

            concats = concat_question_and_endings(
                example.question, example.endings
            )
            features.append(create_input_features(
                contexts=example.contexts,
                endings=concats,
                example_id=example_id,
                label=label_map[example.label],
                max_length=max_length,
                tokenizer=tokenizer,
            ))

    for f in features[:2]:
        logger.info("*** Example ***")
        logger.info("feature: %s" % f)

    return features
