# mc-transformers
Code to run experiments over Multiple Choice QA with huggingface/transformers.
A big part of the code comes from [huggingface/transformers](https://huggingface.co/transformers/), so its license may apply.

## Code
* `utils_multiple_choice.py`: Contains processors specific to each MC QA collection (RACE, SWAG, EntranceExams...)
* `run_multple_choice.py`: Code to train/eval/test models over any collection with transfomers

## Why
As I experiment with more MC collection and training modes (i.e.: tpu), support for more collections or more models is required. Instead of forking the whole transformers library I do it here.
