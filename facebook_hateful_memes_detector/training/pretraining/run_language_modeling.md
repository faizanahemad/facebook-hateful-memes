# Running MLM over Huggingface models

## Steps
**Step 1** Download the [python file for MLM Training](https://raw.githubusercontent.com/huggingface/transformers/v2.11.0/examples/language-modeling/run_language_modeling.py)

```bash
wget https://raw.githubusercontent.com/huggingface/transformers/v2.11.0/examples/language-modeling/run_language_modeling.py
```

**Step 2** Create Text file with one example per line. Call it `text.csv`

**Step 3** Run MLM command

```bash
python run_language_modeling.py --output_dir='./model' --model_type=distilroberta-base --model_name_or_path=distilroberta-base --tokenizer_name=distilroberta-base --do_train --train_data_file=text.csv --mlm --learning_rate 1e-4 --num_train_epochs 5 --block_size=128 --line_by_line --overwrite_output_dir --per_device_train_batch_size=16 --gradient_accumulation_steps=1
```

## Finding Right Block Size (Max Tokens)

Once you have `text.csv`, In a jupyter notebook or ipython do
```python
import pandas as pd
import numpy as np
from transformers import AutoModelWithLMHead, AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
texts = pd.read_csv("text.csv", header=None)[0].values

m = lambda x: tokenizer.encode_plus(x, add_special_tokens=True, pad_to_max_length=False, truncation=False)
tlens = [len(d['input_ids']) for d in map(m, texts)]
np.percentile(tlens, [97, 99, 99.5, 99.9, 100])
```

## References
- [Blog How to train a new language model from scratch using Transformers and Tokenizers](https://huggingface.co/blog/how-to-train)
- [Raw Python file: run_language_modeling.py](https://raw.githubusercontent.com/huggingface/transformers/v2.11.0/examples/language-modeling/run_language_modeling.py)
- [Possible Training Args Ref](https://github.com/huggingface/transformers/blob/v2.11.0/src/transformers/training_args.py)
- [Reference Page](https://github.com/huggingface/transformers/tree/v2.11.0/examples/language-modeling)
- [Trainer Class](https://github.com/huggingface/transformers/blob/b42586ea560a20dcadb78472a6b4596f579e9043/src/transformers/trainer.py#L153)

