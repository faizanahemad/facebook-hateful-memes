import pandas as pd
import numpy as np
import argparse
from collections import Counter
import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, required=True)
parser.add_argument('--files', type=str, nargs="+", required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()
files = [os.path.join(args.dir, f) for f in args.files]
files = [pd.read_csv(f) for f in files]

labels = list(zip(*[f["label"].values for f in files]))

labels = [Counter(l).most_common(1)[0][0] for l in labels]
results = pd.DataFrame({"ID": files[0]["ID"], "label": labels})
results.to_csv(os.path.join(args.dir, args.output), index=False)





