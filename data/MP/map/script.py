from __future__ import annotations

import gzip
import json
import sys
from collections import defaultdict

import numpy as np

sample_map = defaultdict(int)

tasks = np.load(sys.argv[1])
sample_map["GGA"] = list(tasks)

with gzip.open("map_sample.json.gz", "wt") as f:
    json.dump(sample_map, f)
