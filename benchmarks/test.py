import argparse
import asyncio
import csv
import json
import numpy as np
import os
import sys
import time
from tqdm import tqdm
from typing import List, Tuple

import aiohttp

from exp_suite import BenchmarkConfig, get_all_suites, to_dict, BASE_MODEL, LORA_DIR
from trace import generate_requests, get_real_requests
sys.path.append("../bench_lora")
from slora.utils.metric import reward, attainment_func
from exp_suite import breakdown_suite

suite= "a10g"
exps = [{suite: breakdown_suite[suite]}]
# print(exps)
# for exp in exps:
#     print(exp)
#     for workload in exp:
#         print(workload)
#         (num_adapters, alpha, req_rate, cv, duration,input_range, output_range) = exps[workload]
#         print(f"Running {workload} with {num_adapters} adapters, alpha={alpha}, req_rate={req_rate}, cv={cv}, duration={duration}, input_range={input_range}, output_range={output_range}")
tic = 0
cv=1
req_rate=2
tot_req = 120
shape = 1 / (cv * cv)
scale = cv * cv / req_rate
    # intervals = np.random.exponential(1.0 / req_rate, tot_req)
intervals = np.random.gamma(shape, scale, tot_req)
print(f"Intervals: {intervals}")