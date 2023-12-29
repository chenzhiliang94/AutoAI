import os
import random

from LLM.helper import *
from LLM.BO import *

import torch
from botorch.fit import fit_gpytorch_mll
from SearchAlgorithm.AcquisitionFunction import *
from SearchAlgorithm.turbo import *


os.environ["TOKENIZERS_PARALLELISM"] = "false"
openai.api_key = "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"

sub_tasks = ['negation', 'num_to_verbal',
             'active_to_passive', 'singular_to_plural', 'rhymes',
             'second_word_letter', 'sentence_similarity', 'sentiment', 'orthography_starts_with',
             'sum', 'synonyms', 'translation_en-de', 'translation_en-es',
             'translation_en-fr', 'word_in_context']
sub_tasks = ['negation']


bo_all_trials = llm_turbo(sub_tasks, to_normalize_y=True, total_trials=1, iterations=1)

# need to make sure each trial has same number of iterations
max_len = max([len(x) for x in bo_all_trials])
for x in range(0, len(bo_all_trials)):
    while len(bo_all_trials[x]) < max_len:
        bo_all_trials[x].append(bo_all_trials[x][-1])

output_dir = "result/llm_turbo.csv"
results = np.array(bo_all_trials)
np.savetxt(output_dir, results)