import os
import random

from LLM.helper import *
from LLM.BO import run_BO_GP_UCB


os.environ["TOKENIZERS_PARALLELISM"] = "false"
sub_tasks = ['negation', 'num_to_verbal',
             'active_to_passive', 'singular_to_plural', 'rhymes',
             'second_word_letter', 'sentence_similarity', 'sentiment', 'orthography_starts_with',
             'sum', 'synonyms', 'translation_en-de', 'translation_en-es',
             'translation_en-fr', 'word_in_context']
openai.api_key = "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"

BO_iterations=1
trials=1
sub_tasks = ["num_to_verbal"]
result = run_BO_GP_UCB(BO_iterations, trials, sub_tasks)
output_dir = "result/llm_vanilla_bo.csv"
result = np.array(result)
np.savetxt(output_dir, result)