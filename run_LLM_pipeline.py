from LLM.helper import *

sub_tasks = ['negation', 'num_to_verbal',
             'active_to_passive', 'singular_to_plural', 'rhymes',
             'second_word_letter', 'sentence_similarity', 'sentiment', 'orthography_starts_with',
             'sum', 'synonyms', 'translation_en-de', 'translation_en-es',
             'translation_en-fr', 'word_in_context']

# total_trials = 1
# bounds = torch.stack([torch.ones(2) * 0.01, torch.ones(2) * 1.0])
# down_sample_size = 0.1
# BO_iteration = 10
# to_use_specific_model = False
# sample_k = 1

# for task in sub_tasks:
#     print("task: ", task)
#     loss_space_bo_all_trials = []
#     for trial in range(1):
#         accuracy = llm_prompt_task(task, bounds = bounds,
#                             down_sample_size = down_sample_size, BO_iteration=BO_iteration, to_use_specific_model = to_use_specific_model,
#                             sample_size=sample_k)
#         loss_space_bo_all_trials.append(accuracy)
#     file_name = "result/llm/" + str(task) + ".csv"
#     np.savetxt(file_name, loss_space_bo_all_trials)

from LLM.helper import *

sub_tasks = ['antonyms', 'diff', 'first_word_letter',
             'informal_to_formal', 'larger_animal', 'letters_list', 'taxonomy_animal', 'negation', 'num_to_verbal',
             'active_to_passive', 'singular_to_plural', 'rhymes',
             'second_word_letter', 'sentence_similarity', 'sentiment', 'orthography_starts_with',
             'sum', 'synonyms', 'translation_en-de', 'translation_en-es',
             'translation_en-fr', 'word_in_context']

total_trials = 1
bounds = torch.stack([torch.ones(2) * 0.001, torch.ones(2) * 1.0])
down_sample_size = 0.01
BO_iteration = 3
to_use_specific_model = False
sample_k = 3
epochs = 2
model = 'gpt-3.5-turbo'
model = 'text-davinci-002'
task = "all_task_predict_at_once_epochs_5_k_3_down_sample_size_0.05_itr_10_model_davinci"

units = sample_k * epochs * down_sample_size / 0.25
print("predicted total time taken (hours): ", str(BO_iteration * units * 2400 / 3600))
loss_space_bo_all_trials = []
for trial in range(1):
    accuracy = llm_prompt_task_modified(task, bounds = bounds,
                        down_sample_size = down_sample_size, BO_iteration=BO_iteration, to_use_specific_model = to_use_specific_model,
                        sample_size=sample_k, epochs=epochs, gpt_model=model)
    loss_space_bo_all_trials.append(accuracy)
file_name = "result/llm/" + str(task) + ".csv"
np.savetxt(file_name, loss_space_bo_all_trials)