from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate
from transformers import DataCollatorWithPadding
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import numpy as np

class SpecialTrainer(Trainer):
    target_loss = 0.0
    def compute_loss(self, model, inputs, return_outputs=False):
        # implement custom logic here
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return ((loss - self.target_loss)**2, outputs) if return_outputs else (loss - self.target_loss)**2

f1_score = evaluate.load("f1")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return f1_score.compute(predictions=predictions, references=labels)


def train_all(data, model_output_name, epochs=3, target_loss=0.0):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_data = data.map(preprocess_function, batched=True)

    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}   
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
    )

    training_args = TrainingArguments(
    output_dir=model_output_name,
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=epochs,
    weight_decay=0.01,
    logging_steps=100,
    gradient_accumulation_steps=1,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    )
    
    trainer = SpecialTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["train"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    )

    trainer.target_loss = target_loss

    trainer.train()
    trainer.save_model(model_output_name)
    return model
    
def modelA_pick_examples(q, a, model):
    def combine(q, a):
        examples = []
        for x,y in zip(q,a):
            examples.append(str(x + ". " + y))
        return examples

    data = combine(q, a)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    output_q = []
    output_a = []

    for idx, example in enumerate(data):
        inputs = tokenizer(example, return_tensors="pt")

        with torch.no_grad():
            logits = model.to("cpu")(**inputs).logits
            predicted_class_id = logits.argmax().item()
        if (predicted_class_id):
            output_q.append(q[idx])
            output_a.append(a[idx])
    return output_q, output_a

def modelB_pick_prompts(q_filtered, a_filtered, prompts_generated, model):
    def combine(q, a):
        examples = []
        for x,y in zip(q,a):
            examples.append(str(x + ". " + y))
        return examples

    examples = combine(q_filtered, a_filtered)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    prompt_scores = []
    for idx, prompt in enumerate(prompts_generated):
        prompt_score_on_examples = 0
        for idx, example in enumerate(examples):
            #print("input: ", example + ". " + prompt)
            inputs = tokenizer(example + ". " + prompt, return_tensors="pt")

            with torch.no_grad():
                logits = model.to("cpu")(**inputs).logits
                predicted_class_id = logits.argmax().item()
                #print("prediction: ", predicted_class_id)
            if (predicted_class_id):
                prompt_score_on_examples +=1 # may need to just change to logit score? idk
        prompt_scores.append(prompt_score_on_examples)
    #print(prompt_scores)
    highest_prompt_score = max(prompt_scores)
    idx_of_best_prompts = [i for i, j in enumerate(prompt_scores) if j == highest_prompt_score]
    best_idx = random.choice(idx_of_best_prompts)
    best_prompt = prompts_generated[best_idx]
    return best_prompt

def train_model(task, classification_type, target_loss, epochs=10, to_down_sample=False, down_sample_size=0.2):
    ds = load_from_disk("instruction_induction/" + classification_type + "_data/" + task + ".hf")
    if to_down_sample:
        ds_split = ds["train"].train_test_split(test_size=1-down_sample_size, shuffle=True)
        ds["train"] = ds_split["train"]
    
    model = train_all(ds, classification_type + "_" + task, epochs=epochs, target_loss=target_loss)
    return model

import json
import random
sub_tasks = ['antonyms', 'diff', 'first_word_letter',
             'informal_to_formal', 'larger_animal', 'letters_list', 'taxonomy_animal', 'negation', 'num_to_verbal',
             'active_to_passive', 'singular_to_plural', 'rhymes',
             'second_word_letter', 'sentence_similarity', 'sentiment', 'orthography_starts_with',
             'sum', 'synonyms', 'translation_en-de', 'translation_en-es',
             'translation_en-fr', 'word_in_context']
from datasets import load_dataset, DatasetDict, Dataset, load_from_disk, concatenate_datasets
def get_data_prompt_picker(task_name):
    f = open('instruction_induction/raw/induce/' + task_name + '.json')
    correct_promp = open('instruction_induction/annotations/' + task_name + '.json')
    examples_data = json.load(f)
    prompt_data = json.load(correct_promp)
    text = []
    label = []

    # label 1 (correct prompt)
    for example_idx in examples_data["examples"]:

        if "input" not in  examples_data["examples"][str(example_idx)] or "output" not in  examples_data["examples"][str(example_idx)]:
            print("task: ", task_name, " has weird input output example field, pls check!")
            return {"text":text, "label":label}
    
        q = examples_data["examples"][str(example_idx)]["input"]
        a = examples_data["examples"][str(example_idx)]["output"]
        prompts = prompt_data["annotations"]
        for prompt in prompts:
            instruction = q + " " + a + " " + prompt
            text.append(instruction)
            label.append(1)

    # label 0 (wrong prompt)
    for example_idx in examples_data["examples"]:
        
        wrong_task = random.choice(sub_tasks)
        wrong_f = open('instruction_induction/annotations/' + wrong_task + '.json')
        wrong_prompts = json.load(wrong_f)
        q = examples_data["examples"][str(example_idx)]["input"]
        a = examples_data["examples"][str(example_idx)]["output"]
        wrong_prompts = wrong_prompts["annotations"]
        for wrong_prompt in wrong_prompts:
            instruction = q + " " + a + " " + wrong_prompt
            text.append(instruction)
            label.append(0)

    dataset = {"text":text, "label":label}
    return dataset

def get_data_example_picker(task_name):
    f = open('instruction_induction/raw/induce/' + task_name + '.json')
    examples_data = json.load(f)
    text = []
    label = []

    # label 1 (correct example)
    for example_idx in examples_data["examples"]:

        if "input" not in  examples_data["examples"][str(example_idx)] or "output" not in  examples_data["examples"][str(example_idx)]:
            print("task: ", task_name, " has weird input output example field, pls check!")
            return {"text":text, "label":label}
    
        q = examples_data["examples"][str(example_idx)]["input"]
        a = examples_data["examples"][str(example_idx)]["output"]
        example = q + ". " + a
        text.append(example)
        label.append(1)

        wrong_task = random.choice(sub_tasks)
        wrong_f = open('instruction_induction/raw/induce/' + wrong_task + '.json')
        wrong_example = json.load(wrong_f)
        s = len(wrong_example["examples"])
        picked_wrong_idx = random.randint(1, s)
        q_wrong = wrong_example["examples"][str(picked_wrong_idx)]["input"]
        a_wrong = wrong_example["examples"][str(picked_wrong_idx)]["output"]
        wrong_example_one = q + ". " + a_wrong
        text.append(wrong_example_one)
        label.append(0)
        wrong_example_two = q_wrong + ". " + a
        text.append(wrong_example_two)
        label.append(0)

    dataset = {"text":text, "label":label}
    return dataset

# generate examples and q,a,prompt classification data
def generate_all_data():
    tasks = sub_tasks
    all_data = []
    for task in tasks:
        print("processing task: ", task)
        d = get_data_prompt_picker(task)
        ds = Dataset.from_dict(d)
        all_data.append(ds)
        ds = ds.train_test_split(test_size=0.3, shuffle=True)
        ds.save_to_disk("instruction_induction/prompt_picker_data/" + task + ".hf")
    ds = concatenate_datasets(all_data)
    ds = ds.train_test_split(test_size=0.3, shuffle=True)
    ds.save_to_disk("instruction_induction/prompt_picker_data/" + "all_task_combined" + ".hf")
    
    all_data = []
    for task in tasks:
        print("processing task: ", task)
        d = get_data_example_picker(task)
        ds = Dataset.from_dict(d)
        all_data.append(ds)
        ds = ds.train_test_split(test_size=0.3, shuffle=True)
        ds.save_to_disk("instruction_induction/example_picker_data/" + task + ".hf")
    ds = concatenate_datasets(all_data)
    ds = ds.train_test_split(test_size=0.3, shuffle=True)
    ds.save_to_disk("instruction_induction/example_picker_data/" + "all_task_combined" + ".hf")

def get_eval_data(task_name):
    with open("instruction_induction/evaluation_data/input_" + task_name, "r") as fp:
        q = json.load(fp)
    with open("instruction_induction/evaluation_data/output_" + task_name, "r") as fp:
        a = json.load(fp)
    return q,a

import openai
from automatic_prompt_engineer import generate, evaluate, config, template, data, llm, ape
from botorch.acquisition import UpperConfidenceBound
import torch
from botorch.models import SingleTaskGP, HeteroskedasticSingleTaskGP, KroneckerMultiTaskGP, MultiTaskGP, FixedNoiseMultiTaskGP, FixedNoiseGP, HeteroskedasticSingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood, MarginalLogLikelihood
from botorch.optim import optimize_acqf
import time

def llm_prompt_task(task_name, bounds = torch.stack([torch.ones(2) * 0.01, torch.ones(2) * 0.1]),
                    down_sample_size = 0.05, BO_iteration=10, to_use_specific_model = False, sample_size = 1, epochs=10):
    openai.api_key = "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"
    task = task_name
    q_eval,a_eval = get_eval_data(task)
    target_loss = [0.5, 0.5]
    best_accuracy = -1
    all_best_accuracy = []
    X = []
    y = []
    for i in range(BO_iteration):
        start = time.time()

        model_to_train_on = "all_task_combined"
        if to_use_specific_model:
            model_to_train_on = task
        
        accuracy = 0.0
        for k in range(sample_size):
            model_example_picker = train_model(model_to_train_on, "example_picker", target_loss[0], epochs=epochs, to_down_sample=True, down_sample_size=down_sample_size)
            model_prompt_picker = train_model(model_to_train_on, "prompt_picker", target_loss[1], epochs=epochs, to_down_sample=True, down_sample_size=down_sample_size)

            # deploy model A : classify each example and get a score, get top k example
            #q_filtered, a_filtered = modelA_pick_examples(q_eval,a_eval, model = AutoModelForSequenceClassification.from_pretrained("./example_picker_" + task + "/", local_files_only=True))
            print("number of queries before filtering: ", len(q_eval))
            q_filtered, a_filtered = modelA_pick_examples(q_eval,a_eval, model = model_example_picker)
            print("number of queries after filtering: ", len(q_filtered))
            if len(q_filtered) == 0:
                q_filtered = q_eval
                a_filtered = a_eval
            
            # use ape.simple_ape(...) to get a list of prompts (maybe without filtering, so we have bad prompts)
            eval_template = \
            """Instruction: [PROMPT]
            Input: [INPUT]
            Output: [OUTPUT]"""
            prompts_generated = ape.ape_to_produce_prompt_autoAI(dataset=(q_filtered, a_filtered), eval_template=eval_template)
            print(prompts_generated)

            # filter the prompts
            prompt_filtered = modelB_pick_prompts(q_filtered, a_filtered, prompts_generated, model = model_prompt_picker)
            print(prompt_filtered)

            result = ape.evaluate_prompts_autoAI([prompt_filtered], eval_data=(q_eval,a_eval), demo_data=(q_filtered,a_filtered))
            accuracy = max(accuracy, result.sorted()[1][0])


        if accuracy > best_accuracy:
            best_accuracy = accuracy
        print("X: ", target_loss)
        print("current accuracy: ", accuracy)
        print("current best: ", best_accuracy)
        all_best_accuracy.append(best_accuracy)
        
        X.append(target_loss)
        y.append([accuracy])
        gp = SingleTaskGP(torch.DoubleTensor(X), torch.DoubleTensor(y).double())
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        UCB = UpperConfidenceBound(gp, beta=0.5)
        candidate, acq_value = optimize_acqf(
        UCB, bounds=bounds, q=1, num_restarts=1, raw_samples=20)
        target_loss = candidate[0]
        end = time.time()
        print("time taken for one BO iteration: ", end-start)
    print("finished. X, y: ")
    print(X)
    print(y)
    return all_best_accuracy

def llm_prompt_task_modified(task_name, bounds = torch.stack([torch.ones(2) * 0.01, torch.ones(2) * 0.1]),
                    down_sample_size = 0.05, BO_iteration=10, to_use_specific_model = False, sample_size = 1, epochs = 10, gpt_model='text-davinci-002'):
    sub_tasks = ['negation', 'num_to_verbal',
             'active_to_passive', 'singular_to_plural', 'rhymes',
             'second_word_letter', 'sentence_similarity', 'sentiment', 'orthography_starts_with',
             'sum', 'synonyms', 'translation_en-de', 'translation_en-es',
             'translation_en-fr', 'word_in_context']
    openai.api_key = "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"
    target_loss = [0.5, 0.5]
    best_accuracy = -1
    all_best_accuracy = []
    X = []
    y = []
    for i in range(BO_iteration):
        start = time.time()

        model_to_train_on = "all_task_combined"
        # if to_use_specific_model:
        #     model_to_train_on = task
        
        overall_accuracy = 0.0
        for k in range(sample_size):
            model_example_picker = train_model(model_to_train_on, "example_picker", target_loss[0], epochs=epochs, to_down_sample=True, down_sample_size=down_sample_size)
            model_prompt_picker = train_model(model_to_train_on, "prompt_picker", target_loss[1], epochs=epochs, to_down_sample=True, down_sample_size=down_sample_size)

            accuracy = {}
            for task in sub_tasks:
                q_eval,a_eval = get_eval_data(task)
        
                # deploy model A : classify each example and get a score, get top k example
                #q_filtered, a_filtered = modelA_pick_examples(q_eval,a_eval, model = AutoModelForSequenceClassification.from_pretrained("./example_picker_" + task + "/", local_files_only=True))
                print("number of queries before filtering: ", len(q_eval))
                q_filtered, a_filtered = modelA_pick_examples(q_eval,a_eval, model = model_example_picker)
                print("number of queries after filtering: ", len(q_filtered))
                if len(q_filtered) == 0:
                    q_filtered = q_eval
                    a_filtered = a_eval
                
                # use ape.simple_ape(...) to get a list of prompts (maybe without filtering, so we have bad prompts)
                eval_template = \
                """Instruction: [PROMPT]
                Input: [INPUT]
                Output: [OUTPUT]"""
                prompts_generated = ape.ape_to_produce_prompt_autoAI(dataset=(q_filtered, a_filtered), eval_template=eval_template)
                print(prompts_generated)

                # filter the prompts
                prompt_filtered = modelB_pick_prompts(q_filtered, a_filtered, prompts_generated, model = model_prompt_picker)
                print(prompt_filtered)

                result = ape.evaluate_prompts_autoAI([prompt_filtered], eval_data=(q_eval,a_eval), demo_data=(q_filtered,a_filtered), eval_model=gpt_model,
               prompt_gen_model=gpt_model)
                accuracy[task] = result.sorted()[1][0]
            print("individual task accuracy: ", accuracy)
            overall_accuracy = max(overall_accuracy, sum(accuracy.values())/len(accuracy.values()))


        if overall_accuracy > best_accuracy:
            best_accuracy = overall_accuracy
        print("X: ", target_loss)
        print("current accuracy: ", overall_accuracy)
        print("current best: ", best_accuracy)
        all_best_accuracy.append(best_accuracy)
        
        X.append(target_loss)
        y.append([overall_accuracy])
        gp = SingleTaskGP(torch.DoubleTensor(X), torch.DoubleTensor(y).double())
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        UCB = UpperConfidenceBound(gp, beta=1)
        candidate, acq_value = optimize_acqf(
        UCB, bounds=bounds, q=1, num_restarts=1, raw_samples=20)
        target_loss = candidate[0]
        end = time.time()
        print("time taken for one BO iteration: ", end-start)
    print("finished. X, y: ")
    print(X)
    print(y)
    return all_best_accuracy