import json
import random
import torch

from transformers import pipeline
from transformers.data.processors.squad import SquadV2Processor
from transformers import AutoTokenizer, AutoModelForQuestionAnswering


tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased',return_token_type_ids = True)
model = AutoModelForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')
qa_model = pipeline('question-answering', model=model, tokenizer=tokenizer, device=0)


def load_data(file_loc, file_name):
    """This function was based on work from
    https://qa.fastforwardlabs.com/no%20answer/null%20threshold/bert/distilbert/exact%20match/f1/robust%20predictions/2020/06/09/Evaluating_BERT_on_SQuAD.html'

    """
    proc = SquadV2Processor()
    data = proc.get_dev_examples(file_loc, file_name)
    return data

def get_triples(data):
    """This function was based on work from
    https://qa.fastforwardlabs.com/no%20answer/null%20threshold/bert/distilbert/exact%20match/f1/robust%20predictions/2020/06/09/Evaluating_BERT_on_SQuAD.html'

    """
    idx = list(range(len(data)))
    q = [data[i].question_text for i in idx]
    c = [data[i].context_text for i in idx]
    a = [[answer['text'] for answer in data[i].answers] for i in idx]
    return list(zip(q, a, c))

def exact_match(prediction, truth):
    count = 0
    for i in zip(truth, prediction):
        if i[0] == i[1]:
            count += 1
    return count/len(truth)
    
def question_answer(question, context):
    output = qa_model(question=question, context=context)
    return output


if __name__ == '__main__':
    data = load_data('./data/squad/', 'dev-v2.0.json')
    qac = random.sample(get_triples(data), 1000)
    values = [question_answer(x[0], x[2]) for x in qac]
    prediction =[x['answer'] for x in values]
    truth_values = [q[1] for q in qac]
    truth = ['' if len(x) == 0 else x[0] for x in truth_values]
    exact_match(prediction, truth)