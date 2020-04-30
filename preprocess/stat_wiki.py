import json
import os
from os.path import join
import stanfordnlp
import time

DATA_PATH = '/home/luyang/emnlp2020/ParlAI/data/wizard_of_wikipedia/'

file_dict = {
    'test_unseen': 'test_topic_split.json',
    'test_seen': 'test_random_split.json',
    'valid_unseen': 'valid_topic_split.json',
    'valid_seen': 'valid_random_split.json',
    'train': 'train.json'

}

def stats(split):
    assert split in ['test_unseen', 'test_seen', 'train']

    wizard_count = 0
    knowledge_count = 0
    no_knowledge_count = 0
    question_count = 0


    data_path = join(DATA_PATH, file_dict[split])
    with open(data_path) as f:
        data = json.load(f)
    print('{} : {}'.format(split, len(data)))
    for i in range(len(data)):
        dialog = data[i]['dialog']
        kl = set()
        for j, turn in enumerate(dialog):
            speaker = turn['speaker']
            text = turn['text']
            # print('{}: {}  '.format(speaker, text))
            if 'Wizard' in speaker:
                wizard_count += 1
                sentence = turn['checked_sentence']
                if '?' in text:
                    question_count += 1
                knowledge = turn['checked_passage']
                if len(knowledge.keys()) < 1:
                    no_knowledge_count += 1
                else:
                    if list(knowledge.keys())[0] == 'no_passages_used':
                        no_knowledge_count += 1
                    else:
                        kl.add(list(knowledge.keys())[0])
        knowledge_count += len(kl)
    print('wizard_count:', wizard_count)
    print('knowledge count:', knowledge_count)
    print('no knowledge selection:', no_knowledge_count)
    print('question number:', question_count)
    print('number of dialogues:', len(data))


if __name__ == '__main__':
    for split in ['test_unseen', 'test_seen']:
        stats(split)

