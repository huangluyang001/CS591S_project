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

def tokenize(nlp, t):
    doc = nlp(t)
    content = []
    for sent in doc.sentences:
        sentence = []
        for word in sent.tokens:
            sentence.append(word.text)
        content.append(' '.join(sentence))
    return content

def write_data(i, path, topic, persona, turns, knowledge, abstract):
    with open(join(path, str(i) + '.json'), 'w') as f:
        data = {
            'article': topic + persona + turns + knowledge,
            'abstract': abstract,
            'topic': topic,
            'persona': persona,
            'turns': turns,
            'knowledge': knowledge
        }
        json.dump(data, f)



def preparedata_pg(split):
    assert split in ['test_unseen', 'test_seen', 'valid', 'train']

    nlp = stanfordnlp.Pipeline(processors='tokenize')

    data_count = 0

    write_path = '/data/luyang/dialogdata/wizardofwiki/noknowledge/' + split
    write_path2 = '/data/luyang/dialogdata/wizardofwiki/goldknowledge/' + split
    write_path3 = '/data/luyang/dialogdata/wizardofwiki/knowledge/' + split

    if not os.path.exists(write_path3):
        os.makedirs(write_path3)

    if not os.path.exists(write_path):
        os.makedirs(write_path)

    if not os.path.exists(write_path2):
        os.makedirs(write_path2)

    if split == 'valid':
        data_path = join(DATA_PATH, file_dict['valid_seen'])
        data_path_2 = join(DATA_PATH, file_dict['valid_unseen'])
        with open(data_path) as f, open(data_path_2) as f2:
            data = json.load(f)
            data_2 = json.load(f2)
        data = data + data_2
    else:
        data_path = join(DATA_PATH, file_dict[split])
        with open(data_path) as f:
            data = json.load(f)
    print('{} : {}'.format(split, len(data)))
    for i in range(len(data)):
        persona = data[i]['persona']
        topic = data[i]['chosen_topic']
        dialog = data[i]['dialog']
        persona = tokenize(nlp, persona)
        topic = tokenize(nlp, topic)
        previous_turns = []
        turn_labels = []
        knowledge = []
        for j, turn in enumerate(dialog):
            speaker = turn['speaker']
            text = turn['text']
            text = tokenize(nlp, text)
            # print('{}: {}  '.format(speaker, text))
            if 'Wizard' in speaker:
                sentence = turn['checked_sentence']
                if len(sentence.keys()) < 1:
                    sentence = []
                else:
                    if list(sentence.keys())[0] != 'no_passages_used':
                        sentence = tokenize(nlp, sentence[list(sentence.keys())[0]])
                    else:
                        sentence = []
                knowledge = ['<start>']
                if len(turn['retrieved_passages']) > 1:
                    for _dict in turn['retrieved_passages']:
                        for key, value in _dict.items():
                            knowledge += tokenize(nlp, key) + tokenize(nlp, ' '.join(value))


                #write_data(data_count, write_path, topic, persona, previous_turns, [], text)
                #write_data(data_count, write_path2, topic, persona, previous_turns, sentence, text)
                write_data(data_count, write_path3, topic, persona, previous_turns, knowledge, text)
                data_count += 1
                previous_turns.extend(['Wizard :'] + text)
            else:
                previous_turns.extend(['Apprentice :'] + text)


if __name__ == '__main__':
    for split in ['test_unseen', 'test_seen', 'val', 'train']:
        start = time.time()
        preparedata_pg(split)
        end = time.time()
        print('split {} finished in {}'.format(split, end - start))

