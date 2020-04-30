import glob
import json
from os.path import join

DATA_PATH = '/data/luyang/dialogdata/wizardofwiki/goldknowledge/'
#DATA_PATH = '/data/luyang/dialogdata/wizardofwiki/noknowledge/'
WRITE_PATH = '/data/luyang/dialogdata/wizardofwiki/goldknowledge/bart/'
#WRITE_PATH = '/data/luyang/dialogdata/wizardofwiki/noknowledge/bart'

def write_bart_style_file(split):
    files = glob.glob(join(DATA_PATH, split, '*'))
    source = open(join(WRITE_PATH, split + '.source'), 'w')
    target = open(join(WRITE_PATH, split + '.target'), 'w')
    for i in range(len(files)):
        file = join(DATA_PATH, split, str(i) + '.json')
        with open(file) as f:
            data = json.load(f)
        article = ' '.join(data['article'])
        tgt = ' '.join(data['abstract'])

        source.write(article + '\n')
        target.write(tgt + '\n')


if __name__ == '__main__':
    for split in ['test_unseen', 'test_seen', 'val', 'train']:
        write_bart_style_file(split)


