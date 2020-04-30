import glob
import json, pickle, os, time
import collections
from cytoolz import concat

if __name__ == '__main__':
    #finished_files_dir = '/data2/luyang/process-nyt/finished_files/'
    file_dirs = [
        #'/data/luyang/dialogdata/wizardofwiki/noknowledge/',
        #'/data/luyang/dialogdata/wizardofwiki/goldknowledge/'
        '/data2/luyang/dialogdata/wizardofwiki/knowledge/'
    ]
    #finished_files_dir = '/data/luyang/dialogdata/wizardofwiki/noknowledge/'
    #finished_files_dir2 = '/data/luyang/dialogdata/wizardofwiki/goldknowledge/'
    for finished_files_dir in file_dirs:
        files = glob.glob(finished_files_dir + 'train/*')
        vocab_counter = collections.Counter()
        cl_words = []
        id = 0
        start = time.time()
        for file in files:
            with open(file, 'r') as f:
                try:
                    data = json.load(f)
                except:
                    print('error file')
                    print(file)
                    continue
                raw_article = data['article']
                abss = data['abstract']
                cnt = [_.lower().split(' ') for _ in raw_article]
                abs = [_.lower().split(' ') for _ in abss]
                art_tokens = list(concat(cnt))
                abs_tokens = list(concat(abs))
                tokens = art_tokens + abs_tokens
                tokens = [t.strip() for t in tokens]  # strip
                tokens = [t for t in tokens if t != ""]  # remove empty
                vocab_counter.update(tokens)
                id += 1
                if id % 10000 == 0:
                    print('{} finished'.format(id))
                    print(time.time() - start)

        print("Writing vocab file...")
        with open(os.path.join(finished_files_dir, "vocab_cnt.pkl"),
                  'wb') as vocab_file:
            pickle.dump(vocab_counter, vocab_file)
        print("Finished writing vocab file")
        print(sorted(vocab_counter.items(), key=lambda x:x[1], reverse=True)[:50])
        vocab = dict(sorted(vocab_counter.items(), key=lambda x:x[1], reverse=True)[:50000])