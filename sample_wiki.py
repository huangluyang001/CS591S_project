import json
from os.path import join



pg_nok = '/data/luyang/dialog/wizardofwiki/results/pg_nok/'
pg_goldk = '/data/luyang/dialog/wizardofwiki/results/pg_goldk/'
data = '/data/luyang/dialogdata/wizardofwiki/goldknowledge/'
bart_nok = '/data/luyang/dialog/wizardofwiki/results/bart_nok/'

bart_noks = []
with open(join(bart_nok, 'test_unseen.hypo')) as f:
    for line in f:
        bart_noks.append(line.strip())

for i in range(100):
    with open(join(data, 'test_unseen', str(i) + '.json')) as g:
        data_dict = json.load(g)

    with open(join(pg_goldk, 'test_unseen', 'output', str(i) + '.dec')) as f:
        output = []
        for line in f:
            output.append(line.strip())

    with open(join(pg_nok, 'test_unseen', 'output', str(i) + '.dec')) as f:
        noks = []
        for line in f:
            noks.append(line.strip())
    print('Topic:', data_dict['topic'])
    print('previous turns:  ')
    if len(data_dict['turns']) > 0:
        print('\n'.join(data_dict['turns']))
    print('\n')
    print('persona: ', data_dict['persona'])
    print('\n')
    print('gold knowledge:', data_dict['knowledge'])
    print('\n')
    print('human reply: ', data_dict['abstract'][0])
    print('pg gold knowledge: ', ' '.join(output))
    print('pg no knowledge: ', ' '.join(noks))
    print('bart no knowledge: ', bart_noks[i])

    print('\n\n')