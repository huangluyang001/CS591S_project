import json
from collections import Counter
import re
from bleu_scorer import BleuScorer
from os.path import join
import glob
from copy import deepcopy
import argparse

re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')

def normalize(s):
    # from https://github.com/facebookresearch/ParlAI/blob/df8926ad0436d391fe3cf56c0b5c236d1bce29f6/parlai/core/metrics.py#L290
    s = s.lower()
    s = re_punc.sub(' ', s)
    s = re_art.sub(' ', s)
    s = ' '.join(s.split())
    return s


def compute_f1(output, ref):
    # input: string
    # lower text and remove punctuation extra whitespace.
    output = normalize(output).split(' ')
    ref = normalize(ref).split(' ')

    common = Counter(output) & Counter(ref)
    num_same = sum(common.values())
    if num_same == 0:
        return 0

    p = 1.0 * num_same / len(output)
    r = 1.0 * num_same / len(ref)
    f = (2 * p * r) / (p + r)
    return f

def bleu(output, ref):
    scorer = BleuScorer(n=4)
    scorer += (output.lower(), [ref.lower()])
    score, _ = scorer.compute_score()

    return score

def compute_pg(out_path, ref_path):
    score_dict = {
        'b1': [],
        'b2': [],
        'b3': [],
        'b4': [],
        'f1': []
    }
    score_dicts = {
        'test_unseen': deepcopy(score_dict),
        'test_seen': deepcopy(score_dict)
    }
    for split in ['test_seen', 'test_unseen']:
        files = glob.glob(join(out_path, split, 'output', '*'))
        for i in range(len(files)):
            ref_file = join(ref_path, split, str(i) + '.json')
            out_file = join(out_path, split, 'output', str(i) + '.dec')
            with open(ref_file) as f, open(out_file) as g:
                ref = json.load(f)
                ref = ' '.join(ref['abstract'])
                predict = []
                for line in g:
                    predict.append(line.strip())
                predict = ' '.join(predict)
            score_bleu = bleu(predict, ref)
            score_f1 = compute_f1(predict, ref)
            score_dicts[split]['b1'].append(score_bleu[0])
            score_dicts[split]['b2'].append(score_bleu[1])
            score_dicts[split]['b3'].append(score_bleu[2])
            score_dicts[split]['b4'].append(score_bleu[3])
            score_dicts[split]['f1'].append(score_f1)
    return score_dicts

def compute_bart(out_path, ref_path):
    score_dict = {
        'b1': [],
        'b2': [],
        'b3': [],
        'b4': [],
        'f1': []
    }
    score_dicts = {
        'test_unseen': deepcopy(score_dict),
        'test_seen': deepcopy(score_dict)
    }
    for split in ['test_seen', 'test_unseen']:
        files = glob.glob(join(ref_path, split, '*'))
        bart_file = join(out_path, split + '.hypo')
        outputs = []
        with open(bart_file) as f:
            for line in f:
                outputs.append(line.strip())
        for i in range(len(files)):
            ref_file = join(ref_path, split, str(i) + '.json')
            with open(ref_file) as f:
                ref = json.load(f)
                ref = ' '.join(ref['abstract'])
            predict = outputs[i]
            score_bleu = bleu(predict, ref)
            score_f1 = compute_f1(predict, ref)
            score_dicts[split]['b1'].append(score_bleu[0])
            score_dicts[split]['b2'].append(score_bleu[1])
            score_dicts[split]['b3'].append(score_bleu[2])
            score_dicts[split]['b4'].append(score_bleu[3])
            score_dicts[split]['f1'].append(score_f1)

    return score_dicts

def write_scores(out_path, ref_path, score_dicts):
    # write to files not built yet
    for key, value in score_dicts.items():
        print('{}:'.format(key))
        for score_type, values in value.items():
            print('{}: {}'.format(score_type, sum(values) / len(values)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=('bart decoding')
    )
    parser.add_argument('--out_path', required=True, help='path to out')
    parser.add_argument('--ref_path', required=True, help='path to ref')
    parser.add_argument('--type', required=True, help='ref type')
    args = parser.parse_args()


    if args.type == 'pg':
        score_dicts = compute_pg(args.out_path, args.ref_path)
        write_scores(args.out_path, args.ref_path, score_dicts)
    elif args.type == 'bart':
        score_dicts = compute_bart(args.out_path, args.ref_path)
        write_scores(args.out_path, args.ref_path, score_dicts)
    else:
        raise Exception

