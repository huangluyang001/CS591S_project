""" evaluation scripts"""
import re
import os
from os.path import join
import logging
import tempfile
import subprocess as sp
import math
import collections
#import bert_score

#from my_bert_score import my_bert_score, my_q_bleu
from transformers import AutoTokenizer
import torch
import numpy as np

from cytoolz import curry

from pyrouge import Rouge155
from pyrouge.utils import log


try:
    _ROUGE_PATH = os.environ['ROUGE']
except KeyError:
    print('Warning: ROUGE is not configured')
    _ROUGE_PATH = None
def eval_rouge(dec_pattern, dec_dir, ref_pattern, ref_dir,
               cmd='-c 95 -r 1000 -n 2 -m -d', system_id=1):
    print('evaluate')
    """ evaluate by original Perl implementation"""
    # silence pyrouge logging
    assert _ROUGE_PATH is not None
    log.get_global_console_logger().setLevel(logging.WARNING)
    with tempfile.TemporaryDirectory() as tmp_dir:
        Rouge155.convert_summaries_to_rouge_format(
            dec_dir, join(tmp_dir, 'dec'))
        Rouge155.convert_summaries_to_rouge_format(
            ref_dir, join(tmp_dir, 'ref'))
        Rouge155.write_config_static(
            join(tmp_dir, 'dec'), dec_pattern,
            join(tmp_dir, 'ref'), ref_pattern,
            join(tmp_dir, 'settings.xml'), system_id
        )
        cmd = (join(_ROUGE_PATH, 'ROUGE-1.5.5.pl')
               + ' -e {} '.format(join(_ROUGE_PATH, 'data'))
               + cmd
               + ' -a {}'.format(join(tmp_dir, 'settings.xml')))
        output = sp.check_output(cmd.split(' '), universal_newlines=True)

    rouge_1 = []
    rouge_2 = []
    rouge_l = []

    for line in output.split('\n'):
        if 'ROUGE-1 Eval' in line:
            rouge_1.append(line.split()[-1][2:])
        if 'ROUGE-2 Eval' in line:
            rouge_2.append(line.split()[-1][2:])
        if 'ROUGE-L Eval' in line:
            rouge_l.append(line.split()[-1][2:])

    rouge_1 = '\n'.join(rouge_1)
    rouge_2 = '\n'.join(rouge_2)
    rouge_l = '\n'.join(rouge_l)

    return rouge_1, rouge_2, rouge_l


try:
    _METEOR_PATH = os.environ['METEOR']
except KeyError:
    print('Warning: METEOR is not configured')
    _METEOR_PATH = None
def eval_meteor(dec_pattern, dec_dir, ref_pattern, ref_dir):
    """ METEOR evaluation"""
    assert _METEOR_PATH is not None
    ref_matcher = re.compile(ref_pattern)
    refs = sorted([r for r in os.listdir(ref_dir) if ref_matcher.match(r)],
                  key=lambda name: int(name.split('.')[0]))
    dec_matcher = re.compile(dec_pattern)
    decs = sorted([d for d in os.listdir(dec_dir) if dec_matcher.match(d)],
                  key=lambda name: int(name.split('.')[0]))
    @curry
    def read_file(file_dir, file_name):
        with open(join(file_dir, file_name)) as f:
            return ' '.join(f.read().split())
    with tempfile.TemporaryDirectory() as tmp_dir:
        with open(join(tmp_dir, 'ref.txt'), 'w') as ref_f,\
             open(join(tmp_dir, 'dec.txt'), 'w') as dec_f:
            ref_f.write('\n'.join(map(read_file(ref_dir), refs)) + '\n')
            dec_f.write('\n'.join(map(read_file(dec_dir), decs)) + '\n')

        cmd = 'java -Xmx2G -jar {} {} {} -l en -norm'.format(
            _METEOR_PATH, join(tmp_dir, 'dec.txt'), join(tmp_dir, 'ref.txt'))
        output = sp.check_output(cmd.split(' '), universal_newlines=True)

    scores = []
    for line in output.split('\n'):
        if 'Segment' in line:
            scores.append(line.split()[-1])

    output = '\n'.join(scores)

    return output


def eval_qbleu(dec_pattern, dec_dir, ref_pattern, ref_dir):
    ref_matcher = re.compile(ref_pattern)
    refs = sorted([r for r in os.listdir(ref_dir) if ref_matcher.match(r)],
                  key=lambda name: int(name.split('.')[0]))
    dec_matcher = re.compile(dec_pattern)
    decs = sorted([d for d in os.listdir(dec_dir) if dec_matcher.match(d)],
                  key=lambda name: int(name.split('.')[0]))

    @curry
    def read_file(file_dir, file_name):
        with open(join(file_dir, file_name)) as f:
            return ' '.join(f.read().strip().split())

    ref_corpus = list(filter(bool, map(read_file(ref_dir), refs)))
    dec_corpus = list(filter(bool, map(read_file(dec_dir), decs)))

    scores = my_q_bleu.get_answerability_scores(dec_corpus, ref_corpus, 0.6, 0.2, 0.1, ngram_metric='Bleu_1')

    output = '\n'.join([str(score) for score in scores])

    return output


def eval_bert_score(dec_pattern, dec_dir, ref_pattern, ref_dir):
    ref_matcher = re.compile(ref_pattern)
    refs = sorted([r for r in os.listdir(ref_dir) if ref_matcher.match(r)],
                  key=lambda name: int(name.split('.')[0]))
    dec_matcher = re.compile(dec_pattern)
    decs = sorted([d for d in os.listdir(dec_dir) if dec_matcher.match(d)],
                  key=lambda name: int(name.split('.')[0]))

    @curry
    def read_file(file_dir, file_name):
        with open(join(file_dir, file_name)) as f:
            return ' '.join(f.read().split())

    ref_corpus = map(read_file(ref_dir), refs)
    dec_corpus = map(read_file(dec_dir), decs)

    ref_corpus = list(ref_corpus)
    dec_corpus = list(dec_corpus)

    score = bert_score.score(dec_corpus, ref_corpus, '/data2/shuyang/pretrain_language_models/roberta-large', verbose=True)

    p, r, f = score[0].tolist(), score[1].tolist(), score[2].tolist()

    p = '\n'.join([str(pp) for pp in p])
    r = '\n'.join([str(rr) for rr in r])
    f = '\n'.join([str(ff) for ff in f])

    return p, r, f


def eval_semantic_score(dec_pattern, dec_dir, ref_pattern, ref_dir, match_type, match_stop):
    ref_matcher = re.compile(ref_pattern)
    refs = sorted([r for r in os.listdir(ref_dir) if ref_matcher.match(r)],
                  key=lambda name: int(name.split('.')[0]))
    dec_matcher = re.compile(dec_pattern)
    decs = sorted([d for d in os.listdir(dec_dir) if dec_matcher.match(d)],
                  key=lambda name: int(name.split('.')[0]))

    @curry
    def read_file(file_dir, file_name):
        with open(join(file_dir, file_name)) as f:
            return ' '.join(f.read().split())

    ref_corpus = map(read_file(ref_dir), refs)
    dec_corpus = map(read_file(dec_dir), decs)

    ref_corpus = list(ref_corpus)
    dec_corpus = list(dec_corpus)

    model = my_bert_score.get_model('/data2/shuyang/pretrain_language_models/roberta-large',
                                    my_bert_score.model2layers['roberta-large'])
    tokenizer = AutoTokenizer.from_pretrained('/data2/shuyang/pretrain_language_models/roberta-large')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.to(device)

    scores = my_bert_score.get_bert_cos_sim(dec_corpus, ref_corpus, model, tokenizer, device=device, match_type=match_type,
                                           mask_stop=not match_stop, normalize=True, len_penalty=True, verbose=True)

    output = '\n'.join([str(score) for score in scores])

    return output


def eval_bleu(dec_pattern, dec_dir, ref_pattern, ref_dir):
    ref_matcher = re.compile(ref_pattern)
    refs = sorted([r for r in os.listdir(ref_dir) if ref_matcher.match(r)],
                  key=lambda name: int(name.split('.')[0]))
    dec_matcher = re.compile(dec_pattern)
    decs = sorted([d for d in os.listdir(dec_dir) if dec_matcher.match(d)],
                  key=lambda name: int(name.split('.')[0]))

    @curry
    def read_file(file_dir, file_name):
        with open(join(file_dir, file_name)) as f:
            return f.read().split()

    ref_corpus = map(read_file(ref_dir), refs)
    dec_corpus = map(read_file(dec_dir), decs)

    scores = []
    for ref, dec in zip(ref_corpus, dec_corpus):
        bleu_scores = collections.OrderedDict()
        for max_order in [1, 2, 3, 4]:
            bleu, precisions, bp, ratio, translation_length, reference_length = compute_bleu([[ref]], [dec],
                                                                                             max_order)
            bleu_scores[max_order] = bleu
        scores.append(bleu_scores[4])

    output = '\n'.join([str(score) for score in scores])

    return output


def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=False):
  """Computes BLEU score of translated segments against one or more references.
  Args:
    reference_corpus: list of lists of references for each translation. Each
        reference should be tokenized into a list of tokens.
    translation_corpus: list of translations to score. Each translation
        should be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.
    smooth: Whether or not to apply Lin et al. 2004 smoothing.
  Returns:
    3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
    precisions and brevity penalty.
  """
  matches_by_order = [0] * max_order
  possible_matches_by_order = [0] * max_order
  reference_length = 0
  translation_length = 0
  for (references, translation) in zip(reference_corpus,
                                       translation_corpus):
    reference_length += min(len(r) for r in references)
    translation_length += len(translation)

    merged_ref_ngram_counts = collections.Counter()
    for reference in references:
      merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
    translation_ngram_counts = _get_ngrams(translation, max_order)
    overlap = translation_ngram_counts & merged_ref_ngram_counts
    for ngram in overlap:
      matches_by_order[len(ngram)-1] += overlap[ngram]
    for order in range(1, max_order+1):
      possible_matches = len(translation) - order + 1
      if possible_matches > 0:
        possible_matches_by_order[order-1] += possible_matches

  precisions = [0] * max_order
  for i in range(0, max_order):
    if smooth:
      precisions[i] = ((matches_by_order[i] + 1.) /
                       (possible_matches_by_order[i] + 1.))
    else:
      if possible_matches_by_order[i] > 0:
        precisions[i] = (float(matches_by_order[i]) /
                         possible_matches_by_order[i])
      else:
        precisions[i] = 0.0

  if min(precisions) > 0:
    p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
    geo_mean = math.exp(p_log_sum)
  else:
    geo_mean = 0

  ratio = float(translation_length) / reference_length

  if ratio > 1.0:
    bp = 1.
  elif ratio == 0.0:
    bp = 0.0
  else:
    bp = math.exp(1 - 1. / ratio)

  bleu = geo_mean * bp

  return (bleu, precisions, bp, ratio, translation_length, reference_length)


def _get_ngrams(segment, max_order):
  """Extracts all n-grams upto a given maximum order from an input segment.
  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this
        methods.
  Returns:
    The Counter containing all n-grams upto max_order in segment
    with a count of how many times each n-gram occurred.
  """
  ngram_counts = collections.Counter()
  for order in range(1, max_order + 1):
    for i in range(0, len(segment) - order + 1):
      ngram = tuple(segment[i:i+order])
      ngram_counts[ngram] += 1
  return ngram_counts