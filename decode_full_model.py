""" run decoding of rnn-ext + abs + RL (+ rerank)"""
import argparse
import json
import os
from os.path import join
from datetime import timedelta
from time import time
from collections import Counter, defaultdict
from itertools import product
from functools import reduce
import operator as op

from cytoolz import identity, concat, curry

import torch
from torch.utils.data import DataLoader
from torch import multiprocessing as mp

from data.batcher import tokenize, preproc
from data.RLbatcher import prepro_rl_graph, prepro_rl_graph_bert

from decoding import Abstractor, RLExtractor, DecodeDataset, BeamAbstractor, SCExtractor, DecodeDatasetEntity, DecodeDatasetGAT
from decoding import make_html_safe
from nltk import sent_tokenize
from pytorch_transformers import BertTokenizer, BertModel, BertConfig
import pickle

MAX_LEN_BERT = 512

def decode(save_path, model_dir, split, batch_size,
           beam_size, diverse, max_len, cuda, sc, min_len):
    start = time()
    # setup model
    with open(join(model_dir, 'meta.json')) as f:
        meta = json.loads(f.read())
    if meta['net_args']['abstractor'] is None:
    #if not meta['net_args'].__contains__('abstractor'):
        # NOTE: if no abstractor is provided then
        #       the whole model would be extractive summarization
        assert beam_size == 1
        abstractor = identity
    else:
        if beam_size == 1:
            abstractor = Abstractor(join(model_dir, 'abstractor'),
                                    max_len, cuda)
        else:
            abstractor = BeamAbstractor(join(model_dir, 'abstractor'),
                                        max_len, cuda, min_len)

    if sc:
        extractor = SCExtractor(model_dir, cuda=cuda)
    else:
        extractor = RLExtractor(model_dir, cuda=cuda)

    #check if use bert

    try:
        _bert = extractor._net._bert
    except:
        _bert = False
        print('no bert arg:')

    if _bert:
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking')
        print('bert tokenizer loaded')



    # setup loader
    def coll(batch):
        articles = list(filter(bool, batch))
        return articles
    dataset = DecodeDataset(split)

    n_data = len(dataset)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4,
        collate_fn=coll
    )

    # prepare save paths and logs
    os.makedirs(join(save_path, 'output'))
    dec_log = {}
    dec_log['abstractor'] = meta['net_args']['abstractor']
    dec_log['extractor'] = meta['net_args']['extractor']
    dec_log['rl'] = True
    dec_log['split'] = split
    dec_log['beam'] = beam_size
    dec_log['diverse'] = diverse
    with open(join(save_path, 'log.json'), 'w') as f:
        json.dump(dec_log, f, indent=4)

    # Decoding
    if sc:
        i = 0
        length = 0
        with torch.no_grad():
            for i_debug, raw_article_batch in enumerate(loader):
                tokenized_article_batch = map(tokenize(None), raw_article_batch)
                ext_arts = []
                ext_inds = []
                if _bert:
                    for raw_art_sents, raw_art in zip(tokenized_article_batch, raw_article_batch):
                        tokenized_sents = [tokenizer.tokenize(source_sent.lower()) for source_sent in raw_art]
                        tokenized_sents = [tokenized_sent + ['[SEP]'] for tokenized_sent in tokenized_sents]
                        tokenized_sents[0] = ['[CLS]'] + tokenized_sents[0]
                        word_num = [len(tokenized_sent) for tokenized_sent in tokenized_sents]
                        truncated_word_num = []
                        total_count = 0
                        for num in word_num:
                            if total_count + num < MAX_LEN_BERT:
                                truncated_word_num.append(num)
                            else:
                                truncated_word_num.append(MAX_LEN_BERT - total_count)
                                break
                            total_count += num
                        tokenized_sents = list(concat(tokenized_sents))[:MAX_LEN_BERT]
                        tokenized_sents = tokenizer.convert_tokens_to_ids(tokenized_sents)
                        art_sents = tokenize(None, raw_art)
                        _input = (art_sents, tokenized_sents, truncated_word_num)

                        ext = extractor(_input)[:]  # exclude EOE
                        if not ext:
                            # use top-3 if nothing is extracted
                            # in some rare cases rnn-ext does not extract at all
                            ext = list(range(3))[:len(raw_art_sents)]
                        else:
                            ext = [i for i in ext]
                        ext_inds += [(len(ext_arts), len(ext))]
                        ext_arts += [raw_art_sents[i] for i in ext]
                else:
                    for raw_art_sents in tokenized_article_batch:
                        ext = extractor(raw_art_sents)[:]  # exclude EOE
                        if not ext:
                            # use top-5 if nothing is extracted
                            # in some rare cases rnn-ext does not extract at all
                            ext = list(range(5))[:len(raw_art_sents)]
                        else:
                            ext = [i for i in ext]
                        ext_inds += [(len(ext_arts), len(ext))]
                        ext_arts += [raw_art_sents[i] for i in ext]
                if beam_size > 1:
                    all_beams = abstractor(ext_arts, beam_size, diverse)
                    dec_outs = rerank_mp(all_beams, ext_inds)
                else:
                    dec_outs = abstractor(ext_arts)
                assert i == batch_size*i_debug
                for j, n in ext_inds:
                    decoded_sents = [' '.join(dec) for dec in dec_outs[j:j+n]]
                    with open(join(save_path, 'output/{}.dec'.format(i)),
                              'w') as f:
                        f.write(make_html_safe('\n'.join(decoded_sents)))
                    i += 1
                    print('{}/{} ({:.2f}%) decoded in {} seconds\r'.format(
                        i, n_data, i/n_data*100,
                        timedelta(seconds=int(time()-start))
                    ), end='')
                    length += len(decoded_sents)
    else:
        i = 0
        length = 0
        with torch.no_grad():
            for i_debug, raw_article_batch in enumerate(loader):
                tokenized_article_batch = map(tokenize(None), raw_article_batch)
                ext_arts = []
                ext_inds = []
                for raw_art_sents in tokenized_article_batch:
                    ext = extractor(raw_art_sents)[:-1]  # exclude EOE
                    if not ext:
                        # use top-5 if nothing is extracted
                        # in some rare cases rnn-ext does not extract at all
                        ext = list(range(5))[:len(raw_art_sents)]
                    else:
                        ext = [i.item() for i in ext]
                    ext_inds += [(len(ext_arts), len(ext))]
                    ext_arts += [raw_art_sents[i] for i in ext]
                if beam_size > 1:
                    all_beams = abstractor(ext_arts, beam_size, diverse)
                    dec_outs = rerank_mp(all_beams, ext_inds)
                else:
                    dec_outs = abstractor(ext_arts)
                assert i == batch_size*i_debug
                for j, n in ext_inds:
                    decoded_sents = [' '.join(dec) for dec in dec_outs[j:j+n]]
                    with open(join(save_path, 'output/{}.dec'.format(i)),
                              'w') as f:
                        f.write(make_html_safe('\n'.join(decoded_sents)))
                    i += 1
                    print('{}/{} ({:.2f}%) decoded in {} seconds\r'.format(
                        i, n_data, i/n_data*100,
                        timedelta(seconds=int(time()-start))
                    ), end='')
                    length += len(decoded_sents)
    print('average summary length:', length / i)

def decode_graph(save_path, model_dir, split, batch_size,
           beam_size, diverse, max_len, cuda, sc, min_len, docgraph, paragraph):
    start = time()
    # setup model
    with open(join(model_dir, 'meta.json')) as f:
        meta = json.loads(f.read())
    if meta['net_args']['abstractor'] is None:
    #if not meta['net_args'].__contains__('abstractor'):
        # NOTE: if no abstractor is provided then
        #       the whole model would be extractive summarization
        assert beam_size == 1
        abstractor = identity
    else:
        if beam_size == 1:
            abstractor = Abstractor(join(model_dir, 'abstractor'),
                                    max_len, cuda)
        else:
            abstractor = BeamAbstractor(join(model_dir, 'abstractor'),
                                        max_len, cuda, min_len=min_len)

    print('docgraph:', docgraph)
    extractor = SCExtractor(model_dir, cuda=cuda, docgraph=docgraph, paragraph=paragraph)
    adj_type = extractor._net._adj_type
    bert = extractor._net._bert
    if bert:
        tokenizer = extractor._net._bert
        try:
            with open('/data/luyang/process-nyt/bert_tokenizaiton_aligns/robertaalign-base-cased.pkl', 'rb') as f:
                align = pickle.load(f)
        except FileNotFoundError:
            with open('/data2/luyang/process-nyt/bert_tokenizaiton_aligns/robertaalign-base-cased.pkl', 'rb') as f:
                align = pickle.load(f)

        try:
            with open('/data/luyang/process-cnn-dailymail/bert_tokenizaiton_aligns/robertaalign-base-cased.pkl',
                      'rb') as f:
                align2 = pickle.load(f)
        except FileNotFoundError:
            with open('/data2/luyang/process-cnn-dailymail/bert_tokenizaiton_aligns/robertaalign-base-cased.pkl',
                      'rb') as f:
                align2 = pickle.load(f)

        align.update(align2)


    # setup loader
    def coll(batch):
        batch = list(filter(bool, batch))
        return batch

    dataset = DecodeDatasetGAT(split, args.key)

    n_data = len(dataset)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4,
        collate_fn=coll
    )

    # prepare save paths and logs
    os.makedirs(join(save_path, 'output'))
    dec_log = {}
    dec_log['abstractor'] = meta['net_args']['abstractor']
    dec_log['extractor'] = meta['net_args']['extractor']
    dec_log['rl'] = True
    dec_log['split'] = split
    dec_log['beam'] = beam_size
    dec_log['diverse'] = diverse
    with open(join(save_path, 'log.json'), 'w') as f:
        json.dump(dec_log, f, indent=4)

    # Decoding

    i = 0
    length = 0
    sent_selected = 0
    with torch.no_grad():
        for i_debug, raw_input_batch in enumerate(loader):
            raw_article_batch, nodes, edges, paras, subgraphs = zip(*raw_input_batch)
            if bert:
                art_sents = [[tokenizer.tokenize(source_sent) for source_sent in source_sents] for source_sents in raw_article_batch]
                for _i in range(len(art_sents)):
                    art_sents[_i][0] = [tokenizer.bos_token] + art_sents[_i][0]
                    art_sents[_i][-1] = art_sents[_i][-1] + [tokenizer.eos_token]
                truncated_word_nums = []
                word_nums = [[len(sent) for sent in art_sent] for art_sent in art_sents]
                for word_num in word_nums:
                    truncated_word_num = []
                    total_count = 0
                    for num in word_num:
                        if total_count + num < args.max_dec_word:
                            truncated_word_num.append(num)
                        else:
                            truncated_word_num.append(args.max_dec_word - total_count)
                            break
                        total_count += num
                    truncated_word_nums.append(truncated_word_num)
                sources = [list(concat(art_sent))[:args.max_dec_word] for art_sent in art_sents]
            else:
                tokenized_article_batch = map(tokenize(None), raw_article_batch)
            #processed_clusters = map(preproc(list(tokenized_article_batch), clusters))
            #processed_clusters = list(zip(*processed_clusters))
            ext_arts = []
            ext_inds = []
            pre_abs = []
            beam_inds = []
            if bert:
                for raw_art_sents, source, art_sent, word_num, raw_nodes, raw_edges, raw_paras, raw_subgraphs in zip(raw_article_batch, sources, art_sents, truncated_word_nums, nodes, edges, paras, subgraphs):
                    processed_nodes = prepro_rl_graph_bert(align, raw_art_sents, source, art_sent, args.max_dec_word, raw_nodes, raw_edges, raw_paras, raw_subgraphs, adj_type, docgraph)
                    _input = (raw_art_sents, source) + processed_nodes + (word_num, )
                    ext = extractor(_input)[:]
                    sent_selected += len(ext)
                    if not ext:
                        # use top-3 if nothing is extracted
                        # in some rare cases rnn-ext does not extract at all
                        ext = list(range(3))[:len(raw_art_sents)]
                    else:
                        ext = [i for i in ext]
                    ext_art = list(map(lambda i: raw_art_sents[i], ext))
                    pre_abs.append([word for sent in ext_art for word in sent])
                    beam_inds += [(len(beam_inds), 1)]

            else:
                for raw_art_sents, raw_nodes, raw_edges, raw_paras, raw_subgraphs in zip(tokenized_article_batch, nodes, edges, paras, subgraphs):
                    processed_nodes = prepro_rl_graph(raw_art_sents, raw_nodes, raw_edges, raw_paras, raw_subgraphs, adj_type, docgraph)
                    _input = (raw_art_sents, ) + processed_nodes

                    ext = extractor(_input)[:]  # exclude EOE
                    sent_selected += len(ext)
                    if not ext:
                        # use top-3 if nothing is extracted
                        # in some rare cases rnn-ext does not extract at all
                        ext = list(range(3))[:len(raw_art_sents)]
                    else:
                        ext = [i for i in ext]
                    ext_art = list(map(lambda i: raw_art_sents[i], ext))
                    pre_abs.append([word for sent in ext_art for word in sent])
                    beam_inds += [(len(beam_inds), 1)]

            if beam_size > 1:
                # all_beams = abstractor(ext_arts, beam_size, diverse)
                # dec_outs = rerank_mp(all_beams, ext_inds)
                all_beams = abstractor(pre_abs, beam_size, diverse=1.0)
                dec_outs = rerank_mp(all_beams, beam_inds)
            else:
                dec_outs = abstractor(pre_abs)
            for dec_out in dec_outs:
                dec_out = sent_tokenize(' '.join(dec_out))
                ext = [sent.split(' ') for sent in dec_out]
                ext_inds += [(len(ext_arts), len(ext))]
                ext_arts += ext
            dec_outs = ext_arts
            assert i == batch_size*i_debug
            for j, n in ext_inds:
                decoded_sents = [' '.join(dec) for dec in dec_outs[j:j+n]]
                with open(join(save_path, 'output/{}.dec'.format(i)),
                          'w') as f:
                    f.write(make_html_safe('\n'.join(decoded_sents)))
                i += 1
                print('{}/{} ({:.2f}%) decoded in {} seconds\r'.format(
                    i, n_data, i/n_data*100,
                    timedelta(seconds=int(time()-start))
                ), end='')
                length += len(decoded_sents)
    print('average summary length:', length / i)
    print('average sentence selected:', sent_selected)



_PRUNE = defaultdict(
    lambda: 2,
    {1:5, 2:5, 3:5, 4:5, 5:5, 6:4, 7:3, 8:3}
)

def rerank(all_beams, ext_inds):
    beam_lists = (all_beams[i: i+n] for i, n in ext_inds if n > 0)
    return list(concat(map(rerank_one, beam_lists)))

def rerank_mp(all_beams, ext_inds):
    beam_lists = [all_beams[i: i+n] for i, n in ext_inds if n > 0]
    with mp.Pool(8) as pool:
        reranked = pool.map(rerank_one, beam_lists)
    return list(concat(reranked))

def rerank_one(beams):
    @curry
    def process_beam(beam, n):
        for b in beam[:n]:
            b.gram_cnt = Counter(_make_n_gram(b.sequence))
        return beam[:n]
    beams = map(process_beam(n=_PRUNE[len(beams)]), beams)
    best_hyps = max(product(*beams), key=_compute_score)
    dec_outs = [h.sequence for h in best_hyps]
    return dec_outs

def _make_n_gram(sequence, n=2):
    return (tuple(sequence[i:i+n]) for i in range(len(sequence)-(n-1)))

def _make_tri_gram(sequence, n=3):
    return (tuple(sequence[i:i+n]) for i in range(len(sequence)-(n-1)) if '.' not in tuple(sequence[i:i+n]))

def length_wu(cur_len, alpha=0.):
    """GNMT length re-ranking score.
    See "Google's Neural Machine Translation System" :cite:`wu2016google`.
    """
    return ((5 + cur_len) / 6.0) ** alpha


def coverage_summary(cov, beta=0.):
    """Our summary penalty."""
    penalty = torch.max(cov, cov.clone().fill_(1.0)).sum(-1)
    penalty -= cov.size(-1)
    return beta * penalty

def _compute_score(hyps):
    all_cnt = reduce(op.iadd, (h.gram_cnt for h in hyps), Counter())
    # repeat = sum(c-1 for g, c in all_cnt.items() if c > 1)
    # try:
    #     lp = sum(h.logprob for h in hyps) / sum(len(h.sequence) for h in hyps)
    # except ZeroDivisionError:
    #     lp = -1e5
    for h in hyps:
        if h.coverage is None:
            print(h.sequence)
    try:
        lp = sum(h.logprob for h in hyps) / sum(length_wu(len(h.sequence)+1, alpha=0.9) - coverage_summary(h.coverage, beta=5) for h in hyps)
    except ZeroDivisionError:
        lp = -1e5
    # for h in hyps:
    #     print(h.sequence)
    #     tri_grams = _make_tri_gram(h.sequence)
    #     cnt = Counter(tri_grams)
    #     if not all((cnt[g] <= 1 for g in cnt)):
    #         lp = lp - 1e5

    # length = sum([len(h.sequence) for h in hyps]) + 1 # include EOS
    # len_pen = length_wu(length, alpha=0.9)
    # try:
    #     lp = lp / len_pen - sum(coverage_summary(h.coverage, beta=5) for h in hyps)
    # except:
    #     print(hyps[0].sequence)

    # return (-repeat, lp)
    return lp


if __name__ == '__main__':
    torch.cuda.set_device(0)
    parser = argparse.ArgumentParser(
        description='run decoding of the full model (RL)')
    parser.add_argument('--path', required=True, help='path to store/eval')
    parser.add_argument('--model_dir', help='root of the full model')
    parser.add_argument('--sc', action='store_true', help='self critical')
    parser.add_argument('--docgraph', action='store_true', help='docgraph model')
    parser.add_argument('--paragraph', action='store_true', help='paragraph model')
    parser.add_argument('--key', type=str, default='nodes_pruned2', help='use which cluster type')

    # dataset split
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument('--val', action='store_true', help='use validation set')
    data.add_argument('--test', action='store_true', help='use test set')

    # decode options
    parser.add_argument('--batch', type=int, action='store', default=32,
                        help='batch size of faster decoding')
    parser.add_argument('--beam', type=int, action='store', default=1,
                        help='beam size for beam-search (reranking included)')
    parser.add_argument('--div', type=float, action='store', default=1.0,
                        help='diverse ratio for the diverse beam-search')
    parser.add_argument('--max_dec_word', type=int, action='store', default=100,
                        help='maximun words to be decoded for the abstractor')
    parser.add_argument('--min_dec_word', type=int, action='store', default=0,
                        help='maximun words to be decoded for the abstractor')

    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    data_split = 'test' if args.test else 'val'
    if args.docgraph or args.paragraph:
        decode_graph(args.path, args.model_dir,
               data_split, args.batch, args.beam, args.div,
               args.max_dec_word, args.cuda, args.sc, args.min_dec_word, args.docgraph, args.paragraph)
    else:
        decode(args.path, args.model_dir,
           data_split, args.batch, args.beam, args.div,
           args.max_dec_word, args.cuda, args.sc, args.min_dec_word)
