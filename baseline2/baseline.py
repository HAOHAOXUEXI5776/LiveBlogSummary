import sys
import os
import argparse

'''
guardian(L)
UB1 Rouge-1: 0.380137 Rouge-2: 0.127083 Rouge-l: 0.213465
UB2 Rouge-1: 0.371915 Rouge-2: 0.168840 Rouge-l: 0.237165
LexRank Rouge-1: 0.217602 Rouge-2: 0.035400 Rouge-l: 0.139641
TextRank Rouge-1: 0.181454 Rouge-2: 0.025821 Rouge-l: 0.107198

bbc(L)
UB1 Rouge-1: 0.350821 Rouge-2: 0.115721 Rouge-l: 0.186813
UB2 Rouge-1: 0.333293 Rouge-2: 0.145361 Rouge-l: 0.194031
LexRank Rouge-1: 0.179593 Rouge-2: 0.024781 Rouge-l: 0.121246
TextRank Rouge-1: 0.137366 Rouge-2: 0.014770 Rouge-l: 0.090767

Standard ROUGE
guardian(L)
UB1 Rouge-1: 0.498439 Rouge-2: 0.216667 Rouge-l: 0.324901 Rouge-SU*: 0.216997
UB2 Rouge-1: 0.469815 Rouge-2: 0.278474 Rouge-l: 0.344528 Rouge-SU*: 0.208485
LexRank Rouge-1: 0.210933 Rouge-2: 0.037603 Rouge-l: 0.131110 Rouge-SU*: 0.046715
TextRank Rouge-1: 0.184086 Rouge-2: 0.029617 Rouge-l: 0.117287 Rouge-SU*: 0.037783
ICSI Rouge-1: 0.257562 Rouge-2: 0.060022 Rouge-l: 0.157313 Rouge-SU*: 0.065799
Luhn Rouge-1: 0.154681 Rouge-2: 0.022884 Rouge-l: 0.100451 Rouge-SU*: 0.027575
'''

sys.path.append('../')

from utils.data_helpers import load_data
from tqdm import tqdm
from myrouge.rouge import get_rouge_score

from summarize.upper_bound import ExtractiveUpperbound
from summarize.sume_wrap import SumeWrap
from summarize.sumy.nlp.tokenizers import Tokenizer
from summarize.sumy.parsers.plaintext import PlaintextParser
from summarize.sumy.summarizers.lsa import LsaSummarizer
from summarize.sumy.summarizers.kl import KLSummarizer
from summarize.sumy.summarizers.luhn import LuhnSummarizer
from summarize.sumy.summarizers.lex_rank import LexRankSummarizer
from summarize.sumy.summarizers.text_rank import TextRankSummarizer
from summarize.sumy.nlp.stemmers import Stemmer
from nltk.corpus import stopwords
import sys

reload(sys)
sys.setdefaultencoding('utf-8')
parser = argparse.ArgumentParser(description='LiveBlogSum Baseline')
parser.add_argument('-corpus', type=str, default='guardian')
parser.add_argument('-path', type=str, default='../data/')
parser.add_argument('-sum_len', type=int, default=1)

args = parser.parse_args()
args.path = args.path + args.corpus + '/test/'


def get_summary_scores(algo, docs, refs, summary_size):
    language = 'english'
    summary = ''
    if algo == 'UB1':
        summarizer = ExtractiveUpperbound(language)
        summary = summarizer(docs, refs, summary_size, ngram_type=1)
    elif algo == 'UB2':
        summarizer = ExtractiveUpperbound(language)
        summary = summarizer(docs, refs, summary_size, ngram_type=2)
    elif algo == 'ICSI':
        summarizer = SumeWrap(language)
        summary = summarizer(docs, summary_size)
    else:
        doc_string = u'\n'.join([u'\n'.join(doc_sents) for doc_sents in docs])
        parser = PlaintextParser.from_string(doc_string, Tokenizer(language))
        stemmer = Stemmer(language)
        if algo == 'LSA':
            summarizer = LsaSummarizer(stemmer)
        if algo == 'KL':
            summarizer = KLSummarizer(stemmer)
        if algo == 'Luhn':
            summarizer = LuhnSummarizer(stemmer)
        if algo == 'LexRank':
            summarizer = LexRankSummarizer(stemmer)
        if algo == 'TextRank':
            summarizer = TextRankSummarizer(stemmer)

        summarizer.stop_words = frozenset(stopwords.words(language))
        summary = summarizer(parser.document, summary_size)
    hyps, refs = map(list, zip(*[[' '.join(summary), ' '.join(model)] for model in refs]))
    hyp = str(hyps[0]).split()
    hyp = ' '.join(hyp[:summary_size])
    ref = str(refs[0])
    score = get_rouge_score(hyp, ref)
    return score['ROUGE-1']['r'], score['ROUGE-2']['r'], score['ROUGE-L']['r'], score['ROUGE-SU*']['r']


if __name__ == '__main__':
    file_names = os.listdir(args.path)
    algos = ['UB1', 'UB2', 'LexRank', 'TextRank', 'ICSI', 'Luhn']
    R1 = {'UB1': .0, 'UB2': .0, 'ICSI': .0, 'LSA': .0, 'KL': .0, 'Luhn': .0, 'LexRank': .0, 'TextRank': .0}
    R2 = {'UB1': .0, 'UB2': .0, 'ICSI': .0, 'LSA': .0, 'KL': .0, 'Luhn': .0, 'LexRank': .0, 'TextRank': .0}
    Rl = {'UB1': .0, 'UB2': .0, 'ICSI': .0, 'LSA': .0, 'KL': .0, 'Luhn': .0, 'LexRank': .0, 'TextRank': .0}
    Rsu = {'UB1': .0, 'UB2': .0, 'ICSI': .0, 'LSA': .0, 'KL': .0, 'Luhn': .0, 'LexRank': .0, 'TextRank': .0}
    for filename in tqdm(file_names):
        data_file = os.path.join(args.path, filename)
        docs, refs = load_data(data_file)
        sum_len = len(' '.join(refs[0]).split(' ')) * args.sum_len
        print('####', filename, '####')
        for algo in algos:
            r1, r2, rl, rsu = get_summary_scores(algo, docs, refs, sum_len)
            print algo, r1, r2, rl, rsu
            R1[algo] += r1
            R2[algo] += r2
            Rl[algo] += rl
            Rsu[algo] += rsu
    print('Final Results')
    for algo in algos:
        R1[algo] /= len(file_names)
        R2[algo] /= len(file_names)
        Rl[algo] /= len(file_names)
        Rsu[algo] /= len(file_names)
        print('%s Rouge-1: %f Rouge-2: %f Rouge-l: %f Rouge-SU*: %f' % (algo, R1[algo], R2[algo], Rl[algo], Rsu[algo]))
