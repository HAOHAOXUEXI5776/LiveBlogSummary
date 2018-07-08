import sys
import os
from rouge import Rouge
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
'''

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.misc import mkdirp
from utils.misc import set_logger
from utils.data_helpers import load_data
from tqdm import tqdm

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
parser.add_argument('-path', type=str, default='/Users/liuhui/Desktop/LiveBlogSummary/label_data/')
parser.add_argument('-sum_len', type=int, default=1)

args = parser.parse_args()
args.path = args.path + args.corpus + '_label/valid/'


def get_summary_scores(algo, docs, refs, summary_size, rouge):
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
    score = rouge.get_scores(hyp, ref)[0]
    return score['rouge-1']['r'], score['rouge-2']['r'], score['rouge-l']['r']


if __name__ == '__main__':
    rouge = Rouge()
    file_names = os.listdir(args.path)
    algos = ['UB1', 'UB2', 'LexRank', 'TextRank']
    R1 = {'UB1': .0, 'UB2': .0, 'ICSI': .0, 'LSA': .0, 'KL': .0, 'Luhn': .0, 'LexRank': .0, 'TextRank': .0}
    R2 = {'UB1': .0, 'UB2': .0, 'ICSI': .0, 'LSA': .0, 'KL': .0, 'Luhn': .0, 'LexRank': .0, 'TextRank': .0}
    Rl = {'UB1': .0, 'UB2': .0, 'ICSI': .0, 'LSA': .0, 'KL': .0, 'Luhn': .0, 'LexRank': .0, 'TextRank': .0}
    for filename in tqdm(file_names):
        data_file = os.path.join(args.path, filename)
        docs, refs = load_data(data_file)
        sum_len = len(' '.join(refs[0]).split(' ')) * args.sum_len
        # Luhn means TF*IDF method
        print '####', filename, '####'
        for algo in algos:
            r1, r2, rl = get_summary_scores(algo, docs, refs, sum_len, rouge)
            print algo, r1, r2, rl
            R1[algo] += r1
            R2[algo] += r2
            Rl[algo] += rl
    print('Final Results')
    for algo in algos:
        R1[algo] /= len(file_names)
        R2[algo] /= len(file_names)
        Rl[algo] /= len(file_names)
        print('%s Rouge-1: %f Rouge-2: %f Rouge-l: %f' % (algo, R1[algo], R2[algo], Rl[algo]))
