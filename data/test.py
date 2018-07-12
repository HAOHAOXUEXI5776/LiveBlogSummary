from myrouge.rouge import get_rouge_score
from rouge import Rouge
import re


def rouge_1_f(hyp, ref):
    hyp = re.sub(r'[^a-z]', ' ', hyp.lower()).strip().split()
    ref = re.sub(r'[^a-z]', ' ', ref.lower()).strip().split()
    ref_flag = [0 for _ in ref]
    hit = .0
    for w in hyp:
        for i in range(0, len(ref)):
            if w == ref[i] and ref_flag[i] == 0:
                hit += 1
                ref_flag[i] = 1
                break
    p = hit / len(hyp)
    r = hit / len(ref)
    f = 2 * p * r / (p + r)
    print(p, r, f)
    return f


if __name__ == '__main__':
    a = 'nice one cyrille'
    b = u'nice one cyrille nice one son'
    r = Rouge()
    s1 = r.get_scores(a, b)[0]
    s2 = get_rouge_score(a, b)
    print(s1)
    print(s2)
    print(rouge_1_f(a, b))
