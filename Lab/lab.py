from asyncio.streams import FlowControlMixin
import pdb
import nltk
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import math
from nltk.corpus import brown
from nltk.tokenize import RegexpTokenizer
from gensim.models import KeyedVectors
import tqdm
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
nltk.download('brown')

def load_corpus():
    words = brown.words()
    # remove punctuations
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(' '.join(words))
    # convert to lower case
    clean_words = []
    for w in words:
        clean_words.append(w.lower())
    # count frequency
    fdict = nltk.FreqDist(w for w in clean_words)
    # top 5000 words
    flist = []
    for k, v in fdict.items():
        flist.append([k, v])
    flist = pd.DataFrame(flist, columns=['word', 'freq'])
    top = flist.sort_values(by=['freq'], ascending=False)[:5000]
    return clean_words, top, fdict

def load_rg65():
    rg65 = [['cord', 'smile', 0.02], ['hill', 'woodland', 1.48],
            ['rooster', 'voyage', 0.04], ['car', 'journey', 1.55],
            ['noon', 'string', 0.04], ['cemetery', 'mound', 1.69],
            ['fruit', 'furnace', 0.05], ['glass', 'jewel', 1.78],
            ['autograph', 'shore', 0.06], ['magician', 'oracle', 1.82],
            ['automobile', 'wizard', 0.11], ['crane', 'implement', 2.37], 
            ['mound', 'stove', 0.14], ['brother', 'lad', 2.41],
            ['grin', 'implement', 0.18], ['sage', 'wizard', 2.46],
            ['asylum', 'fruit', 0.19], ['oracle', 'sage', 2.61],
            ['asylum', 'monk', 0.39], ['bird', 'crane', 2.63],
            ['graveyard', 'madhouse', 0.42], ['bird', 'cock', 2.63],
            ['glass', 'magician', 0.44], ['food', 'fruit', 2.69],
            ['boy', 'rooster', 0.44], ['brother', 'monk', 2.74],
            ['cushion', 'jewel', 0.45], ['asylum', 'madhouse', 3.04],
            ['monk', 'slave', 0.57], ['furnace', 'stove', 3.11],
            ['asylum', 'cemetery', 0.79], ['magician', 'wizard', 3.21],
            ['coast', 'forest', 0.85], ['hill', 'mound', 3.29], 
            ['grin', 'lad', 0.88], ['cord', 'string', 3.41], 
            ['shore', 'woodland', 0.90], ['glass', 'tumbler', 3.45],
            ['monk', 'oracle', 0.91], ['grin', 'smile', 3.46],
            ['boy', 'sage', 0.96], ['serf', 'slave', 3.46],
            ['automobile', 'cushion', 0.97], ['journey', 'voyage', 3.58], 
            ['mound', 'shore', 0.97], ['autograph', 'signature', 3.59],
            ['lad', 'wizard', 0.99], ['coast', 'shore', 3.60],
            ['forest', 'graveyard', 1.00], ['forest', 'woodland', 3.65],
            ['food', 'rooster', 1.09], ['implement', 'tool', 3.66],
            ['cemetery', 'woodland', 1.18], ['cock', 'rooster', 3.68], 
            ['shore', 'voyage', 1.22], ['boy', 'lad', 3.82],
            ['bird', 'woodland', 1.26], ['cushion', 'pillow', 3.84],
            ['coast', 'hill', 1.26], ['cemetery', 'graveyard', 3.88], 
            ['furnace', 'implement', 1.37], ['automobile', 'car', 3.92],
            ['crane', 'rooster', 1.41], ['midday', 'noon', 3.94], 
            ['gem', 'jewel', 3.94]]
    words = {}
    for line in rg65:
        try:
            words[line[0]] += 1
        except:
            words[line[0]] = 1
        try:
            words[line[1]] += 1
        except:
            words[line[1]] = 1
    return rg65, words

def get_freq():
    corpus, top_words, freq_unigrams = load_corpus()
    rg65, rg65_words = load_rg65()
    all_words = list(set(top_words['word'].tolist()) | set(rg65_words.keys()))
    all_words_dict = {k: v for v, k in enumerate(all_words)}
    bigrams_corpus = nltk.bigrams(corpus)
    freq_bigrams = nltk.FreqDist(bigrams_corpus)
    return freq_unigrams, freq_bigrams, all_words_dict, rg65

def get_M1(freq_bigrams, all_words_dict):
    M1 = np.zeros([len(all_words_dict), len(all_words_dict)], dtype=int)
    for k, v in freq_bigrams.items():
        try:
            M1[all_words_dict[k[0]]][all_words_dict[k[1]]] = v
        except:
            pass
    return M1

def get_M1_plus(freq_bigrams, freq_unigrams, all_words_dict):
    M1_plus = np.zeros([len(all_words_dict), len(all_words_dict)], dtype=float)
    all_bigrams = sum(freq_bigrams.values())
    all_unigrams = sum(freq_unigrams.values())
    for k, v in freq_bigrams.items():
        pmi = math.log((v * all_unigrams * all_unigrams) / (freq_unigrams[k[0]] * freq_unigrams[k[1]] * all_bigrams), 2)
        ppmi = max(pmi, 0)
        try:
            M1_plus[all_words_dict[k[0]]][all_words_dict[k[1]]] = ppmi
        except:
            pass
    return M1_plus

def get_pca(k, M1_plus):
    pca = PCA(n_components=k)
    M2 = pca.fit_transform(M1_plus)
    return M2

def get_PS(rg65, all_words_dict):
    P = []
    S = []
    for line in rg65:
        if line[0] in all_words_dict.keys() and line[1] in all_words_dict.keys():
            P.append((line[0], line[1]))
            S.append(line[-1])
    return P, S

def get_cossim(M, P, all_words_dict):
    sim = []
    for p in P:
        left = M[all_words_dict[p[0]]]
        right = M[all_words_dict[p[1]]]
        if np.dot(left, right) == 0:
            sim.append(0)
        else:
            sim.append(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))
    return sim

def get_sim_w2v(P, model):
    sim = []
    for l, r in P:
        left = model[l]
        right = model[r]
        if np.dot(left, right) == 0:
            sim.append(0)
        else:
            sim.append(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))
    return sim

def load_anology(path, all_words_dict):
    all_words_list = all_words_dict.keys()
    semantic = []
    syntactic = []
    with open(path) as f:
        for line in f:
            if line[0] in 'abcdefghijklmnopqrstuvwxyz':
                line = line.strip().split(' ')
                if line[0] in all_words_list and line[1] in all_words_list and line[2] in all_words_list and line[3] in all_words_list:
                    syntactic.append(line)
            elif line[0] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                line = line.strip().split(' ')
                if line[0].lower() in all_words_list and line[1].lower() in all_words_list and line[2].lower() in all_words_list and line[3].lower() in all_words_list:
                    semantic.append([line[0].lower(), line[1].lower(), line[2].lower(), line[3].lower()])
            else:
                pass
    print(f'#Semantic: {len(semantic)}, #Syntactic: {len(syntactic)}')
    return semantic, syntactic

def anology_w2v(model, data, all_words_dict):
    all_candidates = list(set(model.index_to_key) & set(all_words_dict.keys()))
    acc = 0
    for line in tqdm.tqdm(data):
        left = model[line[0]] - model[line[1]] + model[line[3]]
        sim = []
        can = []
        for candidate in all_candidates:
            if candidate not in [line[0], line[1], line[3]]:
                right =  model[candidate]
                sim.append(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))
                can.append(candidate)
        iftrue = can[np.array(sim).argmax()] == line[2]
        acc += iftrue
    return acc / len(data)

def anology_M2_300(model, M, data, all_words_dict):
    M_dict = {}
    for k, v in all_words_dict.items():
        M_dict[k] = M[v]
    all_candidates = list(set(model.index_to_key) & set(all_words_dict.keys()))
    acc = 0
    for line in tqdm.tqdm(data):
        left = M_dict[line[0]] - M_dict[line[1]] + M_dict[line[3]]
        sim = []
        can = []
        for candidate in all_candidates:
            if candidate not in [line[0], line[1], line[3]]:
                right =  M_dict[candidate]
                sim.append(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))
                can.append(candidate)
        iftrue = can[np.array(sim).argmax()] == line[2]
        acc += iftrue
    return acc / len(data)

def load_emb(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        words = data['w']
        embs = data['E']
    w2e_dict = {}
    for i, w in enumerate(words):
        w2e_dict[w] = embs[i]
    return w2e_dict

def measuring_change(w2e_dict, measure, until):
    ret = []
    for w in w2e_dict.keys():
        embs = w2e_dict[w]
        if measure == 'l1':
            ret.append(np.linalg.norm(embs[until] - embs[0], 1))
        elif measure == 'l2':
            ret.append(np.linalg.norm(embs[until] - embs[0], 2))
        elif measure == 'cos':
            if np.dot(embs[until], embs[0]) == 0:
                ret.append(0)
            else:
                ret.append(- np.dot(embs[until], embs[0])/ (np.linalg.norm(embs[until]) * np.linalg.norm(embs[0])))
        else:
            raise ValueError
    ret = np.array(ret)
    most_ids = np.argsort(ret)[-20:]
    for i in most_ids:
        print(list(w2e_dict.keys())[i])
    print(ret[most_ids])
    least_ids = np.argsort(ret)[:20]
    for i in least_ids:
        print(list(w2e_dict.keys())[i])
    print(ret[least_ids])
    return ret



if __name__ == '__main__':
    freq_unigrams, freq_bigrams, all_words_dict, rg65 = get_freq()
    M1 = get_M1(freq_bigrams, all_words_dict)
    M1_plus = get_M1_plus(freq_bigrams, freq_unigrams, all_words_dict)
    M2_10 = get_pca(10, M1_plus)
    M2_100 = get_pca(100, M1_plus)
    M2_300 = get_pca(300, M1_plus)
    P, S = get_PS(rg65, all_words_dict)
    sim_M1 = get_cossim(M1, P, all_words_dict)
    sim_M1_plus = get_cossim(M1_plus, P, all_words_dict)
    sim_M2_10 = get_cossim(M2_10, P, all_words_dict)
    sim_M2_100 = get_cossim(M2_100, P, all_words_dict)
    sim_M2_300 = get_cossim(M2_300, P, all_words_dict)

    for sim in [sim_M1, sim_M1_plus, sim_M2_10, sim_M2_100, sim_M2_300]:
        print(np.corrcoef(S, sim))
    
    model = KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
    sim_w2v = get_sim_w2v(P, model)
    print(np.corrcoef(S, sim_w2v))
    
    semantic, syntactic = load_anology('./word-test.v1.txt', all_words_dict)
    acc_semantic_w2v = anology_w2v(model, semantic, all_words_dict)
    acc_syntactic_w2v = anology_w2v(model, syntactic, all_words_dict)
    print(acc_semantic_w2v, acc_syntactic_w2v)
    acc_semantic_M2_300 = anology_M2_300(model, M2_300, semantic, all_words_dict)
    acc_syntactic_M2_300 = anology_M2_300(model, M2_300, syntactic, all_words_dict)
    print(acc_semantic_M2_300, acc_syntactic_M2_300)
    
    w2e_dict = load_emb('./embeddings/data.pkl')
    dis_l1 = measuring_change(w2e_dict, measure='l1', until=5)
    print('-----------')
    dis_l2 = measuring_change(w2e_dict, measure='l2', until=5)
    print('-----------')
    dis_cos = measuring_change(w2e_dict, measure='cos', until=5)
    pearson = np.corrcoef(np.array([dis_l1, dis_l2, dis_cos]))
    print(pearson)
    
    l1l2 = np.correlate(dis_l1, dis_l2)
    l1cos = np.correlate(dis_l1, dis_cos)
    l2cos = np.correlate(dis_l2, dis_cos)
    print(l1l2, l1cos, l2cos)
    
    dis_l1_truth = measuring_change(w2e_dict, measure='l1', until=-1)
    print('-----------')
    dis_l2_truth = measuring_change(w2e_dict, measure='l2', until=-1)
    print('-----------')
    dis_cos_truth = measuring_change(w2e_dict, measure='cos', until=-1)
    print(np.corrcoef(dis_l1, dis_l1_truth))
    print(np.corrcoef(dis_l2, dis_l2_truth))
    print(np.corrcoef(dis_cos, dis_cos_truth))
    
    words = ['film', 'shift', 'berkeley']
    ret = []
    for i in range(1, 10):
        ret_line = []
        for w in words:
            ret_line.append(- np.dot(w2e_dict[w][i], w2e_dict[w][i-1])/ (np.linalg.norm(w2e_dict[w][i]) * np.linalg.norm(w2e_dict[w][i-1])))
        ret.append(ret_line)
    ret = pd.DataFrame(ret, columns=['film', 'shift', 'berkeley'])
    sns.lineplot(data=ret)
    plt.savefig('./between.pdf')
    ret = []
    for i in range(1, 10):
        ret_line = []
        for w in words:
            ret_line.append(- np.dot(w2e_dict[w][i], w2e_dict[w][i-1])/ (np.linalg.norm(w2e_dict[w][i]) * np.linalg.norm(w2e_dict[w][i-1])))
        ret.append(ret_line)
    ret = pd.DataFrame(ret, columns=['film', 'shift', 'berkeley'])
    sns.lineplot(data=ret)
    plt.savefig('./until.pdf')
    # pdb.set_trace()
    
    