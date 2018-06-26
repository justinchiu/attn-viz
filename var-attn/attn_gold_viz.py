#!/usr/bin/env python
# coding: utf-8
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pickle

data = pickle.load(open('attn_gold_vae_sample.pkl', 'rb'))
softdata = pickle.load(open("attn_gold_soft.pkl", "rb"))

fontsize = 17.5
def plot_hm(source, targets, p_attn, q_attn, fn):
    ones = np.ones(p_attn.shape)
    zeros = np.zeros(p_attn.shape)
    plt.grid(True)
    fig, ax = plt.subplots(1, 1)
    #im_p = np.dstack([p_attn ** -2, 1-p_attn, 1-p_attn, ones])
    #im_p = np.dstack([p_attn**-2, 1-p_attn, 1-p_attn, np.ones(p_attn.shape)])
    #pa = 1-(p_attn**1.25)*3
    pa = 1-p_attn*3 # good
    qa = 1-q_attn*2 # good
    im_p = np.dstack([ones, pa, pa, np.ones(p_attn.shape)]) # good
    #im_p = np.dstack([pa, pa, pa, np.ones(p_attn.shape)])
    ax.imshow(im_p, interpolation="none")
    im_q = np.dstack([qa, qa, ones, q_attn]) # good
    #im_q = np.dstack([1-q_attn, 1-q_attn, q_attn ** -2, q_attn])
    #im_q = np.dstack([1-q_attn, 1-q_attn, q_attn, q_attn])
    ax.imshow(im_q, interpolation="none")

    ax.xaxis.tick_top()
    ax.yaxis.tick_left()

    ax.figure.set_size_inches(8*float(len(targets))/len(source),8)

    plt.xticks(
        np.arange(len(targets)), targets, rotation=35, ha="left",
        fontsize=fontsize)
    plt.yticks(np.arange(len(source)), source, rotation=0, fontsize=fontsize)

    ax.set_xticks(np.arange(-.5, len(targets), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(source), 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.75)

    plt.savefig(fn, dpi=300, bbox_inches='tight')
    #plt.show()

s_attns = softdata[-1]
Hs = []
for i, attn in enumerate(s_attns):
    Hs.append(np.trace(np.log(attn).T @ attn))
Hs = np.array(Hs)
long_hs = []

pq_norms = []
sq_norms = []

for i, (src, tgt, p_attn, q_attn) in enumerate(zip(data[0], data[1], data[2], data[3])):
    src = src.replace('&apos;t', "'t").split()
    tgt = tgt.replace('&apos;t', "'t").split()
    #src = src.replace('&apos;t', "'t").replace("@@", "").split()
    #tgt = tgt.replace('&apos;t', "'t").replace("@@", "").split()
    norm = np.linalg.norm(p_attn-q_attn)
    pq_norms.append((i, norm))
    #if len(src) > 10 and len(src) < 15 and len(tgt) > 10 and len(tgt) < 15 and norm > 2.5:
    #if len(src) > 10 and len(src) < 15 and len(tgt) > 10 and len(tgt) < 15 and norm > 1.5:
    if i in [2448, 1652, 1760, 1943, 1997, 2001, 2451, 2361, 2166, 2199, 2531]:
        plot_hm(src[:-1], tgt[:-2], p_attn.T[:-1,:-2], q_attn.T[:-1,:-2], str(i) + 'p-q-attn' + '.png')
    #import pdb; pdb.set_trace()
    #if i == 2531:
        #plot_hm(src[:-1], tgt[:-2], p_attn.T[:-1,:-2], q_attn.T[:-1,:-2], str(i) + 'p-q-attn' + '.png')

    norm = np.linalg.norm(s_attns[i] - q_attn)
    sq_norms.append((i, norm))

    #if len(src) >= 10 and len(src) <= 15 and len(tgt) >= 10 and len(tgt) <= 15 and norm > 3:
    #if len(src) > 10 and len(src) < 15 and len(tgt) > 10 and len(tgt) < 15 and Hs[i] > -7:
        #long_hs.append((i, Hs[i]))
    if False and 2500 < i and i < 2515:
        plot_hm(src[:-1], tgt[:-2], s_attns[i].T[:-1,:-2], q_attn.T[:-1,:-2], str(i) + 's-q-attn' + '.png')


#print(sorted(long_hs, key=lambda x: x[1], reverse=True)[:10])
mixture_data = pickle.load(open("softdiff.np", "rb"))
src = mixture_data["src"][:-1]
tgt = mixture_data["tgt"][:-1]
p_attn = mixture_data["p_mean"].squeeze().T[:-1,:-1]
q_attn = mixture_data["q_mean"].squeeze().T[:-1,:-1]
#plot_hm(src, tgt, p_attn, q_attn, "2531p-q-attn-mixture.png")

def plot_hm3(source, targets, p1_, p2_, p3_, fn):
    ones = np.ones(p2_.shape)
    zeros = np.zeros(p2_.shape)
    plt.grid(True)
    fig, ax = plt.subplots(1, 1)

    """
    p1_[p1_ > 0.015] += 0.05
    p2_[p2_ > 0.015] += 0.05
    p3_[p3_ > 0.015] += 0.01
    p1 = 1-p1_*4 # good
    p2 = 1-p2_*4 # good
    p3 = 1-p3_*3 # good

    im_p1 = np.dstack([ones, p1, p1, np.ones(p1.shape)]) # good
    #ax.imshow(im_p1, interpolation="none")
    im_p3 = np.dstack([p3, ones, p3, ones]) # good
    ax.imshow(im_p3, interpolation="none")
    im_p2 = np.dstack([ones, p2, p2, p2_ * 0.8]) # good
    ax.imshow(im_p2, interpolation="none")
    """
    #p1_[p1_ > 0.015] += 0.05
    p2_[p2_ > 0.015] += 0.05
    p3_[p3_ > 0.015] += 0.15
    p3_[p3_ <= 0.015] = 0
    #p1 = 1-p1_*4 # good
    p2 = 1-p2_*4 # good
    #p3 = 1-p3_*3 # good
    p3 = 1-p3_*3 # good

    #im_p1 = np.dstack([ones, p1, p1, np.ones(p1.shape)]) # good
    #ax.imshow(im_p1, interpolation="none")
    im_p3 = np.dstack([p3, ones, p3, ones]) # good
    ax.imshow(p3_, interpolation="none", cmap=plt.get_cmap("Greens"))
    im_p2 = np.dstack([ones, p2, p2, p2_ * 0.8]) # good
    ax.imshow(im_p2, interpolation="none")
    #"""

    ax.xaxis.tick_top()
    ax.yaxis.tick_left()

    ax.figure.set_size_inches(8*float(len(targets))/len(source),8)

    plt.xticks(
        np.arange(len(targets)), targets, rotation=35, ha="left",
        fontsize=fontsize)
    plt.yticks(np.arange(len(source)), source, rotation=0, fontsize=fontsize)

    ax.set_xticks(np.arange(-.5, len(targets), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(source), 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.75)

    plt.savefig(fn, dpi=300, bbox_inches='tight')

mixture_data = pickle.load(open("pmix.pkl", "rb"))
src = mixture_data["src"][:-1]
tgt = mixture_data["tgt"][:-1]

p_mix = mixture_data["p_mean"].squeeze().T[:-1,:-2]
q_mix = mixture_data["q_mean"].squeeze().T[:-1,:-1]
norms = []
for i in range(len(data[0])):
    [src1, tgt1, p_vae, q_vae] = list(zip(*data))[i]
    [src2, tgt2, p_soft, _] = list(zip(*softdata))[i]
    src = src1.split()[:-1]
    tgt = tgt1.split()[:-2]
    q_mix -= q_mix
    q_vae -= q_vae
    tgt = [x.replace('&apos;d', "'d").replace("&apos;t", "'t") for x in tgt]

    norm = np.linalg.norm(p_vae - p_soft)
    norms.append(norm)
    # R B G
    if len(src) > 10 and len(src) < 15 and len(tgt) > 10 and len(tgt) < 15 and norm > 2:
    #if i in [2448, 1652, 1760, 1943, 1997, 2001, 2451, 2361, 2166, 2199, 2531]:
        plot_hm3(src, tgt, None, p_vae.T[:-1,:-2], p_soft.T[:-1,:-2],
            str(i) + 'svattn' + '.png')
#import pdb; pdb.set_trace()
