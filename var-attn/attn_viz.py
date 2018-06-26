#!/usr/bin/env python
# coding: utf-8
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pickle
soft_attn = pickle.load(open('soft-attn.out.pkl.pkl', 'rb'))
vae_attn =  pickle.load(open('vae-sample-attn.out.pkl.pkl', 'rb'))

def plot_hm(source, targets, data, fn):
    sns.set(font_scale=1.5)
    ax = sns.heatmap(
        data, xticklabels = targets, yticklabels = source, cmap="Blues",
        robust=True, cbar=False, linewidth=.5,
        vmin=0.0, vmax=1.0)
    ax.xaxis.tick_top()
    ax.yaxis.tick_left()
    ax.figure.set_size_inches(8*float(len(targets))/len(source),8)
    plt.xticks(rotation=30, ha="left")
    plt.yticks(rotation=0)
    plt.savefig(fn, dpi=300, bbox_inches='tight')

for i, (a1, a2) in enumerate(zip(soft_attn, vae_attn)):
    src = a1[0].replace('&apos;t', "'t").split()
    tgt1 = a1[1].replace('&apos;t', "'t").split()
    tgt2 = a2[1].replace('&apos;t', "'t").split()
    attn1 = a1[-1]
    attn2 = a2[-1]
    gold = a1[2]
    if i == 2154 or i == 2418 or i == 4737:
        #2154 is case where soft gets it right and vae gets it wrong
        #2418 is the case where vae gets it right and soft gets it wrong
        #4737 is the case where both get it right
        #here "right" means 100% accuracy wrt to gold
        plot_hm(src, tgt1, attn1.T, str(i) + 'attn-soft' + '.png')
        plot_hm(src, tgt2, attn2.T, str(i) + 'attn-vae' + '.png')

