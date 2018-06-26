import torch
from torch.distributions import Bernoulli as B
from torch.distributions import Categorical as C

import argparse

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument("--theta", default=0.1)
    args.add_argument("--src_vocab_size", default=500)
    args.add_argument("--n_train", default=10000)
    args.add_argument("--n_valid", default=500)
    args.add_argument("--out_prefix", default="/n/rush_lab/data/iwslt14-de-en/data/toy/")
    return args.parse_args()

args = get_args()

theta = args.theta
src_vocab_size = args.src_vocab_size
n_train = args.n_train
n_valid = args.n_valid
n_samples = n_train + n_valid

train_srcfile = args.out_prefix + "train.src"
train_tgtfile = args.out_prefix + "train.tgt"
valid_srcfile = args.out_prefix + "valid.src"
valid_tgtfile = args.out_prefix + "valid.tgt"

attn_mixing_d = B(theta)
word_d = C(torch.Tensor([1] * src_vocab_size))

# randomly draw 2 source words
# Flip coin attn_mixing_d, if 0 then copy first token, if 1 then copy second token with 0.5 prob

src_words = word_d.sample(torch.Size([n_samples, 2]))
mix = attn_mixing_d.sample(torch.Size([n_samples]))
mix[mix == 1] = 0.5

copy_idx = B(mix).sample().long()
print("The mean of the copy distribution: {}".format(copy_idx.float().mean()))

tgt_words = src_words.gather(1, copy_idx.unsqueeze(-1))

with open(train_srcfile, "w") as srcf, open(train_tgtfile, "w") as tgtf:
    src_string = "\n".join([" ".join(str(w) for w in s) for s in src_words[:n_train].tolist()])
    tgt_string = "\n".join(str(w) for w in tgt_words[:n_train].squeeze().tolist())
    srcf.write(src_string)
    tgtf.write(tgt_string)
with open(valid_srcfile, "w") as srcf, open(valid_tgtfile, "w") as tgtf:
    src_string = "\n".join([" ".join(str(w) for w in s) for s in src_words[n_train:].tolist()])
    tgt_string = "\n".join(str(w) for w in tgt_words[n_train:].squeeze().tolist())
    srcf.write(src_string)
    tgtf.write(tgt_string)
