from scipy.optimize import linear_sum_assignment
from collections import defaultdict
from itertools import filterfalse, product, islice
from matplotlib import pyplot as plt
import re
import pandas as pd
import numpy as np


df = pd.read_csv("cassoela2025_11runners.csv")
# df = pd.read_csv("test.csv")
tracks = np.array([c for c in df.columns if "Strecke" in c])
df_subset = df[["Name"] + list(tracks)]
tracks = np.array([re.search(r"\[(.+)\]", t).group(1) for t in tracks])


df_subnp = df_subset.to_numpy()
names = df_subnp[:,0]
n_runners = len(names)
preferences = df_subnp[:,1:]

# degrees = {
#     "Really don't want": 0,
#     "Prefer not": 1,
#     "Can": 2,
#     "Would like": 3,
#     "Would really love": 4,
# }

degrees = {
    "😫": 0,
    "🙁": 1,
    "😐": 2, 
    "🙂": 3,
    "😄": 4,
}

min_val = min(d for d in degrees.values())
max_val = max(d for d in degrees.values())

weights = np.zeros_like(preferences, dtype=np.int8)
for txt, val in degrees.items():
    weights[preferences == txt] = val



any_track_runner = defaultdict(set)
best_matching = linear_sum_assignment(weights, True)
best_score = weights[best_matching].sum()

track_runner = {}

combos = defaultdict(list)

for runner_i, track_i in zip(*best_matching):
    combos[names[runner_i]].append(tracks[track_i])
    any_track_runner[names[runner_i]].add(tracks[track_i])
    any_track_runner[tracks[track_i]].add(names[runner_i])




print("Best score is:", best_score, "which means", best_score/n_runners, "per runner")

print("Max possible score per user")
for n, s in zip(names, np.max(weights, axis=1)):
    print(n,s)

if all(np.max(weights, axis=1) == weights[best_matching]):
    print("The algo picks the best option for all runners")
else:
    print("Algo doesn't pick the best option for some runners. Some runners are fighting for the same track.")

bw = weights[best_matching].reshape(n_runners, 1)
is_weight_best = weights == bw

ixs = np.nonzero(is_weight_best)
ixs = (np.split(ixs[1], np.unique(ixs[0], return_index=True)[1]))[1:]
print(ixs)


def custom_prod(*iterables):
    pools = [tuple(pool) for pool in iterables]

    comb_counter = [0] * len(pools)
    comb_counter_max = [len(pool)-1 for pool in pools]
    pools_order = np.argsort(comb_counter_max)
    comb_counter_max = sorted(comb_counter_max)

    def inc_counter_at_pos(pos):
        while pos >= 0:
            if comb_counter[pos] < comb_counter_max[pos]:
                break
            pos -= 1
        if pos == -1:
            return False
        comb_counter[pos] += 1
        for j in range(pos+1, len(comb_counter)):
            comb_counter[j] = 0
        return True            

    def inc_counter():
        for i, (cc, ccm) in enumerate(zip(comb_counter[::-1], comb_counter_max[::-1])):
            if cc < ccm:
                return inc_counter_at_pos(len(comb_counter) - i - 1)

    incremented = True
    with open("out.txt", "w") as fp:
        while incremented:
            prod = [None] * len(pools)
            for comb_pool_i, pool_i in enumerate(pools_order):
                pool_el_i = comb_counter[comb_pool_i]
                pool = pools[pool_i]
                pool_el = pool[pool_el_i]
                if pool_el in prod:
                    incremented = inc_counter_at_pos(comb_pool_i)
                    break
                prod[pool_i] = pool_el
                # print(comb_counter, [pools[i][j] for i, j in enumerate(comb_counter)], file=fp, flush=True)
            if not None in prod:
                incremented = inc_counter()
                yield prod

print(list(custom_prod((0,1),(0,2))))

print("skibidi")
max_combs = 10000
p = islice(custom_prod(*ixs), max_combs)
i = filterfalse(lambda x: len(x) != len(set(x)), p)
combs = np.array(list(p))

        


df_possibilities = pd.DataFrame(columns=tracks, index=names)
df_possibilities = df_possibilities.fillna("")


for i_n, name in enumerate(names):
    for track in any_track_runner[name]:
        df_possibilities.loc[name, track] = weights[i_n, list(tracks).index(track)]


df_combos = pd.DataFrame(combos)
df_combos.to_csv("combos.csv")

print(f"Found {len(combs)} optimal combinations.")



if "Missing" in df_combos:
    with open("missing.txt", "w") as fp:
        fp.write("\n".join(sorted(df_combos["Missing"].unique())))


df_possibilities.to_csv("possibilities.csv", sep="\t")


