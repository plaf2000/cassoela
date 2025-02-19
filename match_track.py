from scipy.optimize import linear_sum_assignment
from itertools import islice
from matplotlib import pyplot as plt
import re
import pandas as pd
import numpy as np


def product_without_repetitions(*iterables):
    """
    Efficiently calculate products between pools that do not contain repetitions.
    """
    pools = [tuple(pool) for pool in iterables]

    # The counter is used to give an ordering to the products, so that the products can be returned one by one.
    prod_counter = [0] * len(pools)
    prod_counter_max = [len(pool)-1 for pool in pools]

    # Pools with most elements need to be in the least significant part of the counter for efficiency reason.
    pools_order = np.argsort(prod_counter_max)
    prod_counter_max = sorted(prod_counter_max)

    def inc_counter_at_pos(pos):
        """
        Increment the counter at position `pos` by one. If it already reached the max, try to increment more significant part.
        If also that fails, return False (impossible to increment, max value reached).
        """
        while pos >= 0:
            if prod_counter[pos] < prod_counter_max[pos]:
                break
            pos -= 1
        if pos == -1:
            return False
        prod_counter[pos] += 1
        for j in range(pos+1, len(prod_counter)):
            prod_counter[j] = 0
        return True            

    def inc_counter():
        """
        Increment counter by one (smallest unit).
        """
        for i, (cc, ccm) in enumerate(zip(prod_counter[::-1], prod_counter_max[::-1])):
            if cc < ccm:
                return inc_counter_at_pos(len(prod_counter) - i - 1)

    incremented = True
    while incremented:
        prod = [None] * len(pools)
        for prod_pool_i, pool_i in enumerate(pools_order):
            pool_el_i = prod_counter[prod_pool_i]
            pool = pools[pool_i]
            pool_el = pool[pool_el_i]
            if pool_el in prod:
                # Value of the pool already contained in the product, try the next one from the pool.
                incremented = inc_counter_at_pos(prod_pool_i)
                break
            prod[pool_i] = pool_el
        if not None in prod:
            # Try next product and return the found one.
            incremented = inc_counter()
            yield prod

COMBINATIONS_LIMIT = 1000
SHOW = True

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

best_matching = linear_sum_assignment(weights, True)
best_score = weights[best_matching].sum()

print("Best score is:", best_score, "which means", best_score/n_runners, "per runner")

print("Max possible score per user")
for n, s in zip(names, np.max(weights, axis=1)):
    print(n,s)

if all(np.max(weights, axis=1) == weights[best_matching]):
    print("The algo picks the best option for all runners.")
else:
    print("Algo doesn't pick the best option for some runners. Some runners are fighting for the same track.")

best_weights = weights[best_matching].reshape(n_runners, 1)
is_weight_best = weights == best_weights

is_favourite = np.nonzero(is_weight_best)
favourites = (np.split(is_favourite[1], np.unique(is_favourite[0], return_index=True)[1]))[1:]

p = islice(product_without_repetitions(*favourites), COMBINATIONS_LIMIT)
combos = np.array(list(p))
tot = len(combos)
print(f"Found {tot} optimal combinations.")

tracks_i, tracks_count = np.unique(combos.flatten(), return_counts=True)

fig, ax = plt.subplots()
ax.bar(tracks_i + 1, tracks_count, color="black")
ax.set_xlabel("Track number")
ax.set_ylabel("Number of combinations")
ax.set_title("Number of combinations in which the track is present")
if SHOW:
    plt.show()
fig.tight_layout()
fig.savefig("num_combos.png")

df_combos = pd.DataFrame(combos+1)
df_combos = df_combos.rename(columns = {i: names[i] for i in df_combos.columns})
df_combos.to_csv("combos.csv")

heatmap = np.zeros((n_runners, len(tracks)))

for i, name in enumerate(names):
    tracks_i, tracks_count = np.unique(combos[:, i], return_counts=True)
    heatmap[i, tracks_i] = tracks_count

fig, ax = plt.subplots()
im = ax.imshow(heatmap, cmap="binary")
cbar = ax.figure.colorbar(im, ax=ax)
ax.set_yticks(range(n_runners), names)
ax.set_xticks(range(len(tracks)), [str(i+1) for i in range(len(tracks))])
ax.set_xlabel("Track number")
ax.set_ylabel("Runner")
ax.set_title("Combinations Heatmap")

if SHOW:
    plt.show()
fig.savefig("combos_heatmap.png")




