from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import re
import pandas as pd
import numpy as np


df = pd.read_csv("cassoela - Form.csv")
tracks = np.array([c for c in df.columns if "Strecke" in c])
df_subset = df[["Name"] + list(tracks)]
tracks = np.array([re.search(r"\[(.+)\]", t).group(1) for t in tracks])


df_subnp = df_subset.to_numpy()
names = df_subnp[:,0]
preferences = df_subnp[:,1:]

degrees = {
    "Really don't want": 0,
    "Prefer not": 1,
    "Can": 2,
    "Would like": 3,
    "Would really love": 4,
}

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




for run_i, tr_i in zip(*best_matching):
    new_weights = weights.copy()
    new_weights[run_i, tr_i] = 0

    while True:
        track_runner = {}
        matching = linear_sum_assignment(new_weights, True)
        score = new_weights[matching].sum() 
        if score < best_score:
            break

        old_score = score

        for runner_i, track_i in zip(*matching):
            combos[names[runner_i]].append(tracks[track_i])
            any_track_runner[names[runner_i]].add(tracks[track_i])
            any_track_runner[tracks[track_i]].add(names[runner_i])


        tr_i_ = matching[1][run_i]
        
        new_weights[run_i, tr_i_] = 0
        


df_possibilities = pd.DataFrame(columns=tracks, index=names)
df_possibilities = df_possibilities.fillna("")


for i_n, name in enumerate(names):
    for track in any_track_runner[name]:
        df_possibilities.loc[name, track] = weights[i_n, list(tracks).index(track)]


df_combos = pd.DataFrame(combos)
df_combos.to_csv("combos.csv")

print(f"Found {len(df_combos)} optimal combinations.")



if "Missing" in df_combos:
    with open("missing.txt", "w") as fp:
        fp.write("\n".join(sorted(df_combos["Missing"].unique())))


df_possibilities.to_csv("possibilities.csv", sep="\t")


