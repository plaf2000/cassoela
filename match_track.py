from scipy.optimize import linear_sum_assignment
from collections import defaultdict

import pandas as pd
import numpy as np


df = pd.read_csv("cassoela - Form.csv")
tracks = np.array([c for c in df.columns if "Strecke" in c])
df_subset = df[["Name"] + list(tracks)]
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

for runner_i, track_i in zip(*best_matching):
    track_runner[names[runner_i]] = tracks[track_i]
    track_runner[tracks[track_i]] = names[runner_i]
    any_track_runner[names[runner_i]].add(tracks[track_i])
    any_track_runner[tracks[track_i]].add(names[runner_i])

print("========== Best combo ============")
for n in names:
    print(f"{n}:", track_runner[n])
print()


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
            track_runner[names[runner_i]] = tracks[track_i]
            track_runner[tracks[track_i]] = names[runner_i]
            any_track_runner[names[runner_i]].add(tracks[track_i])
            any_track_runner[tracks[track_i]].add(names[runner_i])

        print("========== Other possible combo ============")
        for n in names:
            print(f"{n}:", track_runner[n])
        print()


        tr_i_ = matching[1][run_i]
        
        new_weights[run_i, tr_i_] = 0
        

print(dict(any_track_runner))

df_possibilities = pd.DataFrame(columns=tracks, index=names)
df_possibilities = df_possibilities.fillna("")



# for name in names:
#     df_possibilities.loc[name, list(any_track_runner[name])] = "x"

for i_n, name in enumerate(names):
    for track in any_track_runner[name]:
        df_possibilities.loc[name, track] = weights[i_n, list(tracks).index(track)]
        

    


df_possibilities.to_csv("possibilities.csv", sep="\t")
print(df_possibilities)


