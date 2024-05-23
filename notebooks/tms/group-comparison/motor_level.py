import pickle

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.cm as cm
from pathlib import Path


from hbmep.model.utils import Site as site

from models import HierarchicalBayesianModel
from constants import (
    DATA_PATH,
    TOML_PATH,
    INFERENCE_FILE,
    NETCODE_FILE,
    BUILD_DIR
)

plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["svg.fonttype"] = "none"

##
M = HierarchicalBayesianModel

# Load posterior
dest = r'/home/mcintosh/Cloud/DataPort/2024-05-23_human_non_inv_inference/inference.pkl'
with open(dest, "rb") as f:
    model, mcmc, posterior_samples_ = pickle.load(f)

posterior_samples = posterior_samples_.copy()
posterior_samples[site.outlier_prob] = 0 * posterior_samples[site.outlier_prob]

# Load data
DATA_PATH = '/home/mcintosh/Cloud/DataPort/2024-05-23_human_non_inv_inference/proc_2024-05-23/proc_2024-05-23.csv'
df = pd.read_csv(DATA_PATH)
model.build_dir = '/home/mcintosh/Cloud/DataPort/2024-05-23_human_non_inv_inference/proc_2024-05-23/meh'
df, encoder_dict = model.load(df=df)
df['participant_str'] = encoder_dict['participant'].inverse_transform(df['participant'])

#%%
a_mea = posterior_samples['a'].mean(axis=0)

response_resort = ['PKPK_Biceps', 'PKPK_ECR', 'PKPK_Triceps', 'PKPK_FCR', 'PKPK_APB', 'PKPK_ADM']

vec_resort = [np.where([i == j for i in model.response])[0][0] for j in response_resort]
ix_biceps = np.where([i == 'PKPK_Biceps' for i in model.response])[0][0]

ix_sci = encoder_dict['participant_condition'].transform(['SCI'])[0]
ix_uni = encoder_dict['participant_condition'].transform(['Uninjured'])[0]

#%%
vec_sci = [False for i in range(a_mea.shape[0])]

plt.figure()
fig, axes = plt.subplots(1, 1, figsize=(8, 8), squeeze=False)
ax = axes[0, 0]
g = []
for ix in range(a_mea.shape[0]):

    participant = encoder_dict['participant'].inverse_transform([ix])[0]
    if participant.startswith('SCS'):
        ix_pc = ix_sci
        vec_sci[ix] = True
    elif participant.startswith('SCA'):
        ix_pc = ix_uni
        vec_sci[ix] = False

    h = ax.plot(a_mea[ix, ix_pc, vec_resort], alpha = 0.5)
    if participant.startswith('SCS'):
        h[0].set_color('red')
    elif participant.startswith('SCA'):
        h[0].set_color('black')
        g.append(a_mea[ix, ix_pc, vec_resort])
    ax.set_ylabel('Threshold (%MSO)')

boo = a_mea[vec_sci, ix_sci, :]
boo_mea = np.mean(boo, axis=0)
boo_sem = np.std(boo, axis=0) / np.sqrt(boo.shape[0])
ax.plot(boo_mea[vec_resort], 'm', linewidth=4)
ax.fill_between(range(len(vec_resort)), boo_mea[vec_resort] - boo_sem[vec_resort], boo_mea[vec_resort] + boo_sem[vec_resort], color='m', alpha=0.3)

# Calculate and plot mean and SEM for non-vec_sci group
boo = a_mea[np.invert(vec_sci), ix_uni, :]
boo_mea = np.mean(boo, axis=0)
boo_sem = np.std(boo, axis=0) / np.sqrt(boo.shape[0])
ax.plot(boo_mea[vec_resort], 'b', linewidth=4)
ax.fill_between(range(len(vec_resort)), boo_mea[vec_resort] - boo_sem[vec_resort], boo_mea[vec_resort] + boo_sem[vec_resort], color='b', alpha=0.3)


ax.set_xticks(range(0, len(response_resort)))
ax.set_xticklabels(response_resort)
fig.show()

p = Path('dnc_figures', 'dnc_uninjured_vs_SCI')
fig.savefig(p.with_suffix('.png'), dpi=300, bbox_inches='tight')
fig.savefig(p.with_suffix('.svg'), bbox_inches='tight')

#%%
vec_sci = [False for i in range(a_mea.shape[0])]

plt.figure()
fig, axes = plt.subplots(1, 1, figsize=(8, 8), squeeze=False)
ax = axes[0, 0]
for ix in range(a_mea.shape[0]):

    participant = encoder_dict['participant'].inverse_transform([ix])[0]
    if participant.startswith('SCS'):
        ix_pc = ix_sci
        vec_sci[ix] = True
    elif participant.startswith('SCA'):
        ix_pc = ix_uni

    o = 0
    o = a_mea[ix, ix_pc, ix_biceps]
    h = ax.plot(a_mea[ix, ix_pc, vec_resort]/o)
    if participant.startswith('SCS'):
        h[0].set_color('red')
    elif participant.startswith('SCA'):
        h[0].set_color('black')
    ax.set_ylabel('% Threshold change relative to Biceps')

boo = a_mea[vec_sci, ix_sci, :] / a_mea[vec_sci, ix_sci, ix_biceps].reshape(-1, 1)
boo_mea = np.mean(boo, axis=0)
boo_sem = np.std(boo, axis=0) / np.sqrt(boo.shape[0])
ax.plot(boo_mea[vec_resort], 'm', linewidth=4)
ax.fill_between(range(len(vec_resort)), boo_mea[vec_resort] - boo_sem[vec_resort], boo_mea[vec_resort] + boo_sem[vec_resort], color='m', alpha=0.3)

# Calculate and plot mean and SEM for non-vec_sci group
boo = a_mea[np.invert(vec_sci), ix_uni, :] / a_mea[np.invert(vec_sci), ix_uni, ix_biceps].reshape(-1, 1)
boo_mea = np.mean(boo, axis=0)
boo_sem = np.std(boo, axis=0) / np.sqrt(boo.shape[0])
ax.plot(boo_mea[vec_resort], 'b', linewidth=4)
ax.fill_between(range(len(vec_resort)), boo_mea[vec_resort] - boo_sem[vec_resort], boo_mea[vec_resort] + boo_sem[vec_resort], color='b', alpha=0.3)


ax.set_xticks(range(0, len(response_resort)))
ax.set_xticklabels(response_resort)
fig.show()
p = Path('dnc_figures', 'dnc_uninjured_vs_SCI_norm')
fig.savefig(p.with_suffix('.png'), dpi=300, bbox_inches='tight')
fig.savefig(p.with_suffix('.svg'), bbox_inches='tight')

#%%
plt.figure()
fig, axes = plt.subplots(1, 7, figsize=(30, 4), squeeze=False)

cmap = cm.get_cmap('viridis', 7)
colors = cmap(np.linspace(0, 1, 7))

for ix in range(a_mea.shape[0]):
    participant = encoder_dict['participant'].inverse_transform([ix])[0]


    if participant.startswith('SCS'):
        ix_pc = ix_sci

    str_seg = (df['MotorLevel'][df['participant_str'] == participant]).unique()[0]
    if type(str_seg) == str:
        ix_seg = int(str_seg[1])
    elif np.isnan(str_seg):
        continue

    ax = axes[0, ix_seg-2]

    o = a_mea[ix, ix_pc, ix_biceps]
    # o = 0
    y = a_mea[ix, ix_pc, vec_resort]/o
    h = ax.plot(y)
    h[0].set_color(colors[ix_seg-2])
    ax.text(5, y[-1], str_seg)

    ax.set_xticks(range(0, len(response_resort)))
    ax.set_xticklabels(response_resort)

    print(str_seg, ix_seg-2)
    if ix_seg-2 == 0:
        ax.set_ylabel('% Threshold change relative to Biceps')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylim([0.5, 1.5])
    ax.set_title(f'Motor level: {str_seg}')
fig.show()
p = Path('dnc_figures', 'dnc_motor_level')
fig.savefig(p.with_suffix('.png'), dpi=300, bbox_inches='tight')
fig.savefig(p.with_suffix('.svg'), bbox_inches='tight')

#%%
plt.figure()
fig, axes = plt.subplots(1, 4, figsize=(20, 4), squeeze=False)

cmap = cm.get_cmap('viridis', 4)
colors = cmap(np.linspace(0, 1, 4))

for ix in range(a_mea.shape[0]):
    participant = encoder_dict['participant'].inverse_transform([ix])[0]


    if participant.startswith('SCS'):
        ix_pc = ix_sci
        vec_sci[ix] = True

    str_seg = (df['NLI'][df['participant_str'] == participant]).unique()[0]
    if type(str_seg) == str:
        ix_seg = int(str_seg[1])
    elif np.isnan(str_seg):
        continue

    ax = axes[0, ix_seg-2]

    o = a_mea[ix, ix_pc, ix_biceps]
    # o = 0
    y = a_mea[ix, ix_pc, vec_resort]/o
    h = ax.plot(y)
    h[0].set_color(colors[ix_seg-2])
    ax.text(5, y[-1], str_seg)

    ax.set_xticks(range(0, len(response_resort)))
    ax.set_xticklabels(response_resort)

    print(str_seg, ix_seg-2)
    if ix_seg-2 == 0:
        ax.set_ylabel('% Threshold change relative to Biceps')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylim([0.5, 1.5])
    ax.set_title(f'NLI: {str_seg}')

fig.show()
p = Path('dnc_figures', 'dnc_nli')
fig.savefig(p.with_suffix('.png'), dpi=300, bbox_inches='tight')
fig.savefig(p.with_suffix('.svg'), bbox_inches='tight')

#%%
plt.figure()
fig, axes = plt.subplots(1, 4, figsize=(20, 4), squeeze=False)

cmap = cm.get_cmap('viridis', 4)
colors = cmap(np.linspace(0, 1, 4))

for ix in range(a_mea.shape[0]):
    participant = encoder_dict['participant'].inverse_transform([ix])[0]


    if participant.startswith('SCS'):
        ix_pc = ix_sci

    str_seg = (df['AIS'][df['participant_str'] == participant]).unique()[0]
    if type(str_seg) == str:
        ix_seg = 'ABCDE'.index(str_seg)
    elif np.isnan(str_seg):
        continue

    ax = axes[0, ix_seg]

    o = a_mea[ix, ix_pc, ix_biceps]
    # o = 0
    y = a_mea[ix, ix_pc, vec_resort]/o
    h = ax.plot(y)
    h[0].set_color(colors[ix_seg])
    ax.text(5, y[-1], str_seg)

    ax.set_xticks(range(0, len(response_resort)))
    ax.set_xticklabels(response_resort)

    print(str_seg, ix_seg-2)
    if ix_seg-2 == 0:
        ax.set_ylabel('% Threshold change relative to Biceps')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylim([0.5, 1.5])
    ax.set_title(f'AIS: {str_seg}')
fig.show()
p = Path('dnc_figures', 'dnc_ais')
fig.savefig(p.with_suffix('.png'), dpi=300, bbox_inches='tight')
fig.savefig(p.with_suffix('.svg'), bbox_inches='tight')

#%%
def select_grassp_arm(row, str_base='GRASSP_ARM_'):
    if row['target_muscle'].startswith('L'):
        return row[f'{str_base}_L']
    elif row['target_muscle'].startswith('R'):
        return row[f'{str_base}_R']
    else:
        return None

# Apply the function to each row to create the new column
df['GRASSP_ARM'] = df.apply(select_grassp_arm, axis=1, args=('GRASSP_ARM',))
df['UEMS_ARM'] = df.apply(select_grassp_arm, axis=1, args=('UEMS_ARM',))

#%%
x = ['AIS', 'GRASSP_ARM','MotorLevel', 'NLI', 'UEMS_ARM', 'activity_level', 'age', 'cueq', 'handedness', 'height', 'sex']

#%%
df_reduced = df.groupby(['participant_str', 'participant_condition', 'ix_visit', 'ix_run', 'target_muscle']).first()

plt.figure()
fig, axes = plt.subplots(3, 4, figsize=(12, 8), squeeze=False)  #
axes = axes.ravel()
for ix in range(len(x)):
    s = x[ix]
    ax = axes[ix]
    if type(df_reduced[s][-1]) == str:
        c = df_reduced[s].dropna().value_counts().sort_index()
        c.plot.bar(ax=ax, title=s, grid=True, alpha=0.75)
    else:
        if s == 'height':
            bin_edges = np.arange(100, 200, 10)
        elif s == 'cueq':
            bin_edges = np.arange(-150, 150, 10)
        else:
            bin_edges = np.arange(0, 110, 10)
        c = df_reduced[s].dropna()
        c.groupby(['participant_condition']).plot.hist(ax=ax, bins=bin_edges, title=s, grid=True, alpha=0.75)
    print(c)
    ax.set_xlabel(s)
fig.subplots_adjust(wspace=0.4, hspace=1.0)
plt.show()
p = Path('dnc_figures', 'dnc_summary_demographics')
fig.savefig(p.with_suffix('.png'), dpi=300, bbox_inches='tight')
fig.savefig(p.with_suffix('.svg'), bbox_inches='tight')

