import pickle

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import arviz as az
from sklearn.linear_model import RANSACRegressor
from sklearn.metrics import r2_score

from hbmep.model.utils import Site as site
from hbmep.nn import functional as F

from models import HierarchicalBayesianModel
from constants import (
    DATA_PATH,
    TOML_PATH,
    INFERENCE_FILE,
    NETCODE_FILE,
    BUILD_DIR
)

pd.set_option('display.max_columns', None)

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
DATA_PATH = '/home/mcintosh/Cloud/DataPort/2024-05-23_human_non_inv_inference/proc_2024-05-28/proc_2024-05-28.csv'
df = pd.read_csv(DATA_PATH)
model.build_dir = '/home/mcintosh/Cloud/DataPort/2024-05-23_human_non_inv_inference/proc_2024-05-28/meh'
df, encoder_dict = model.load(df=df)
df['participant_str'] = encoder_dict['participant'].inverse_transform(df['participant'])
df['TMS_RMT'] = df['TMS_RMT'].apply(lambda x: float(x))

#%%
n_chains = model.mcmc_params['num_chains']
shape_for_hdi = [n_chains, int(posterior_samples[site.a].shape[0]/n_chains), posterior_samples[site.a].shape[1], posterior_samples[site.a].shape[2], posterior_samples[site.a].shape[3]]

#%%
a_mea = posterior_samples['a'].mean(axis=0)

response_resort = ['PKPK_Biceps', 'PKPK_ECR', 'PKPK_Triceps', 'PKPK_FCR', 'PKPK_APB', 'PKPK_ADM']

vec_resort = [np.where([i == j for i in model.response])[0][0] for j in response_resort]
ix_biceps = np.where([i == 'PKPK_Biceps' for i in model.response])[0][0]

ix_sci = encoder_dict['participant_condition'].transform(['SCI'])[0]
ix_uni = encoder_dict['participant_condition'].transform(['Uninjured'])[0]

#%%
impute_val = 1e4
s50 = F.get_s50_from_rectified_logistic(posterior_samples[site.a], posterior_samples[site.b], posterior_samples[site.ell], posterior_samples[site.H])
val_threshold = 50e-3
rr_th = F.solve_rectified_logistic(val_threshold, posterior_samples[site.a], posterior_samples[site.b], posterior_samples[site.L], posterior_samples[site.ell], posterior_samples[site.H])
rr_th_impute = np.nan_to_num(rr_th, nan=impute_val)  # !!!! !! ! ! ! ! !! ! ! ! #!###!)$I!)I@)$! !!!!
rr_fraction_imputed = np.mean(np.isfinite(rr_th), axis=0)

s50_impute = np.nan_to_num(s50, nan=impute_val)  # !!!! !! ! ! ! ! !! ! ! ! #!###!)$I!)I@)$! !!!!
s50_fraction_imputed = np.mean(np.isfinite(s50), axis=0)

#%%
th_mean = np.mean(posterior_samples[site.a], axis=0)
th_hdi = az.hdi(posterior_samples[site.a].reshape(shape_for_hdi))  # internally collapses first 2 dims

rr_th_median = np.nanmedian(rr_th_impute, axis=0)
rr_hdi = az.hdi(rr_th_impute.reshape(shape_for_hdi))  # internally collapses first 2 dims
# if nan don't use for computation of mean/median/hdi
for i in range(rr_th.shape[1]):
    for j in range(rr_th.shape[2]):
        for k in range(rr_th.shape[3]):
            data_slice = np.array(posterior_samples[site.a][:, i, j, k])
            if not np.all(np.isnan(data_slice)):
                rr_hdi[i, j, k] = az.hdi(data_slice[~np.isnan(data_slice)])
                rr_th_median[i, j, k] = np.nanmedian(data_slice[~np.isnan(data_slice)])

s50_mean = np.mean(s50_impute, axis=0)
s50_hdi = az.hdi(s50_impute.reshape(shape_for_hdi))  # internally collapses first 2 dims
# if nan don't use for computation of mean/median/hdi
for i in range(s50.shape[1]):
    for j in range(s50.shape[2]):
        for k in range(s50.shape[3]):
            data_slice = np.array(s50[:, i, j, k])
            if not np.all(np.isnan(data_slice)):
                s50_hdi[i, j, k] = az.hdi(data_slice[~np.isnan(data_slice)])
                s50_mean[i, j, k] = np.mean(data_slice[~np.isnan(data_slice)])

#%%
df_reduced = df.groupby(['participant_str', 'ix_visit', 'ix_run', 'target_muscle'])['TMS_RMT'].unique().reset_index()
assert df_reduced['participant_str'].is_unique, "The 'participant_str' column contains duplicate values."

df_reduced['TMS_RMT'] = df_reduced['TMS_RMT'].apply(lambda x: float(x))
df_reduced['rr_th_median'] = np.nan

for index, row in df_reduced.iterrows():
    participant = df_reduced.at[index, 'participant_str']
    target_muscle_unsided = 'PKPK_' + df_reduced.at[index, 'target_muscle'][1:]

    case_target = [i == target_muscle_unsided for i in model.response]

    if np.sum(case_target) == 1:
        ix_target = np.where(case_target)[0][0]
        ix_participant_posterior = encoder_dict['participant'].transform([participant])[0]

        if participant.startswith('SCS'):
            ix_pc = ix_sci
        elif participant.startswith('SCA'):
            ix_pc = ix_uni
        else:
            raise Exception('?')

        df_reduced.at[index, 'rr_th_median'] = rr_th_median[ix_participant_posterior, ix_pc, ix_target]
        df_reduced.at[index, 'rr_hdi_low'] = rr_hdi[ix_participant_posterior, ix_pc, ix_target, 0]
        df_reduced.at[index, 'rr_hdi_high'] = rr_hdi[ix_participant_posterior, ix_pc, ix_target, 1]
        df_reduced.at[index, 'rr_percentage_imputed'] = rr_fraction_imputed[ix_participant_posterior, ix_pc, ix_target] * 100

        df_reduced.at[index, 's50_mean'] = s50_mean[ix_participant_posterior, ix_pc, ix_target]
        df_reduced.at[index, 's50_hdi_low'] = s50_hdi[ix_participant_posterior, ix_pc, ix_target, 0]
        df_reduced.at[index, 's50_hdi_high'] = s50_hdi[ix_participant_posterior, ix_pc, ix_target, 1]
        df_reduced.at[index, 's50_percentage_imputed'] = s50_fraction_imputed[ix_participant_posterior, ix_pc, ix_target] * 100

        df_reduced.at[index, 'th_mea'] = th_mean[ix_participant_posterior, ix_pc, ix_target]
        df_reduced.at[index, 'th_hdi_low'] = th_hdi[ix_participant_posterior, ix_pc, ix_target, 0]
        df_reduced.at[index, 'th_hdi_high'] = th_hdi[ix_participant_posterior, ix_pc, ix_target, 1]



#%%
p = Path('dnc_figures', 'dnc_temp')
df_reduced.to_excel(p.with_suffix('.xlsx'), index=False)
#%%
df_reduced = df_reduced.dropna(subset=['th_mea']).reset_index()

#%%
vec_sci = [False for i in range(a_mea.shape[0])]

plt.figure()
fig, axes = plt.subplots(1, df_reduced.shape[0], figsize=(24, 3), squeeze=False)
g = []
for index, row in df_reduced.iterrows():
    participant = df_reduced.at[index, 'participant_str']
    ax = axes[0, index]

    if participant.startswith('SCS'):
        ix_pc = ix_sci
        # vec_sci[ix] = True
    elif participant.startswith('SCA'):
        ix_pc = ix_uni
        # vec_sci[ix] = False

    x = df_reduced.at[index, 'th_mea']
    xl = df_reduced.at[index, 'th_hdi_low']
    xh = df_reduced.at[index, 'th_hdi_high']
    h = ax.plot(0, x, 'ro', alpha=0.5)
    h = ax.plot([0, 0], [xl, xh], 'r-', alpha=0.5)

    x = df_reduced.at[index, 'rr_th_median']
    xl = df_reduced.at[index, 'rr_hdi_low']
    xh = df_reduced.at[index, 'rr_hdi_high']
    h = ax.plot(1, x, 'go', alpha=0.5)
    h = ax.plot([1, 1], [xl, xh], 'g-', alpha=0.5)
    x_pct = df_reduced.at[index, 'rr_percentage_imputed']
    if x_pct < 99:
        if x > 100:
            x_local = 100
        else:
            x_local = x
        ax.text(1, x_local, '{:.0f}%'.format(100-x_pct), ha='center', va='top')

    x = df_reduced.at[index, 's50_mean']
    xl = df_reduced.at[index, 's50_hdi_low']
    xh = df_reduced.at[index, 's50_hdi_high']
    h = ax.plot(3, x, 'mo', alpha=0.5)
    h = ax.plot([3, 3], [xl, xh], 'm-', alpha=0.5)
    x_pct = df_reduced.at[index, 's50_percentage_imputed']
    if x_pct < 99:
        if x > 100:
            x_local = 100
        else:
            x_local = x
        ax.text(3, x_local, '{:.0f}%'.format(100-x_pct), ha='center', va='top')

    x = df_reduced.at[index, 'TMS_RMT']
    h = ax.plot(2, x, 'bo', alpha=0.5)

    ax.text(1.5, 0, df_reduced.at[index, 'target_muscle'][1:], ha='center', va='bottom')

    ax.set_xlim([-0.5, 3.5])
    ax.set_title(participant)
    ax.set_xticks([])
    if index > 0:
        ax.set_yticks([])
    else:
        ax.set_ylabel('Intensity (%MSO)')

    # if participant.startswith('SCS'):
    #     h[0].set_color('red')
    # elif participant.startswith('SCA'):
    #     h[0].set_color('black')
    #     g.append(a_mea[ix, ix_pc, vec_resort])
    # ax.set_ylabel('Threshold (%MSO)')

# boo = a_mea[vec_sci, ix_sci, :]
# boo_mea = np.mean(boo, axis=0)
# boo_sem = np.std(boo, axis=0) / np.sqrt(boo.shape[0])
# ax.plot(boo_mea[vec_resort], 'm', linewidth=4)

# ax.set_xticks(range(0, len(response_resort)))
# ax.set_xticklabels(response_resort)
    ax.set_ylim([0, 100])

fig.show()

p = Path('dnc_figures', 'dnc_thresholds')
fig.savefig(p.with_suffix('.png'), dpi=300, bbox_inches='tight')
fig.savefig(p.with_suffix('.svg'), bbox_inches='tight')
#%%
columns_to_plot = ['TMS_RMT', 'rr_th_median', 'th_mea']


fig, ax = plt.subplots(figsize=(4, 8))

# Plot the gray lines linking individual entries
for participant in df_reduced['participant_str'].unique():
    participant_data = df_reduced[df_reduced['participant_str'] == participant]
    if participant.startswith('SCS'):
        c = 'r'
    elif participant.startswith('SCA'):
        c = 'k'
    ax.plot(
        [1, 2, 3],
        [participant_data['TMS_RMT'].values[0], participant_data['rr_th_median'].values[0], participant_data['th_mea'].values[0]],
        color=c, linestyle='-', linewidth=0.5, alpha=0.5
    )

# Plot the candlestick plots
positions = [1, 2, 3]
box_data = [df_reduced['TMS_RMT'].dropna(), df_reduced['rr_th_median'].dropna(), df_reduced['th_mea'].dropna()]
ax.boxplot(box_data, positions=positions, widths=0.6)

# Set x-axis labels
ax.set_xticks(positions)
ax.set_xticklabels(columns_to_plot)

# Set plot title and labels
ax.set_ylabel('Intensity (%MSO)')
ax.set_ylim([0, 100])

# Adjust layout and show the plot
plt.tight_layout()
plt.show()

p = Path('dnc_figures', 'dnc_summary')
fig.savefig(p.with_suffix('.png'), dpi=300, bbox_inches='tight')
fig.savefig(p.with_suffix('.svg'), bbox_inches='tight')

#%%
# Create a lower triangular pair plot
def plot_with_fit(ax, x, y, xerr=None, yerr=None, xlabel='', ylabel='', title=''):
    ax.scatter(x, y)
    c1, c2 = 50, 100
    if xerr is not None and yerr is not None:
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o')
        valid_mask = (abs(xerr[1] - xerr[0]) <= c1) & (abs(yerr[1] - yerr[0]) <= c1) & (x < c2) & (y < c2)
        ax.scatter(x[np.invert(valid_mask)], y[np.invert(valid_mask)], color='red', s=50, zorder=3)
        x = x[valid_mask]
        y = y[valid_mask]
    elif yerr is not None:
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o')
        valid_mask = (abs(yerr[1] - yerr[0]) < c1) & (x < c2) & (y < c2)
        ax.scatter(x[np.invert(valid_mask)], y[np.invert(valid_mask)], color='red', s=50, zorder=3)
        x = x[valid_mask]
        y = y[valid_mask]
    elif xerr is not None:
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o')
        valid_mask = (abs(xerr[1] - xerr[0]) < c1) & (x < c2) & (y < c2)
        ax.scatter(x[np.invert(valid_mask)], y[np.invert(valid_mask)], color='red', s=50, zorder=3)
        x = x[valid_mask]
        y = y[valid_mask]

    else:
        ax.errorbar(x, y, fmt='o')

    # Add dashed x=y line
    # lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
    lims = [0, 100]
    ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)

    # Robust linear fit
    model = RANSACRegressor()
    model.fit(x.values.reshape(-1, 1), y.values)

    line_x = np.linspace(0, 100, 100)
    line_y = model.predict(line_x.reshape(-1, 1))
    ax.plot(line_x, line_y, color='red')

    # Calculate R2
    y_pred = model.predict(x.values.reshape(-1, 1))
    r2 = r2_score(y, y_pred)
    slope = model.estimator_.coef_[0]
    intercept = model.estimator_.intercept_
    ax.text(0.05, 0.85, f'Intercept = {intercept:.2f}', transform=ax.transAxes, verticalalignment='top')
    ax.text(0.05, 0.90, f'Slope = {slope:.2f}', transform=ax.transAxes, verticalalignment='top')
    ax.text(0.05, 0.95, f'R² = {r2:.2f}', transform=ax.transAxes, verticalalignment='top')


    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])


# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 14))

# Plot TMS_RMT vs rr_th_median
plot_with_fit(axes[0, 0], df_reduced['TMS_RMT'], df_reduced['rr_th_median'],
              yerr=[df_reduced['rr_th_median'] - df_reduced['rr_hdi_low'],
                    df_reduced['rr_hdi_high'] - df_reduced['rr_th_median']],
              xlabel='TMS_RMT', ylabel='rr_th_median', title='TMS_RMT vs rr_th_median')

# Plot TMS_RMT vs th_mea
plot_with_fit(axes[1, 0], df_reduced['TMS_RMT'], df_reduced['th_mea'],
              yerr=[df_reduced['th_mea'] - df_reduced['th_hdi_low'],
                    df_reduced['th_hdi_high'] - df_reduced['th_mea']],
              xlabel='TMS_RMT', ylabel='th_mea', title='TMS_RMT vs th_mea')

# Plot rr_th_median vs th_mea
plot_with_fit(axes[1, 1], df_reduced['rr_th_median'], df_reduced['th_mea'],
              xerr=[df_reduced['rr_th_median'] - df_reduced['rr_hdi_low'],
                    df_reduced['rr_hdi_high'] - df_reduced['rr_th_median']],
              yerr=[df_reduced['th_mea'] - df_reduced['th_hdi_low'],
                    df_reduced['th_hdi_high'] - df_reduced['th_mea']],
              xlabel='rr_th_median', ylabel='th_mea', title='rr_th_median vs th_mea')

# Hide the unused subplot
axes[0, 1].axis('off')

plt.tight_layout()
plt.show()
p = Path('dnc_figures', 'dnc_scatter')
fig.savefig(p.with_suffix('.png'), dpi=300, bbox_inches='tight')
fig.savefig(p.with_suffix('.svg'), bbox_inches='tight')

#%%
fig, axes = plt.subplots(1, 2, figsize=(14, 7), squeeze=False)

# Extract columns for the thresholds
thresholds = df_reduced[['TMS_RMT', 'rr_th_median', 'th_mea']]

# Calculate standard deviation across participants for each method
std_across_participants = thresholds.std(axis=0)

# Calculate standard deviation across methods for each participant
std_across_methods = thresholds.std(axis=1)

# Plot standard deviation across participants for each method
plt.figure(figsize=(14, 7))

ax = axes[0][0]
ax.bar(std_across_participants.index, std_across_participants.values)
ax.set_title('Standard Deviation Across Participants for Each Method')
ax.set_xlabel('Methods')
ax.set_ylabel('Standard Deviation')
ax.set_ylim([0, 20])

# Plot standard deviation across methods for each participant
ax = axes[0][1]
ax.bar(thresholds.index, std_across_methods.values)
ax.set_title('Standard Deviation Across Methods for Each Participant')
ax.set_xlabel('Participants')
ax.set_ylabel('Standard Deviation')
ax.set_ylim([0, 20])

plt.tight_layout()
plt.show()
p = Path('dnc_figures', 'dnc_sd_vs_sd')
fig.savefig(p.with_suffix('.png'), dpi=300, bbox_inches='tight')
fig.savefig(p.with_suffix('.svg'), bbox_inches='tight')

#%%
print('done')
