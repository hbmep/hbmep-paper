import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from models import RectifiedLogistic
from hbmep.model.utils import Site as site

from hbmep import functional as F
from hbmep import smooth_functional as S

EPS = 1e-3


def main():
    # Update this path to the path in /media folder
    src = "/home/mcintosh/Cloud/DataPort/2024-11-07_posterior_for_ni_data_default_model/rectified_logistic/inference.pkl"
    src_nm = "/home/mcintosh/Cloud/DataPort/2024-11-07_posterior_for_ni_data_default_model/least_squares/params.pkl"
    dest = "/home/mcintosh/Cloud/DataPort/2024-11-07_posterior_for_ni_data_default_model/rectified_logistic_report/vis.png"
    # src = "/home/vishu/repos/hbmep-paper/reports/cross-validation/tms/rectified_logistic/inference.pkl"
    # dest = "/home/vishu/vis.png"

    with open(src_nm, "rb") as f: params_nm, = pickle.load(f)
    with open(src, 'rb') as f:
        model, mcmc, posterior_samples = pickle.load(f)

    n_post = 500
    NUM_POINTS = 250
    named_params = [site.a, site.b, site.L, site.ell, site.H]
    params = [posterior_samples[site] for site in named_params]
    params = [param[..., None, :] for param in params]
    params = [np.concatenate(NUM_POINTS * [param], axis=-2) for param in params]

    params_red = [params[i][:n_post, ...] for i in range(len(params))]

    intensity = np.linspace(0, 100, NUM_POINTS)
    intensity = intensity[None, None, :, None]
    intensity = np.broadcast_to(intensity, params_red[0].shape)

    grad = F.prime(S.rectified_logistic, intensity, *params_red, EPS * np.ones_like(intensity))
    max_grad = grad.max(axis=-2)

    p1 = posterior_samples[site.a][:n_post, ...]
    p2 = posterior_samples[site.H][:n_post, ...]  # debatable if you should add L
    p3 = posterior_samples[site.b][:n_post, ...]  # debatable if you should add L
    p4 = posterior_samples[site.ell][:n_post, ...]  # debatable if you should add L
    p5 = max_grad

    # APB, ADM, ECR, FCR, so swap ADM and APB (and later only plot first two muscles - since the HB posterior has triceps etc. etc.)
    p1_mean_nm = np.nanmean(params_nm[site.a], axis=1)
    p2_mean_nm = np.nanmean(params_nm[site.H], axis=1)
    p3_mean_nm = np.nanmean(params_nm[site.b], axis=1)
    p4_mean_nm = np.nanmean(params_nm[site.ell], axis=1)
    p1_mean_nm[:, [0, 1]] = p1_mean_nm[:, [1, 0]]
    p2_mean_nm[:, [0, 1]] = p2_mean_nm[:, [1, 0]]
    p3_mean_nm[:, [0, 1]] = p3_mean_nm[:, [1, 0]]
    p4_mean_nm[:, [0, 1]] = p4_mean_nm[:, [1, 0]]

    # The first 14 subjects are uninjured
    BREAK = 14

    nrows, ncols = 1, model.n_response
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 6), squeeze=False, constrained_layout=True)

    for response_ind, response in enumerate(model.response[:2]):
        ax = axes[0, response_ind]

        for ix_p in range(p1.shape[1]):
            if ix_p < BREAK:
                c = 'b'
                cnm = 'c'
            else:
                c = 'r'
                cnm = 'm'

            x = p1[..., ix_p, response_ind].reshape(-1)
            y = p2[..., ix_p, response_ind].reshape(-1)
            x_nm = p1_mean_nm[ix_p, response_ind]
            y_nm = p2_mean_nm[ix_p, response_ind]
            sns.kdeplot(x=x, y=y, ax=ax, levels=[0.5], color=c, linewidths=2, alpha=0.3)
            ax.plot(x.mean(), y.mean(), 'o', alpha=.5, color=c)
            ax.plot(x_nm, y_nm, 'o', alpha=.5, color=cnm)
            ax.plot([x.mean(), x_nm], [y.mean(), y_nm], '-', color='black', alpha=0.7)

        ax.set_title(response)

    fig.savefig(dest)
    print(f"Saved to {dest}")
    return

if __name__ == '__main__':
    main()
