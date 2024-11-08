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
    dest = "/home/mcintosh/Cloud/DataPort/2024-11-07_posterior_for_ni_data_default_model/rectified_logistic_report/vis.png"
    # src = "/home/vishu/repos/hbmep-paper/reports/cross-validation/tms/rectified_logistic/inference.pkl"
    # dest = "/home/vishu/vis.png"

    with open(src, 'rb') as f:
        model, mcmc, posterior_samples = pickle.load(f)


    NUM_POINTS = 1000
    named_params = [site.a, site.b, site.L, site.ell, site.H]
    params =[posterior_samples[site] for site in named_params]
    params = [param[..., None, :] for param in params]
    params = [np.concatenate(NUM_POINTS * [param], axis=-2) for param in params]

    intensity = np.linspace(0, 100, NUM_POINTS)
    intensity = intensity[None, None, :, None]
    intensity = np.broadcast_to(intensity, params[0].shape)

    grad = F.prime(S.rectified_logistic, intensity, *params, EPS * np.ones_like(intensity))
    max_grad = grad.max(axis=-2)

    p1 = posterior_samples[site.a]
    p2 = posterior_samples[site.b]

    # The first 14 subjects are uninjured
    BREAK = 14
    # x_sca = x[:, :BREAK, :]
    # y_sca = y[:, :BREAK, :]

    # The rest are injured
    # x_sci = x[:, BREAK:, :]
    # y_sci = y[:, BREAK:, :]

    nrows, ncols = 1, model.n_response
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 6), squeeze=False, constrained_layout=True)

    for response_ind, response in enumerate(model.response):
        ax = axes[0, response_ind]

        for ix_p in range(p1.shape[1]):
        # Plot contour for Uninjured group
            if ix_p < BREAK:
                c = 'b'
            else:
                c = 'r'

            x = p1[..., ix_p, response_ind].reshape(-1)[::50]
            y = p2[..., ix_p, response_ind].reshape(-1)[::50]
            sns.kdeplot(x=x, y=y, ax=ax, levels=[0.95], color=c, linewidths=2, alpha=0.3)

            # Plot contour for Injured group
            # x = a_sci[..., response_ind].mean(axis=0).reshape(-1)
            # y = H_sci[..., response_ind].mean(axis=0).reshape(-1)
            # sns.kdeplot(x=x, y=y, ax=ax, levels=3, label="Injured", color="red", fill=True, alpha=0.3)

        ax.set_title(response)
        # ax.legend()

    fig.savefig(dest)
    print(f"Saved to {dest}")
    return

if __name__ == '__main__':
    main()
