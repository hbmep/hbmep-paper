import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from models import RectifiedLogistic
from hbmep.model.utils import Site as site

from hbmep import functional as F
from hbmep import smooth_functional as S

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier

EPS = 1e-3


def main():
    # Update this path to the path in /media folder
    src = "/home/mcintosh/Cloud/DataPort/2024-11-07_posterior_for_ni_data_default_model/rectified_logistic/inference.pkl"
    src_nm = "/home/mcintosh/Cloud/DataPort/2024-11-07_posterior_for_ni_data_default_model/least_squares/params.pkl"
    dest = "/home/mcintosh/Cloud/DataPort/2024-11-07_posterior_for_ni_data_default_model/rectified_logistic_report/vis.svg"
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

    # The first 14 subjects are uninjured
    BREAK = 14
    just_use_threshold = True

    use_hb = False
    if use_hb:
        p1_mean = np.nanmean(posterior_samples[site.a], axis=0)
        p2_mean = np.nanmean(posterior_samples[site.H], axis=0)
        p3_mean = np.nanmean(posterior_samples[site.b], axis=0)
        p4_mean = np.nanmean(posterior_samples[site.L], axis=0)
        p5_mean = max_grad.mean(axis=0)

    else:
        # APB, ADM, ECR, FCR, so swap ADM and APB (and later only plot first two muscles - since the HB posterior has triceps etc. etc.)
        p1_mean = np.nanmean(params_nm[site.a], axis=1)
        p2_mean = np.nanmean(params_nm[site.H], axis=1)
        p3_mean = np.nanmean(params_nm[site.b], axis=1)
        p4_mean = np.nanmean(params_nm[site.L], axis=1)
        p1_mean[:, [0, 1]] = p1_mean[:, [1, 0]]
        p2_mean[:, [0, 1]] = p2_mean[:, [1, 0]]
        p3_mean[:, [0, 1]] = p3_mean[:, [1, 0]]
        p4_mean[:, [0, 1]] = p4_mean[:, [1, 0]]


    # Prepare the target vector y_target
    y_target = np.zeros(p1_mean.shape[0], dtype=int)
    y_target[BREAK:] = 1  # 0 for uninjured, 1 for SCI

    clf = LinearSVC(dual=False, max_iter=10000)

    nrows, ncols = 1, model.n_response
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, 1 * 4), squeeze=False, constrained_layout=True)

    for response_ind, response in enumerate(model.response[:2]):
        ax = axes[0, response_ind]

        if just_use_threshold:
            X_feat = np.column_stack([
                p1_mean[:, response_ind],
            ])
        else:
            X_feat = np.column_stack([
                p1_mean[:, response_ind],
                p2_mean[:, response_ind],
                p3_mean[:, response_ind],
                # p4_mean[:, response_ind],
                # p5_mean[:, response_ind],
            ])

        loo = LeaveOneOut()
        scores = cross_val_score(clf, X_feat, y_target, cv=loo)
        mean_accuracy = scores.mean()

        # For plot
        clf.fit(X_feat, y_target)
        weights = clf.coef_[0]

        # Project data
        w = weights / np.linalg.norm(weights)
        X_projected = X_feat @ w

        sns.kdeplot(x=X_projected[y_target == 0], ax=ax, label='Uninjured', shade=True, color='b')
        sns.kdeplot(x=X_projected[y_target == 1], ax=ax, label='SCI', shade=True, color='r')

        # Decision threshold - not sure about this but looks ok...
        threshold = -clf.intercept_[0] / np.linalg.norm(weights)
        ax.axvline(threshold, color='k', linestyle='--')
        str_acc = f"Accuracy for {response}: {mean_accuracy:.2f}"
        ax.set_title(str_acc)
        ax.set_xlabel('Projection')
        ax.legend()


    fig.savefig(dest)
    print(f"Saved to {dest}")
    return

if __name__ == '__main__':
    main()
