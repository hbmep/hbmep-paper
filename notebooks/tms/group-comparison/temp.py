axis_label_size = 10
subdural_bgcolor = (.7, .7, .7, .2)

figure = plt.figure(figsize=(12, 12), constrained_layout=True)

(topfig, midfig, bottomfig) = figure.subfigures(3, 1, height_ratios=[1, 1, .5])

fig = topfig
topaxes = fig.subplots(2, 3, squeeze=False)

fig = midfig
midaxes = fig.subplots(2, 3, squeeze=False)
fig.set_facecolor(subdural_bgcolor)

fig = bottomfig
bottomaxes = fig.subplots(1, 4, squeeze=False)

allfigs = [topfig, midfig, bottomfig]
allaxes = [topaxes, midaxes, bottomaxes]

topaxes[0, 0].set_title("Epidural", fontweight="bold", loc="left")
midaxes[0, 0].set_title("Subdural", fontweight="bold", loc="left")

""" Epidural and Subdural"""
for depth in [0, 1]:
    fig = allfigs[depth]
    axes = allaxes[depth]

    for muscle_ind in range(model.n_response):
        for side in [0, 1]:
            c = (side, depth, 0)
            ind = df[model.features].apply(tuple, axis=1).isin([c])
            temp_df = df[ind].reset_index(drop=True).copy()

            pred_ind = prediction_df[model.features].apply(tuple, axis=1).isin([c])
            temp_pred_df = prediction_df[pred_ind].reset_index(drop=True).copy()
            temp_mu = mu[:, pred_ind, muscle_ind].mean(axis=0)
            temp_obs = obs[:, pred_ind, muscle_ind].mean(axis=0)
            temp_obs_hpdi = obs_hpdi[:, pred_ind, muscle_ind]

            ax = axes[0, muscle_ind]
            sns.scatterplot(x=temp_df[model.intensity], y=temp_df[model.response[muscle_ind]], color=response_colors[side][muscle_ind], ax=ax)
            sns.lineplot(x=temp_pred_df[model.intensity], y=temp_mu, color=response_colors[side][muscle_ind], ax=ax)
            # ax.axvline(x=a_adj[*c[::-1], muscle_ind], color=response_colors[side][muscle_ind], linestyle="--")
            ax.fill_between(
                temp_pred_df[model.intensity],
                temp_obs_hpdi[0, :],
                temp_obs_hpdi[1, :],
                color=response_colors[side][muscle_ind],
                alpha=.15
            )
            ax.sharex(topaxes[0, 0])
            ax.sharey(topaxes[0, 0])

            ax = axes[1, muscle_ind]
            sns.kdeplot(a[:, *c[::-1], muscle_ind] + a_kde_offset[*c[::-1], muscle_ind], color=response_colors[side][muscle_ind], ax=ax)
            # ax.axvline(x=a_adj[*c[::-1], muscle_ind], color=response_colors[side][muscle_ind], linestyle="--")
            ax.set_xlim(right=8)
            ax.sharex(axes[0, 0])
            ax.sharey(axes[1, 0])

for axes in allaxes:
    nrows, ncols = axes.shape
    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i, j]
            sides = ['right', 'top']
            for side in sides:
                ax.spines[side].set_visible(False)
            ax.tick_params(
                axis='both',
                which='both',
                left=True,
                bottom=True,
                right=False,
                top=False,
                labelleft=True,
                labelbottom=True,
                labelright=False,
                labeltop=False,
                labelrotation=15
            )
            ax.set_xlabel("")
            ax.set_ylabel("")

for depth in [0, 1]:
    axes = allaxes[depth]
    ax = axes[0, 0]
    ax.set_xticks([0, 4, 8])
    ax.set_yticks([0, 15, 30])
    ax.set_ylabel("AUC (µV$\cdot$s)", size=axis_label_size)

    ax = axes[1, 0]
    ax.set_xticks([0, 4, 8])
    ax.set_ylabel("Prob. Density", size=axis_label_size)

    for muscle_ind in range(model.n_response):
        for side in [0, 1]:
            ax = axes[0, muscle_ind]
            ax.set_xlabel("Stimulation Intensity (mA)", size=axis_label_size)

            ax = axes[1, muscle_ind]
            ax.set_xlabel("Threshold (mA)", size=axis_label_size)
            ax.tick_params(
                axis='both',
                which='both',
                left=False,
                labelleft=False
            )

ax = topaxes[0, 0]
ax.set_ylim(top=35)

axes = bottomaxes
for muscle_ind in range(model.n_response):

    for depth, depth_name in [(0, "Epidural"), (1, "Subdural")]:
        ax = axes[0, depth]
        samples = a[:, 0, depth, 1, muscle_ind] - a[:, 0, depth, 0, muscle_ind]
        sns.kdeplot(samples, ax=ax, color=response_colors[1][muscle_ind])
        prob = (samples > 0).mean()
        # ax.set_title(f"Pr( Midline > Lateral | {depth_name} ) = {prob:.2f}", fontweight="bold")
        ax.set_xlabel(f"Midline > Lateral | {depth_name}", fontweight="bold")

    for side, side_name in [(0, "Lateral"), (1, "Midline")]:
        ax = axes[0, 2 + side]
        samples = a[:, 0, 0, side, muscle_ind] - a[:, 0, 1, side, muscle_ind]
        sns.kdeplot(samples, ax=ax, color=response_colors[1][muscle_ind])
        prob = (samples > 0).mean()
        # ax.set_title(f"Pr( Epidural > Subdural | {side_name} ) = {prob:.2f}", fontweight="bold")
        ax.set_xlabel(f"Epidural > Subdural | {side_name}", fontweight="bold")

ax = axes[0, 0]
ax.set_xticks([0, 3, 6])

ax = axes[0, 1]
ax.set_xlim(left=-3, right=3)
ax.set_xticks([-2, 0, 2])

ax = axes[0, 2]
ax.set_xticks([-1, 0, 1])

ax = axes[0, 3]
ax.set_xlim(left=-6, right=6)
ax.set_xticks([-5, 0, 5])

for j in range(4):
    ax = axes[0, j]
    ax.tick_params(
        axis='both',
        which='both',
        left=False,
        labelleft=False
    )
    ax.set_ylabel("")

for fig in allfigs:
    fig.align_xlabels()
    fig.align_ylabels()

# dest = os.path.join(model.build_dir, "subdural-epidural.svg")
# figure.savefig(dest, dpi=600)
