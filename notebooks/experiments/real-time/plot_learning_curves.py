import logging
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import re
import os

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = 'sans-serif'

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

logger = logging.getLogger(__name__)
FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def find_consecutive_true(df, eta, T):
    diff_ratio = df.pct_change().abs()
    condition = diff_ratio < eta
    result = np.zeros(len(condition.columns)) * np.nan
    for ix_col, col in enumerate(condition.columns):
        count = 0
        for i in range(1, len(condition)):
            if condition[col].iloc[i]:
                count += 1
                if count == T:
                    result[ix_col] = i + 1
            else:
                count = 0
    return result


def find_consecutive_gt(df, eta, T):
    condition = df < eta
    result = np.zeros(len(condition.columns)) * np.nan
    for ix_col, col in enumerate(condition.columns):
        count = 0
        for i in range(1, len(condition)):
            if condition[col].iloc[i]:
                count += 1
                if count == T:
                    result[ix_col] = i + 1
            else:
                count = 0
    return result


def extract_info_from_paths(csv_list):
    pattern = r'/\w+_(\w+)_muscle.*?ix(\d+)/'
    extracted_info = []
    for path in csv_list:
        match = re.search(pattern, path)
        if match:
            word_before_muscle = match.group(1)
            number_after_ix = int(match.group(2))
            extracted_info.append((word_before_muscle, number_after_ix))
    return extracted_info


if __name__ == "__main__":
    fig_format = 'png'
    overwrite = True
    root_dir = '/home/mcintosh/Cloud/DataPort/2024_active_learning_sims_for_R03/hbmep_sim/build/figures/'

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    csv_list = [
        '/home/mcintosh/Cloud/DataPort/2024_active_learning_sims_for_R03/hbmep_sim/build/test38_N40_triple_muscle_a_ix0062/REC_MT_cond_norm_mean_threshold.csv',
        '/home/mcintosh/Cloud/DataPort/2024_active_learning_sims_for_R03/hbmep_sim/build/test38_N40_ECR_muscle_a_ix0062/REC_MT_cond_norm_mean_threshold.csv',
        '/home/mcintosh/Cloud/DataPort/2024_active_learning_sims_for_R03/hbmep_sim/build/test38_N40_FCR_muscle_a_ix0062/REC_MT_cond_norm_mean_threshold.csv',
        '/home/mcintosh/Cloud/DataPort/2024_active_learning_sims_for_R03/hbmep_sim/build/test38_N40_APB_muscle_a_ix0062/REC_MT_cond_norm_mean_threshold.csv',
        '/home/mcintosh/Cloud/DataPort/2024_active_learning_sims_for_R03/hbmep_sim/build/test38_N40_triple_muscle_a_ix0040/REC_MT_cond_norm_mean_threshold.csv',
        '/home/mcintosh/Cloud/DataPort/2024_active_learning_sims_for_R03/hbmep_sim/build/test38_N40_ECR_muscle_a_ix0040/REC_MT_cond_norm_mean_threshold.csv',
        '/home/mcintosh/Cloud/DataPort/2024_active_learning_sims_for_R03/hbmep_sim/build/test38_N40_FCR_muscle_a_ix0040/REC_MT_cond_norm_mean_threshold.csv',
        '/home/mcintosh/Cloud/DataPort/2024_active_learning_sims_for_R03/hbmep_sim/build/test38_N40_APB_muscle_a_ix0040/REC_MT_cond_norm_mean_threshold.csv',
        '/home/mcintosh/Cloud/DataPort/2024_active_learning_sims_for_R03/hbmep_sim/build/test38_N40_triple_muscle_a_ix0020/REC_MT_cond_norm_mean_threshold.csv',
        '/home/mcintosh/Cloud/DataPort/2024_active_learning_sims_for_R03/hbmep_sim/build/test38_N40_ECR_muscle_a_ix0020/REC_MT_cond_norm_mean_threshold.csv',
        '/home/mcintosh/Cloud/DataPort/2024_active_learning_sims_for_R03/hbmep_sim/build/test38_N40_FCR_muscle_a_ix0020/REC_MT_cond_norm_mean_threshold.csv',
        '/home/mcintosh/Cloud/DataPort/2024_active_learning_sims_for_R03/hbmep_sim/build/test38_N40_APB_muscle_a_ix0020/REC_MT_cond_norm_mean_threshold.csv',
    ]

    for csv in csv_list:
        loaded_df = pd.read_csv(csv)
        csv_info = extract_info_from_paths([csv])
        loaded_df_lower = pd.read_csv(csv.replace('mean_threshold', 'ci_threshold_lower'))
        loaded_df_upper = pd.read_csv(csv.replace('mean_threshold', 'ci_threshold_upper'))
        last_row_vector = loaded_df.iloc[-1].values
        loaded_df = loaded_df.iloc[:-1]

        plt.figure()
        colors = sns.color_palette('colorblind')
        colors_dict = {'APB': (208 / 255, 28 / 255, 138 / 255), 'ECR': colors[0], 'FCR': colors[1]}
        ix_stop = find_consecutive_gt((loaded_df - last_row_vector).abs(), 1.5, 4)

        print(f'rng:{csv_info[0][1]}')
        if csv_info[0][0] == 'triple':
            print(f'Triple: {np.max(ix_stop):0.1f}, {ix_stop}')
        else:
            print(f'{csv_info[0][0]}: {ix_stop[[csv_info[0][0] == x for x in loaded_df.columns.to_list()]][0]:0.1f}')

        for idx, column in enumerate(loaded_df.columns):
            plt.fill_between(loaded_df.index, loaded_df_lower[column], loaded_df_upper[column],
                             color=colors_dict[column], alpha=0.2)
            plt.plot(loaded_df[column], marker='.', linestyle='-', color=colors_dict[column], label=column)
            plt.axhline(y=last_row_vector[idx], color=colors_dict[column], linestyle='--', label=f'GT {column}')
            plt.axvline(x=ix_stop[idx] - 1, color=colors_dict[column], linestyle='--', label=f'GT {column}')

        plt.title(f'{csv_info[0][0]}, rng{csv_info[0][1]}')
        plt.xlabel('Index')
        plt.ylabel('Values')
        plt.legend()
        plt.grid(True)
        plt.ylim([0, 75])
        plt.xlim([0, 40])

        file_name = f"{csv_info[0][0]}_rng{csv_info[0][1]}.png"
        save_path = os.path.join(root_dir, file_name)
        plt.savefig(save_path, format='png', dpi=300)
        plt.close()