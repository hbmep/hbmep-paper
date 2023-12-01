#!/usr/bin/env python
# coding: utf-8

# In[23]:


#get_ipython().run_line_magic('reload_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')

import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from tqdm import tqdm
from hbmep.utils import timing
import mat73
import tomllib


# In[24]:


MUSCLES = ["ADM", "APB", "Biceps", "ECR", "FCR", "Triceps"]
PKPK_MUSCLES = ["PKPK_" + m for m in MUSCLES]
AUC_MUSCLES = ["AUC_" + m for m in MUSCLES]

@timing
def load_non_invasive_data(
    dir: Path,
    subjects: list[str]
):
    df = None

    for subject in tqdm(subjects):
        print(subject)
        subdir = os.path.join(dir, subject)

        fpath = glob.glob(f"{subdir}/*REC-PAIRED_table.csv")[0]
        temp_df = pd.read_csv(fpath)

        fpath = glob.glob(f"{subdir}/*ep_matrix.mat")[0]
        data_dict = mat73.loadmat(fpath)

        temp_mat = data_dict["ep_sliced"]

        fpath = glob.glob(f"{subdir}/*cfg_proc.toml")[0]
        with open(fpath, "rb") as f:
            cfg_proc = tomllib.load(f)

        temp_df["participant"] = subject

        # Rename columns to actual muscle names
        muscles = cfg_proc["st"]["channel"]
        pkpk_muscles_map = {
            f"pkpk_{i + 1}": "PKPK_" + m for i, m in enumerate(muscles)
        }
        auc_muscles_map = {
            f"auc_{i + 1}": "AUC_" + m for i, m in enumerate(muscles)
        }
        temp_df = temp_df.rename(columns=pkpk_muscles_map).copy()
        temp_df = temp_df.rename(columns=auc_muscles_map).copy()

        # Reorder MEP matrix
        temp_mat = temp_mat[..., np.argsort(muscles)]

        assert temp_df["target_muscle"].unique().shape[0] == 1
        side = temp_df["target_muscle"].unique()[0][0]

        pkpk_side_muscles = ["PKPK_" + side + muscle for muscle in MUSCLES]
        temp_df[PKPK_MUSCLES] = temp_df[pkpk_side_muscles]

        auc_side_muscles = ["AUC_" + side + muscle for muscle in MUSCLES]
        temp_df[AUC_MUSCLES] = temp_df[auc_side_muscles]

        side_muscles = [side + muscle for muscle in MUSCLES]

        ind = [i for i, m in enumerate(sorted(muscles)) if m in side_muscles]
        temp_mat = temp_mat[..., ind]

        if df is None:
            df = temp_df.copy()
            mat = temp_mat

            muscles_sorted = sorted(muscles)

            assert len(set(muscles_sorted)) == len(muscles_sorted)
            continue

        assert set(muscles) == set(muscles_sorted)

        df = pd.concat([df, temp_df], ignore_index=True).reset_index(drop=True).copy()
        mat = np.vstack((mat, temp_mat))

    return df, mat, pkpk_muscles_map, auc_muscles_map


# In[25]:


dir = "/mnt/hdd1/va_data_temp/proc/preproc_tables/paired-rec-visit-last-most"

subjects = [
    "SCA01",
    "SCA02",
    "SCA03",
    "SCA04",
    "SCA05",
    "SCA06",
    "SCA07",
    "SCA09",
    "SCA10",
    "SCA11",
    "SCA13",
    "SCA14",
    "SCS01",
    "SCS02",
    "SCS03",
    "SCS04",
    "SCS05",
    "SCS06",
    "SCS07",
    "SCS08",
]

df, mat, pkpk_muscles_map, auc_muscles_map = load_non_invasive_data(dir=dir, subjects=subjects)
df.shape


# In[26]:


mat.shape


# In[27]:


target_df = df \
    .groupby(by=["participant"], as_index=False) \
    .agg({"target_muscle": np.unique}) \
    .explode(column="target_muscle") \
    .reset_index(drop=True) \
    .copy()

target_df


# In[28]:


for subject in df.participant.unique():
    ind = df.participant.isin([subject])
    temp_df = df[ind].reset_index(drop=True).copy()

    side = temp_df.target_muscle.unique()[0][0]
    pkpk_sided_muscles = ["PKPK_" + side + m for m in MUSCLES]
    auc_sided_muscles = ["AUC_" + side + m for m in MUSCLES]

    assert (temp_df[PKPK_MUSCLES].to_numpy() == temp_df[pkpk_sided_muscles].to_numpy()).all()
    assert (temp_df[AUC_MUSCLES].to_numpy() == temp_df[auc_sided_muscles].to_numpy()).all()


# In[29]:


dir = "/home/mcintosh/Local/temp/test_hbmep/data"

dest = os.path.join(dir, "proc_2023-11-29_paired.csv")
df.to_csv(dest, index=False)
print("data", dest)

dest = os.path.join(dir, "proc_2023-11-29_paired.npy")
np.save(dest, mat)
print("mat", dest)


# In[30]:


import json

dest = os.path.join(dir, "pkpk_muscles_map.json")
f = open(dest, "w")
f.write(json.dumps(pkpk_muscles_map))
f.close;

dest = os.path.join(dir, "auc_muscles_map.json")
f = open(dest, "w")
f.write(json.dumps(auc_muscles_map))
f.close;


# In[31]:


pkpk_muscles_map


# In[32]:


auc_muscles_map


# In[33]:


src = "/mount/hdd1/human_non-inv/proc-2023-11-20/tms-visit-last-most/SCA01/SCA01_V1T0_TMS_REC_ep_matrix.mat"
data_dict = mat73.loadmat(src)

t = data_dict["t_sliced"]
print(t.min(), t.max())


# In[ ]:


df.shape


# In[ ]:


mat.shape


# In[ ]:


sorted(MUSCLES) == MUSCLES


# In[ ]:




