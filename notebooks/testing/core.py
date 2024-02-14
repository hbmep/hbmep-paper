DATA_PATH = "/home/vishu/data/hbmep-processed/human/tms/proc_2023-11-28.csv"
TOML_PATH = "/home/vishu/repos/hbmep-paper/configs/tms/config.toml"


import pandas as pd
from hbmep.config import Config
from hbmep.model import BaseModel
import hbmep as mep
import hbmep.functional as F

from hbmep_paper.utils import setup_logging

config = Config(TOML_PATH)
config.FEATURES = ["participant", "participant_condition"]
config.BUILD_DIR = "/home/vishu/testing"
config.RESPONSE = config.RESPONSE[0:1]
# config.FEATURES = []
setup_logging(config.BUILD_DIR, "logs")
model = BaseModel(config)


df = pd.read_csv(DATA_PATH)
df, encoder_dict = model.load(df)
# print(encoder_dict.keys())
# print(type(encoder_dict.keys()))
# print(type(encoder_dict))
# print(model._features)
# print(model.features)
# print(model.regressors)
# print(model.response)

# model.plot(df=df, encoder_dict=encoder_dict)
print(df[model.features].to_numpy().T.shape)
