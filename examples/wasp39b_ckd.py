# %%
import numpy as np
import matplotlib.pyplot as plt
from jax import config
import pandas as pd
# ExoJAX imports
from exojax.test.emulate_mdb import mock_mdbExomol, mock_wavenumber_grid
from exojax.opacity import OpaCKD, OpaPremodit
from exojax.rt import ArtTransPure

# Enable 64-bit precision for accurate calculations
config.update("jax_enable_x64", True)


# %%
data = pd.read_csv("data/spectrum_wasp39b_g395h.csv")
wav_obs = data["wavelength_nm"].values
rr_obs = data["radius_ratio_rr"].values


# %%
