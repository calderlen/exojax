from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

# ---------------------------------------------------------------------------
# Configure output paths and NUMBA cache _before_ importing RADIS/ExoMol.
# RADIS installs as a zipped egg in this environment; without an explicit cache
# directory, numba fails with "no locator available" when enabling caching.
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUMBA_CACHE_DIR = Path(os.environ.get("NUMBA_CACHE_DIR", OUTPUT_DIR / "numba_cache"))
NUMBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(NUMBA_CACHE_DIR))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jax import config

from exojax.database.exomol.api import MdbExomol
from exojax.opacity import OpaCKD, OpaPremodit
from exojax.rt import ArtTransPure
from exojax.utils.astrofunc import gravity_jupiter
from exojax.utils.constants import RJ, Rs
from exojax.utils.grids import wavenumber_grid

config.update("jax_enable_x64", True)

REPO_ROOT = HERE.parent
DATA_FILE = HERE / "data" / "spectrum_wasp39b_g395h.csv"
CKD_CACHE = OUTPUT_DIR / "ckd_h2o_wasp39b.npz"
FIG_SPECTRUM = OUTPUT_DIR / "wasp39b_ckd_spectrum.png"
FIG_DIAGNOSTIC = OUTPUT_DIR / "wasp39b_ckd_offsets.png"

PLANET_MASS_MJ = 0.28
PLANET_RADIUS_RJ = 1.27
STELLAR_RADIUS_RS = 0.93

EXOMOL_PATH_ENV = os.environ.get("EXOJAX_H2O_PATH")
if EXOMOL_PATH_ENV:
    EXOMOL_PATH = Path(EXOMOL_PATH_ENV).expanduser()
else:
    EXOMOL_PATH = (REPO_ROOT / ".database/H2O/1H2-16O/POKAZATEL").expanduser()


def load_observation(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = pd.read_csv(path)
    wav_nm = data["wavelength_nm"].to_numpy()
    rr = data["radius_ratio_rr"].to_numpy()
    err = data[["error_of_rr_minus", "error_of_rr_plus"]].to_numpy().T
    return wav_nm, rr, err


def build_model_grid(wav_nm: np.ndarray, ngrid: int = 4096) -> np.ndarray:
    nu_obs = 1.0e7 / wav_nm
    margin = 40.0
    nu_min = max(np.min(nu_obs) - margin, 0.1)
    nu_max = np.max(nu_obs) + margin
    nu_grid, _, resolution = wavenumber_grid(nu_min, nu_max, ngrid, xsmode="premodit")
    print(f"Model grid: {nu_min:.1f}–{nu_max:.1f} cm⁻¹ with R≈{resolution:,.0f} across {ngrid} points")
    return nu_grid


def load_h2o_database(nu_grid: np.ndarray) -> MdbExomol:
    return MdbExomol(
        str(EXOMOL_PATH),
        nurange=nu_grid,
        gpu_transfer=False,
        inherit_dataframe=False,
    )


def compute_lbl(
    base_opa: OpaPremodit,
    art: ArtTransPure,
    Tarr: np.ndarray,
    mmr: np.ndarray,
    mmw: np.ndarray,
    gravity: np.ndarray,
    molmass: float,
    radius_btm: float,
    gravity_btm: float,
) -> np.ndarray:
    xsmatrix = base_opa.xsmatrix(Tarr, art.pressure)
    dtau = art.opacity_profile_xs(xsmatrix, mmr, molmass, gravity)
    return np.asarray(art.run(dtau, Tarr, mmw, radius_btm, gravity_btm))


def setup_ckd(base_opa: OpaPremodit, Tarr: np.ndarray, pressure: np.ndarray) -> OpaCKD:
    if CKD_CACHE.exists():
        print(f"Loading cached CKD tables from {CKD_CACHE}")
        return OpaCKD.from_saved_tables(CKD_CACHE, base_opa=base_opa)

    print("Precomputing CKD tables … this may take a few minutes on the first run")
    opa_ckd = OpaCKD(base_opa, Ng=16, band_width=5.0, band_spacing="linear")
    T_grid = np.linspace(Tarr.min(), Tarr.max(), 12)
    P_grid = np.geomspace(pressure.min(), pressure.max(), 12)
    opa_ckd.precompute_tables(T_grid, P_grid, to_path=str(CKD_CACHE), overwrite=True)
    return opa_ckd


def main(lbl_compute=False) -> None:
    wav_nm, rr_obs, rr_err = load_observation(DATA_FILE)
    wav_obs_um = wav_nm * 1e-3

    nu_grid = build_model_grid(wav_nm)
    art = ArtTransPure(pressure_top=1.0e-10, pressure_btm=1.0, nlayer=200, integration="simpson")

    temperature_profile = np.linspace(950.0, 1250.0, art.nlayer)
    mmw_profile = np.full(art.nlayer, 2.3)
    mmr_h2o = np.full(art.nlayer,1.e-3)

    radius_btm = PLANET_RADIUS_RJ * RJ
    stellar_radius = STELLAR_RADIUS_RS * Rs
    gravity_btm = gravity_jupiter(PLANET_RADIUS_RJ, PLANET_MASS_MJ)
        
    gravity_profile = np.asarray(
        art.gravity_profile(temperature_profile, mmw_profile, radius_btm, gravity_btm)
    )
    
    molmass_h2o = 18.01528  
    already_ckd_exists = CKD_CACHE.exists()
    if already_ckd_exists: 
        print(f"Using existing CKD cache at {CKD_CACHE}")
        base_opa=None
    else:
        mdb_h2o = load_h2o_database(nu_grid)
        base_opa = OpaPremodit(
            mdb_h2o,
            nu_grid,
            auto_trange=[800.0, 1600.0],
            dit_grid_resolution=1.0,
        )

        transit_lbl = compute_lbl(
            base_opa,
            art,
            temperature_profile,
            mmr_h2o,
            mmw_profile,
            gravity_profile,
            molmass_h2o,
            radius_btm,
            gravity_btm,
        )

        radius_ratio_lbl = np.sqrt(transit_lbl) * radius_btm / stellar_radius
        wav_lbl_um = 1.0e4 / nu_grid
        order_lbl = np.argsort(wav_lbl_um)

    opa_ckd = setup_ckd(base_opa, temperature_profile, art.pressure)
    xs_ckd = opa_ckd.xstensor_ckd(temperature_profile, art.pressure)
    dtau_ckd = art.opacity_profile_xs_ckd(
        xs_ckd, mmr_h2o, molmass_h2o, gravity_profile
    )
    transit_ckd = np.asarray(
        art.run_ckd(
            dtau_ckd,
            temperature_profile,
            mmw_profile,
            radius_btm,
            gravity_btm,
            opa_ckd.ckd_info.weights,
        )
    )
    radius_ratio_ckd = np.sqrt(transit_ckd) * radius_btm / stellar_radius

    band_edges = np.asarray(opa_ckd.band_edges)
    
    if lbl_compute:
        transit_avg = []
        for low, high in band_edges:
            mask = (nu_grid >= low) & (nu_grid < high)
            if np.any(mask):
                transit_avg.append(np.mean(transit_lbl[mask]))
            else:
                transit_avg.append(np.nan)
        transit_avg = np.asarray(transit_avg)
        radius_ratio_avg = np.sqrt(transit_avg) * radius_btm / stellar_radius

        valid = np.isfinite(transit_avg)
        rms_rel = np.sqrt(
            np.mean(((transit_ckd[valid] - transit_avg[valid]) / transit_avg[valid]) ** 2)
        )
        max_rel = np.max(np.abs((transit_ckd[valid] - transit_avg[valid]) / transit_avg[valid]))
        print(f"CKD RMS relative error: {rms_rel:.3e}; max relative error: {max_rel:.3e}")

    nu_bands = np.asarray(opa_ckd.nu_bands)
    wav_ckd_um = 1.0e4 / nu_bands
    order_ckd = np.argsort(wav_ckd_um)

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        wav_obs_um,
        rr_obs,
        yerr=rr_err,
        fmt=".",
        color="black",
        alpha=0.5,
        label="JWST G395H",
    )
    if lbl_compute:
        plt.plot(
            wav_lbl_um[order_lbl],
            radius_ratio_lbl[order_lbl],
            color="tab:blue",
            linewidth=1.0,
            label="PreMODIT LBL",
        )
    plt.plot(
        wav_ckd_um[order_ckd],
        radius_ratio_ckd[order_ckd],
        color="tab:orange",
        linewidth=1.5,
        label="OpaCKD (Ng=16)",
    )
    plt.xlabel("Wavelength [µm]")
    plt.ylabel("Radius ratio Rp/Rs")
    plt.title("WASP-39b transmission with H2O (CKD demonstration)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_SPECTRUM, dpi=200)
    print(f"Saved spectrum figure to {FIG_SPECTRUM}")

    if lbl_compute:
    
        plt.figure(figsize=(10, 3.5))
        plt.axhline(0.0, color="0.7", linestyle="--", linewidth=1.0)
        plt.plot(wav_ckd_um[order_ckd], diff_ppm[order_ckd], color="tab:red", linewidth=1.2)
        plt.xlabel("Wavelength [µm]")
        plt.ylabel("CKD − LBL avg [ppm]")
        plt.title("CKD fidelity relative to band-averaged LBL")
        plt.tight_layout()
        plt.savefig(FIG_DIAGNOSTIC, dpi=200)
        print(f"Saved diagnostic figure to {FIG_DIAGNOSTIC}")


if __name__ == "__main__":
    main()
