import numpy as np
import pandas as pd
import xarray as xr
import ipywidgets as widgets
import pickle
import numpy.random as rd
import matplotlib.pyplot as plt
import os
import streamlit as st
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# ------------------------------------------------------------------
# Resolve paths robustly
# ------------------------------------------------------------------
# Get the absolute path of the folder containing this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Go one level up (from scripts/ to APP/) and into data/
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
DATA_DIR = os.path.abspath(DATA_DIR)  # normalize path

print("Data directory being used:", DATA_DIR)

# ------------------------------------------------------------------
# Load files
# ------------------------------------------------------------------
filename = os.path.join(DATA_DIR, "nested_dict_ALL.pkl")
with open(filename, "rb") as f:
    nested_dict_original = pickle.load(f)

filename = os.path.join(DATA_DIR, "nested_dict_KALMAN_nan.pkl")
with open(filename, "rb") as f:
    nested_dict = pickle.load(f)

nested_dict_nan = nested_dict.copy()
SOI = pd.read_csv(os.path.join(DATA_DIR, "ONI_classification.csv"))


def check_coords2(n=10, plot=True):
    coords = list(nested_dict.keys())  # make sure it's indexable
    figs = []

    for i, coord in enumerate(coords[:n]):  # limit directly
        # try:
        result_df, filtered_ds = nested_dict[coord]
        result_df_original, filtered_ds_original = nested_dict_original[coord]

        # Ensure datetime conversion
        result_df['time'] = (['index'], pd.to_datetime(result_df['time'].values))
        result_df_original['time'] = (['index'], pd.to_datetime(result_df_original['time'].values))
        if plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(result_df['time'], result_df['D1'], label='MLD KALMAN')
            ax.scatter(result_df_original['time'], result_df_original['D1'], label='MLD ORIGINAL')
            ax.set_xlabel('Time')
            ax.set_ylabel('MLD')
            ax.set_title(f'MLD Time Series @ {coord}')
            ax.legend()
            figs.append(fig)

        # except Exception as e:
        #     st.warning(f"Skipping coord {coord} due to error: {e}")

    for fig in figs:
        st.pyplot(fig)


def check_coords(n=5):
    coords = list(nested_dict.keys())[:n]

    for coord in coords:
        result_df, filtered_ds = nested_dict[coord]
        result_df_original, filtered_ds_original = nested_dict_original[coord]


        result_df['time'] = (['index'], pd.to_datetime(result_df['time'].values))
        result_df_original['time'] = (['index'], pd.to_datetime(result_df_original['time'].values))
        # create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(result_df['time'], result_df['D1'], label='MLD KALMAN')
        ax.scatter(result_df_original['time'], result_df_original['D1'], label='MLD ORIGINAL')
        ax.set_xlabel('Time')
        ax.set_ylabel('MLD')
        ax.set_title(f"MLD Time Series at {coord}")
        ax.legend()

        # display figure in main page
        st.pyplot(fig)



from scipy.optimize import curve_fit

def harmonic_select_coord_allONI():
    """Harmonic fit plots for one coord (lat, lon) with 4 subplots and map of all coords."""
    
    # Get available coordinates
    coords = list(nested_dict.keys())
    lats = sorted(set(c[0] for c in coords))
    lons = sorted(set(c[1] for c in coords))
    
    # Sidebar selection using sliders
    with st.sidebar:
        st.subheader("Harmonic Fit Options")
        lat = st.select_slider("Select Latitude", options=lats, value=lats[0])
        lon = st.select_slider("Select Longitude", options=lons, value=lons[0])
        run = st.button("Run Harmonic Fit")
        st.markdown(f"**Selected coordinate:** ({lat}, {lon})")

        # --- Map of all coordinates ---
        fig_map = plt.figure(figsize=(2, 1))

        ax_map = fig_map.add_subplot(1,1,1, projection=ccrs.PlateCarree(central_longitude=180))
        ax_map.set_global()
        ax_map.coastlines()
        ax_map.add_feature(cfeature.LAND, facecolor='lightgray')
        ax_map.add_feature(cfeature.OCEAN, facecolor='lightblue')

        all_lats = [c[0] for c in coords]
        all_lons = [c[1] for c in coords]
        ax_map.scatter(all_lons, all_lats, color='blue', s=10, alpha=0.6, transform=ccrs.PlateCarree())
        ax_map.scatter(lon, lat, color='green', s=30, transform=ccrs.PlateCarree())  # selected point

        ax_map.set_extent([120, 280, -10, 10])
        st.pyplot(fig_map)

    if not run:
        return  # nothing to plot until user clicks

    coord = (lat, lon)
    if coord not in nested_dict:
        st.error("No data for this coord")
        return

    result_df, filtered_ds = nested_dict_nan[coord]

    # --- Rest of your harmonic plotting code ---
    if not isinstance(result_df, pd.DataFrame):
        result_df = result_df.to_dataframe().reset_index()

    result_df["time"] = pd.to_datetime(result_df["time"].values)
    result_df["DOY"] = result_df["time"].dt.dayofyear
    result_df["Year"] = result_df["time"].dt.year

    result_df = result_df.merge(SOI, on="Year", how="left")

    oni_categories = {
        "All": None,
        "El Niño": ["ENS", "ENM", "ENW", "ENV"],
        "La Niña": ["LNS", "LNM", "LNW"],
        "Neutral": ["Neutral"],
    }

    oni_colors = {
        "Neutral": "grey",
        "ENS": "red", "ENM": "red", "ENW": "red", "ENV": "red",
        "LNS": "blue", "LNM": "blue", "LNW": "blue",
    }

    result_df["color"] = result_df["ONI"].map(oni_colors)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, (oni_type, categories) in enumerate(oni_categories.items()):
        ax = axes[i]

        if categories is None:
            plot_df = result_df.dropna(subset=["color"])
        else:
            plot_df = result_df[result_df["ONI"].isin(categories)]

        t = plot_df["DOY"].values
        y = plot_df["D1"].values
        mask = np.isfinite(t) & np.isfinite(y)
        t, y = t[mask], y[mask]

        if len(t) < 5:
            ax.set_title(f"{oni_type} (No data)")
            ax.axis("off")
            continue

        T = 365
        omega = 2 * np.pi / T

        def harmonic(t, A0, A1, B1, A2=0, B2=0):
            return (A0 +
                    A1 * np.cos(omega * t) + B1 * np.sin(omega * t) +
                    A2 * np.cos(2 * omega * t) + B2 * np.sin(2 * omega * t))

        try:
            popt2, _ = curve_fit(harmonic, t, y)
            y_fit = harmonic(t, *popt2)
            rmse = np.sqrt(np.mean((y - y_fit) ** 2))
        except:
            rmse = np.nan
            y_fit = None

        ax.scatter(t, y, s=10, c=plot_df["color"].values[mask], alpha=0.3)
        if y_fit is not None:
            t_fit = np.linspace(1, 365, 365)
            ax.plot(t_fit, harmonic(t_fit, *popt2), "b-")
        ax.set_title(f"{oni_type}\nRMSE: {rmse:.2f}")
        ax.set_xlabel("DOY")
        if i % 2 == 0:
            ax.set_ylabel("MLD (m)")
        ax.grid(True)

    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color="blue", lw=1, label="2 harmonics fit"),
        Line2D([0], [0], marker="o", linestyle="None", color="grey", label="Neutral"),
        Line2D([0], [0], marker="o", linestyle="None", color="red", label="El Niño"),
        Line2D([0], [0], marker="o", linestyle="None", color="blue", label="La Niña"),
    ]
    fig.legend(handles=custom_lines, loc="upper right")
    fig.suptitle(f"Harmonic Seasonal Cycle at (Lat {lat}, Lon {lon})", fontsize=14)
    plt.tight_layout()

    st.pyplot(fig)

def harmonic_plot_at_coord(lat, lon):
    """Plot harmonic fit for a single coordinate (lat, lon)."""
    coord = (lat, lon)
    if coord not in nested_dict:
        print("No data for this coord")
        return None

    result_df, filtered_ds = nested_dict[coord]

    if not isinstance(result_df, pd.DataFrame):
        result_df = result_df.to_dataframe().reset_index()

    result_df['time'] = pd.to_datetime(result_df['time'].values)
    result_df['DOY'] = result_df['time'].dt.dayofyear
    result_df['Year'] = result_df['time'].dt.year
    result_df = result_df.merge(SOI, on='Year', how='left')

    oni_categories = {
        "All": None,
        "El Niño": ['ENS', 'ENM', 'ENW', 'ENV'],
        "La Niña": ['LNS', 'LNM', 'LNW'],
        "Neutral": ['Neutral']
    }

    oni_colors = {
        'Neutral': 'grey',
        'ENS': 'red', 'ENM': 'red', 'ENW': 'red', 'ENV': 'red',
        'LNS': 'blue', 'LNM': 'blue', 'LNW': 'blue'
    }

    result_df['color'] = result_df['ONI'].map(oni_colors)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, (oni_type, categories) in enumerate(oni_categories.items()):
        ax = axes[i]

        if categories is None:
            plot_df = result_df.dropna(subset=['color'])
        else:
            plot_df = result_df[result_df['ONI'].isin(categories)]

        t = plot_df['DOY'].values
        y = plot_df['D1'].values
        mask = np.isfinite(t) & np.isfinite(y)
        t, y = t[mask], y[mask]

        if len(t) < 5:
            ax.set_title(f"{oni_type} (No data)")
            ax.axis("off")
            continue

        T = 365
        omega = 2 * np.pi / T

        def harmonic(t, A0, A1, B1, A2=0, B2=0):
            return (A0 + A1*np.cos(omega*t) + B1*np.sin(omega*t) +
                    A2*np.cos(2*omega*t) + B2*np.sin(2*omega*t))

        try:
            popt2, _ = curve_fit(harmonic, t, y)
            y_fit = harmonic(t, *popt2)
            rmse = np.sqrt(np.mean((y - y_fit)**2))
        except:
            y_fit = None
            rmse = np.nan

        ax.scatter(t, y, s=10, c=plot_df['color'].values[mask], alpha=0.3)
        if y_fit is not None:
            t_fit = np.linspace(1, 365, 365)
            ax.plot(t_fit, harmonic(t_fit, *popt2), 'b-')
        ax.set_title(f"{oni_type}\nRMSE: {rmse:.2f}")
        ax.set_xlabel("DOY")
        if i % 2 == 0:
            ax.set_ylabel("MLD (m)")
        ax.grid(True)

    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='blue', lw=1, label='2 harmonics fit'),
        Line2D([0], [0], marker='o', linestyle='None', color='grey', label='Neutral'),
        Line2D([0], [0], marker='o', linestyle='None', color='red', label='El Niño'),
        Line2D([0], [0], marker='o', linestyle='None', color='blue', label='La Niña')
    ]
    fig.legend(handles=custom_lines, loc='upper right')
    fig.suptitle(f"Harmonic Seasonal Cycle at (Lat {lat}, Lon {lon})", fontsize=14)
    plt.tight_layout()
    return fig

