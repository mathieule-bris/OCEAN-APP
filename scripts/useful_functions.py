import os
import numpy as np
import pandas as pd
import xarray as xr
# from SHDR import *
# from SHDR_utils import *
import pickle
#from ProcessRAWv2 import *
import numpy.random as rd
import matplotlib.pyplot as plt




def plot_times_available_dict(nested_dict,title=None):
    """
    Plot the available time points for each (lat, lon) key in the nested_dict.

    Parameters:
    -----------
    nested_dict : dict
        Dictionary mapping (lat, lon) -> (result_df, filtered_ds), where:
            - result_df is an xarray.Dataset containing SHDR parameters
            - filtered_ds is an xarray.Dataset with filtered temperature profiles

    This function generates a time series plot for each spatial location,
    showing when valid profiles are available after filtering.
    """

    for coord, (result_df, filtered_ds) in nested_dict.items():
        times = pd.to_datetime(filtered_ds['time'].values)
        
        plt.figure(figsize=(10, 1.5))
        plt.plot(times, [1]*len(times), '|', markersize=10)
        plt.title(f"Profile Times at Location {coord}")
        plt.xlabel("Time")
        plt.yticks([])
        plt.grid(True)
        if title:
            plt.title(title)
        plt.tight_layout()
        plt.show()


def convert360_180(_ds):
    """
    Convert longitude from 0-360 to -180 to 180 degrees.
    """
    if 'lon' in _ds.coords:  # Ensure 'lon' exists in dataset
        if _ds['lon'].min() >= 0:  # Check if longitudes are in 0-360 range
            with xr.set_options(keep_attrs=True):
                _ds.coords['lon'] = (_ds['lon'] + 180) % 360 - 180
            _ds = _ds.sortby('lon')  # Sort by longitude after conversion
    return _ds

import matplotlib.pyplot as plt
import numpy as np

def plot_mld_vs_lon_by_lat(mld_values, lats, lons, title=None, ylabel=None):

    """
    Plot MLD variability as a function of longitude for each latitude.
    Longitudes are converted to [0, 360] inside the function.
    """
    lats = np.array(lats)
    lons = np.array(lons)
    mld_values = np.array(mld_values)

    unique_lats = np.unique(lats)
    
    plt.figure(figsize=(14, 6))
    
    for lat in unique_lats:
        mask = lats == lat
        mld_lat = mld_values[mask]
        lons_lat = (lons[mask] + 360) % 360  # Convert to [0, 360]

        # Sort by longitude
        sorted_idx = np.argsort(lons_lat)
        lons_sorted = lons_lat[sorted_idx]
        mld_sorted = mld_lat[sorted_idx]

        plt.plot(lons_sorted, mld_sorted, marker='o', label=f'Lat {lat:.1f}°')
    
    plt.xlabel('Longitude (°E)')
    if ylabel:
        plt.ylabel(ylabel)
    else:
        plt.ylabel('MLD (m)')

    if title:
        plt.title(title)
    else:
        plt.title('MLD Variability vs Longitude for Each Latitude')
    
    plt.legend(title='Latitude')
    plt.grid(True)
    plt.tight_layout()
    plt.show()




def plot_times_available_raw_dict(nested_dict):
    """
    Plot the available time points for each (lat, lon) key in the nested_dict.
    ...
    """

    for coord, raw in nested_dict.items():
        times = pd.to_datetime(raw['time'].values)

        plt.figure(figsize=(10, 1.5))
        plt.plot(times, [1]*len(times), '|', markersize=10)
        plt.title(f"Profile Times at Location {coord}")
        plt.xlabel("Time")
        plt.yticks([])
        plt.grid(True)
        plt.tight_layout()
        plt.show()



def harmonic_select_coord(nested_dict, SOI):###Needs nested_dict and SOI as input

    # Get available coordinates
    coords = list(nested_dict.keys())
    lats = sorted(set(c[0] for c in coords))
    lons = sorted(set(c[1] for c in coords))

    # Create sliders for latitude and longitude with available values
    lat_slider = widgets.SelectionSlider(
        options=lats, 
        value=lats[0], 
        description='Latitude:',
        continuous_update=False
    )

    lon_slider = widgets.SelectionSlider(
        options=lons, 
        value=lons[0], 
        description='Longitude:',
        continuous_update=False
    )


    def update_plot(lat, lon):
        coord = (lat, lon)
        if coord in nested_dict:
            result_df, filtered_ds = nested_dict[coord]

            # Ensure result_df is a pandas DataFrame
            if not isinstance(result_df, pd.DataFrame):
                result_df = result_df.to_dataframe().reset_index()

            result_df['time'] = pd.to_datetime(result_df['time'].values)
            result_df['DOY'] = result_df['time'].dt.dayofyear
            result_df['Year'] = result_df['time'].dt.year

            # Merge with SOI to get ONI classification
            result_df = result_df.merge(SOI, on='Year', how='left')

            # Define ONI to color mapping
            oni_colors = {
                'Neutral': 'grey',
                'ENS': 'red', 'ENM': 'red', 'ENW': 'red', 'ENV': 'red',
                'LNS': 'blue', 'LNM': 'blue', 'LNW': 'blue'
            }

            # Map colors to each ONI type and drop NaNs
            result_df['color'] = result_df['ONI'].map(oni_colors)
            plot_df = result_df.dropna(subset=['color'])

            # Prepare data for fitting
            t = plot_df['DOY'].values  # Day of Year
            y = plot_df['a1'].values   # SST values

            mask = np.isfinite(t) & np.isfinite(y)
            t = t[mask]
            y = y[mask]

            # Harmonic model with 2 harmonics
            T = 365
            omega = 2 * np.pi / T

            def harmonic(t, A0, A1, B1, A2=0, B2=0):
                return (A0 +
                        A1 * np.cos(omega * t) + B1 * np.sin(omega * t) +
                        A2 * np.cos(2 * omega * t) + B2 * np.sin(2 * omega * t))

            # Fit 2 harmonics only
            popt2, _ = curve_fit(harmonic, t, y)

            # Calculate fitted values and RMSE
            y_fit = harmonic(t, *popt2)
            rmse = np.sqrt(np.mean((y - y_fit) ** 2))

            # Plot
            t_fit = np.linspace(1, 365, 365)

            plt.figure(figsize=(10, 5))
            plt.scatter(t, y, s=10, c=plot_df['color'][mask], alpha=0.3, label='All data points')
            plt.plot(t_fit, harmonic(t_fit, *popt2), 'b-', label='2 harmonics fit')
            plt.xlabel('Day of Year')
            plt.ylabel('SST (°C)')
            plt.title(f'Fitting Harmonic Seasonal Cycle Without Averaging\nRMSE: {rmse:.3f} °C')
            # Custom legend
            from matplotlib.lines import Line2D
        # After plotting
            custom_lines = [
                Line2D([0], [0], color='blue', lw=1, label='2 harmonics fit'),
                Line2D([0], [0], marker='o', linestyle='None', color='grey', label='Neutral'),
                Line2D([0], [0], marker='o', linestyle='None', color='red', label='El Niño'),
                Line2D([0], [0], marker='o', linestyle='None', color='blue', label='La Niña')
            ]

            plt.legend(handles=custom_lines)
            plt.grid(True)
            plt.show()

    # Display interactive sliders
    widgets.interact(update_plot, lat=lat_slider, lon=lon_slider)
