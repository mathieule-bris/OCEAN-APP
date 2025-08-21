import streamlit as st
from scripts.seasonal import check_coords, harmonic_select_coord_allONI

### Imports
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scripts.SHDR import *
from scripts.SHDR_utils import *
from scripts.useful_functions import *
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.optimize import curve_fit

st.set_page_config(page_title="🌊 Ocean Analysis Tool", layout="wide")
st.title("🌊 Ocean Data Explorer")

# Sidebar navigation
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Choose analysis:", ["Check Coordinates", "Harmonic Fit (ONI)"])

# Main page content
if page == "Check Coordinates":
    st.subheader("Coordinate Checking")
    n_coords = st.number_input("Number of coordinates", min_value=1, max_value=20, value=5)
    run_check = st.button("Run Check")

    if run_check:
        figs = check_coords(n=n_coords)  # Make sure check_coords returns a list of matplotlib figures
        if figs is None or len(figs) == 0:
            st.warning("No figures generated.")
        else:
            for fig in figs:
                st.pyplot(fig)

elif page == "Harmonic Fit (ONI)":
    st.subheader("Harmonic Fit by ENSO phase")
    harmonic_select_coord_allONI()
