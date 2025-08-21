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
st.title("TAO Data Explorer")

# Sidebar navigation
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Choose analysis:",
        ["Home", "Check Coordinates", "Harmonic Fit (ONI)"],
        index=0  # Default page is Home
    )

# ----- HOME PAGE -----
if page == "Home":
    st.subheader("by Mathieu Le Bris")
    st.markdown("""
    This web application was developed in the context of a study investigating variability in the upper-ocean vertical structure
    of the tropical Pacific using daily temperature profiles from the TAO/TRITON mooring array (1979–2025). 
    The Sharp Homogenization/Diffusive Retreat (SHDR) algorithm was applied to derive mixed layer depth (MLD) and thermocline structure,
    with a focus on seasonal variability and its modulation by El Niño–Southern Oscillation (ENSO) phases.
    
    While the full scientific significance of the figures generated here was not explored in this project, 
    this app provides a user-friendly interface to explore the database created with the SHDR algorithm and most of the methods developed
    during the study. Users can visualize MLD time series, perform harmonic analysis by ENSO phase, and interact with seasonal variability plots.

    All the codes used to generate these analyses, as well as the data required, are available on [GitHub](https://github.com/mathieule-bris/OCEAN-APP),
    allowing anyone to reproduce the results or apply the tools to their own oceanographic datasets.
    
    **Acknowledgements**
    I would like to thank Raquel Somavilla and Ignasi Vallès Casanova for supervising and guiding me
    throughout this work. I would also like to thank César Gonzales Polla for reviewing my work. I would like also to thank these institutions ENSTA, IUEM, IMT Atlantique (universities) and CSIC - IEO Santander (Institute) for their support and collaboration.


    **Features:**
    -  Check and visualize Mixed Layer Depth (MLD) at multiple coordinates.
    -  Perform harmonic analysis based on ENSO phases (ONI classification).
    -  Explore seasonal variability and generate plots automatically.
    
    **How to use the app:**
    1. Select an analysis from the sidebar.
    2. Adjust parameters as needed.
    3. Click the 'Run' button to display results.
    
    This app is built in Python using Streamlit.
    """)

    st.image("assets/TAO_GRID.png",  caption="TAO/TRITON Array in the Tropical Pacific")  

    st.markdown("""
    All the bibliography used to made this app is available in the `references.bib` file, on the depository https://github.com/mathieule-bris/OCEAN-APP along with the report written in the context of this work.
    """)

    st.markdown("""
    For any inquiries contact me via mathieulebris2210@gmail.com or https://www.linkedin.com/in/mathieu0lebris/
    """)

# ----- CHECK COORDINATES -----
elif page == "Check Coordinates":
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

# ----- HARMONIC FIT -----
elif page == "Harmonic Fit (ONI)":
    st.subheader("Harmonic Fit by ENSO phase")
    harmonic_select_coord_allONI()
