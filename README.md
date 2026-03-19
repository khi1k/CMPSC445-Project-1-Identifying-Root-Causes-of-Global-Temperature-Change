# CMPSC445-Project-1-Identifying-Root-Causes-of-Global-Temperature-Change

## Project Overview - 
This project I have worked on will build a regression-based analytical pipeline that is used in order to identify the 
most influential drivers of global temperature change. I have collected data from four scientific sources to use and merged them
into a singular dataset, trained a regression, xgboost, and random  model from this data. Finally, i used feature ranking 
in order to determine which environmental factors have contributed most to temperature variation globally.

## Report
The full project report is available in the [`report/`](report/) folder:
- [`CMPSC445_Project1_Report.docx`](report/CMPSC445_Project1_Report.docx)

## Dataset Sources - 
- **NOAA Greenhouse Gases**: CO₂, CH₄, N₂O concentrations (monthly, 1979-2025)
- **NASA GISS**: Global temperature anomalies (monthly, 1880-2023) via Kaggle
- **NOAA NCEI**: Total solar irradiance (yearly, 1610-2025)
- **Our World in Data**: CO₂, methane, nitrous oxide emissions (yearly, 1750-2024)

## How to Run:
pip install -r requirements.txt (in terminal) --> run main.py
--> select option 1 for the complete project walkthrough

## Project Structure - 
 - data/: holds all datasets (raw data downloads and processed datasets)
   - raw/: the original downloaded data (not committed to GitHub to save space, can be downloaded through running main.py)
   - processed/: cleaned datasets, as well as merged datasets and the finalized dataset for testing (only including final_climate_dataset.csv in github for simplicity, can be downloaded by running main.py )
 - src/: all of the source code .py files (data_collection, preprocess_data, and model_training)
 - results/: stores all the output files after running main.py
   - correlation_matrix.png 
   - linear_coefficients.png
   - model_results.txt
   - predictions_comparison.png
   - rf_importance.png
 - main.py: runs the complete analysis pipeline of this project
 - requirements.txt: all required python packages you may need to download depending on hardware requirements
 - README.md: current file

## Features Engineered -  
 - Time since baseline: years_since_1979
 - Growth rates: co2_growth_rate, ch4_growth_rate, n2o_growth_rate
 - 12-month moving averages: co2_ma12, ch4_ma12, n2o_ma12
 - Lagged features: co2_lag1, ch4_lag1, n2o_lag1

## Models Used -  
 - Linear Regression
 - Random Forest Regressor
 - XGBoost (Gradient Boosted Trees)

## Results - All outputs will be saved to the 'results' folder:
linear_coefficients.png: Linear regression coefficient plot
rf_importance.png: Random Forest feature importance
predictions_comparison.png: Model predictions vs actual comparing
correlation_matrix.png: Feature correlation heatmap
scatter_top_feature.png: scatter plot with a regression line, showing relationship between CO₂ and global temp change
timeseries_trends.png: A three-part time series plot showing CO₂, CH₄, and N₂O trends alongside global temperature anomalies from 1980-2025
model_results.txt: Performance metrics and feature rankings

## Key Findings - 
 Linear Regression achieved an R² of 0.72, identifying N₂O growth rate and CO₂ as the strongest positive drivers
- Tree-based models struggled with extrapolation as test values exceeded the training range
- Solar irradiance shows minimal influence on recent warming
- For trend-dominated climate data, linear models have proven to be more appropriate and interpretable than the other two models tested