import pandas as pd
import numpy as np
from pathlib import Path

from src.data_collection import load_solar_data

rawdata_directory = Path("data/raw")
processeddata_directory = Path("data/processed")
processeddata_directory.mkdir(parents=True, exist_ok=True)

#this function loads and cleans one NOAA greenhouse gas dataset
def preprocess_noaa_gas(filepath, gas_name, skip_rows):
    print(f"Processing NOAA {gas_name.upper()}")
    #read the file but don't use the first row as header
    df = pd.read_csv(filepath, skiprows=skip_rows, header=None)
    #name the columns manually based on the file format
    df.columns = ['year', 'month', 'decimal', 'average', 'average_unc', 'trend', 'trend_unc']
    #rename the average column to the gas name
    df = df.rename(columns={"average": gas_name})
    #keep only the important columns
    df = df[["year", "month", gas_name]]
    #create a datetime column using year and month
    df["date"] = pd.to_datetime(
        df["year"].astype(str) + "-" +
        df["month"].astype(str).str.zfill(2) + "-01"
    )
    #removes any missing values from the data
    df = df.dropna()

    #sorts the data chronologically
    df = df.sort_values("date")
    print(f"{gas_name.upper()} rows: {len(df)}")
    return df[["date", gas_name]]

#this function merges the three greenhouse gas datasets
def preprocess_noaa_all():
    print("Merging NOAA greenhouse gas datasets")
    co2 = preprocess_noaa_gas(
        rawdata_directory / "noaa_co2_monthly.csv",
        "co2",
        39)
    ch4 = preprocess_noaa_gas(
        rawdata_directory / "noaa_ch4_monthly.csv",
        "ch4",
        46 )
    n2o = preprocess_noaa_gas(
        rawdata_directory / "noaa_n2o_monthly.csv",
        "n2o",
        46)
    merged = co2.merge(ch4, on="date", how="outer")
    merged = merged.merge(n2o, on="date", how="outer")
    merged = merged.sort_values("date")
    print(f"NOAA merged dataset size: {merged.shape}")

    #save the now processed NOAA gas data
    noaa_path = processeddata_directory / "noaa_processed.csv"
    merged.to_csv(noaa_path, index=False)
    print(f"Saved NOAA data to: {noaa_path}")
    return merged

#this function loads the NASA temperature data from the Kaggle CSV
def load_nasa_temps():
    print("Loading NASA temperature data from Kaggle - ")
    filepath = rawdata_directory / "global_temps.csv"
    if not filepath.exists():
        print("NASA temperature data not found, Please run data collection first")
        print(f"Expected file: {filepath}")
        return pd.DataFrame()
    df = pd.read_csv(filepath)

    #gets year-month format of dataframe
    df_melted = df.melt(id_vars=['Year'], var_name='month', value_name='temp_anomaly')

    #abbreviates the month values to numbers
    month_map = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    df_melted['month_num'] = df_melted['month'].map(month_map)
    df_melted = df_melted.dropna(subset=['month_num'])
    df_melted['month_num'] = df_melted['month_num'].astype(int)

    #creates a date column for nasa data
    df_melted['date'] = pd.to_datetime(
        df_melted['Year'].astype(int).astype(str) + '-' +
        df_melted['month_num'].astype(str).str.zfill(2) + '-01')
    result = df_melted[['date', 'temp_anomaly']].dropna().sort_values('date')
    print(f"Loaded NASA data: {len(result)} months from {result['date'].min().year} to {result['date'].max().year}")

    #saves the processed NASA data in the respective folder
    nasa_path = processeddata_directory / "nasa_processed.csv"
    result.to_csv(nasa_path, index=False)
    print(f"Saved NASA data to: {nasa_path}")

    return result

#this function cleans the OWID dataset and filters to global values
def preprocess_owid():
    print("Processing OWID dataset")
    df = pd.read_csv(rawdata_directory / "owid_co2_data.csv", low_memory=False)
    #keep only global rows
    df = df[df["country"] == "World"]
    #select useful columns for climate drivers (only those that exist)
    columns_to_keep = []
    for col in ["year", "co2", "methane", "nitrous_oxide", "temperature_change_from_ghg"]:
        if col in df.columns:
            columns_to_keep.append(col)
        else:
            print(f"Note: {col} not found in dataset")
    df = df[columns_to_keep]

    #create a date column
    df["date"] = pd.to_datetime(df["year"].astype(str) + "-01-01")
    df = df.sort_values("date")
    print(f"OWID rows: {len(df)}")
    print(f"OWID columns: {list(df.columns)}")

    #save OWID processed data
    owid_path = processeddata_directory / "owid_processed.csv"
    df.to_csv(owid_path, index=False)
    print(f"Saved OWID data to: {owid_path}")
    return df

#this function merges the climate datasets together
def merge_datasets():
    noaa = preprocess_noaa_all()
    owid = preprocess_owid()
    nasa = load_nasa_temps()
    solar = load_solar_data()

    #start with NOAA as the base dataset then merge and format the others
    merged = noaa
    merged = pd.merge(merged, nasa, on="date", how="left") 
    merged = pd.merge(merged, owid, on="date", how="left")
    merged = pd.merge(merged, solar, on="date", how="left")
    merged = merged.sort_values("date")
    merged = merged.ffill()
    merged = merged.rename(columns={"co2_x": "co2_Noaa", "co2_y": "co2_Owid"})
    merged = merged.drop_duplicates(subset="date", keep="first")

    print(f"Final merged dataset size: {merged.shape}")

    #saves the merged data
    merged_path = processeddata_directory / "merged_climate_data.csv"
    merged.to_csv(merged_path, index=False)
    print(f"Saved merged data to: {merged_path}")
    return merged

#this function creates engineered features for regression
def create_features(df):
    print("Creating engineered climate features")
    #time since 1979 baseline
    df["years_since_1979"] = df["date"].dt.year - 1979

    #growth rates for greenhouse gases
    if "co2_Noaa" in df.columns:
        df["co2_growth_rate"] = df["co2_Noaa"].diff()
    if "ch4" in df.columns:
        df["ch4_growth_rate"] = df["ch4"].diff()
    if "n2o" in df.columns:
        df["n2o_growth_rate"] = df["n2o"].diff()

    #lagged greenhouse gas features
    if "co2_Noaa" in df.columns:
        df["co2_lag1"] = df["co2_Noaa"].shift(12)
    if "ch4" in df.columns:
        df["ch4_lag1"] = df["ch4"].shift(12)
    if "n2o" in df.columns:
        df["n2o_lag1"] = df["n2o"].shift(12)

    #12 month moving averages
    if "co2_Noaa" in df.columns:
        df["co2_ma12"] = df["co2_Noaa"].rolling(12).mean()
    if "ch4" in df.columns:
        df["ch4_ma12"] = df["ch4"].rolling(12).mean()
    if "n2o" in df.columns:
        df["n2o_ma12"] = df["n2o"].rolling(12).mean()

    #don't drop all of the NaN rows, just keep the ones we need for analysis
    print(f"Rows before dropping: {len(df)}")
    print(f"Rows after feature engineering: {len(df)}")
    return df

def build_final_dataset():
    merged = merge_datasets()
    final_df = create_features(merged)
    print("Final dataset preview:")
    print(final_df.head(10))  #show more rows to see what's happening
    print(f"Final dataset size: {final_df.shape}")

    #save final dataset with features
    final_path = processeddata_directory / "final_climate_dataset.csv"
    final_df.to_csv(final_path, index=False)
    print(f"Saved final dataset to: {final_path}")
    return final_df

if __name__ == "__main__":
    final_df = build_final_dataset()
    print(f"\nAll files saved to: {processeddata_directory}")