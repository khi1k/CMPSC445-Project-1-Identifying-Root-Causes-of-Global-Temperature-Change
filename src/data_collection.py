import pandas as pd
import numpy as np
import requests
from pathlib import Path
import kagglehub
import shutil

#first we setup the folders where our raw and processed data will be stored
rawdata_directory = Path('data/raw')
processeddata_directory = Path('data/processed')

#then will create the folders to hold the data if they dont exist yet
rawdata_directory.mkdir(parents=True, exist_ok=True)
processeddata_directory.mkdir(parents=True, exist_ok=True)

#Source 1:
#these are the NOAA greenhouse gas datasets we will download
noaa_urls = {
    'co2': 'https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_gl.csv',
    'ch4': 'https://gml.noaa.gov/webdata/ccgg/trends/ch4/ch4_mm_gl.csv',
    'n2o': 'https://gml.noaa.gov/webdata/ccgg/trends/n2o/n2o_mm_gl.csv'
}

#this function downloads the greenhouse gas datasets from NOAA
def download_noaa_data():
    print("Downloading NOAA greenhouse gas datasets")
    for gas_name, url in noaa_urls.items():
        print(f"Downloading {gas_name.upper()} data")
        save_path = rawdata_directory / f"noaa_{gas_name}_monthly.csv"
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(save_path, 'wb') as f:#writes the downloaded content into a csv file
                f.write(response.content)
            print(f"{gas_name.upper()} saved -> {save_path.name}")
        except Exception as e:
            print(f"{gas_name.upper()} download failed: {e}")

#this function loads one greenhouse gas csv and then cleans the data
def load_noaa_gas(filepath, gas_name, skip_rows):
    try:
        #reads the csv file while skipping the comment lines at the top
        df = pd.read_csv(filepath,
                         skiprows=skip_rows,
                         names=['year','month','decimal_date',
                                gas_name,f'{gas_name}_unc',
                                'trend','trend_unc'])
        df = df[pd.to_numeric(df['year'], errors='coerce').notna()]#removes rows where the year column is not numeric
        df['year'] = df['year'].astype(int)
        df['month'] = df['month'].astype(int)

        #only keeps the months that are between 1 and 12, aka the valid ones
        df = df[(df['month'] >= 1) & (df['month'] <= 12)]

        #creation of a proper datetime column using the year and month data
        df['date'] = pd.to_datetime(
            df['year'].astype(str) + '-' +
            df['month'].astype(str).str.zfill(2) + '-01'
        )
        #keeps only the date and gas concentration columns for noaa dataset
        result = df[['date', gas_name]].copy()
        print(f"{gas_name.upper()} loaded: {len(result)} rows")
        return result
    except Exception as e:
        print(f"Error loading {gas_name}: {e}")
        return None

#this function loads all three NOAA gases and merges them together
def load_noaa_all():
    print("Loading NOAA greenhouse gas data")
    data_frames = []
    #files and the number of rows we skip at the top of each dataset
    files = [
        ('co2', rawdata_directory / "noaa_co2_monthly.csv", 39),
        ('ch4', rawdata_directory / "noaa_ch4_monthly.csv", 46),
        ('n2o', rawdata_directory / "noaa_n2o_monthly.csv", 46)
    ]
    for gas_name, filepath, skip in files:
        if filepath.exists():
            print(f"Processing {gas_name.upper()}")
            df = load_noaa_gas(filepath, gas_name, skip)
            if df is not None:
                data_frames.append(df)
    #merge the datasets together by date
    if data_frames:
        from functools import reduce
        merged = reduce(
            lambda left, right: pd.merge(left, right, on='date', how='outer'),
            data_frames
        )
        merged = merged.sort_values('date').reset_index(drop=True)
        print(f"NOAA merged dataset size: {merged.shape}")
        return merged
    return pd.DataFrame()

#this function downloads the NASA temperature data from Kaggle
def download_nasa_data():
    print("Downloading NASA temperature data from Kaggle")
    try:
        import kagglehub
        #download the latest version
        print("Downloading dataset from Kaggle..")
        path = kagglehub.dataset_download("sujaykapadnis/global-surface-temperatures")
        print(f"Dataset downloaded to cache: {path}")

        #copy the csv files to our raw data folder
        csv_files = list(Path(path).glob("*.csv"))
        if csv_files:
            for file in csv_files:
                destination = rawdata_directory / file.name
                shutil.copy(file, destination)
                print(f"Copied {file.name} to {destination}")
        else:
            print("No CSV files found in the downloaded dataset")

    except ImportError:
        print("kagglehub not installed. Please install it with: pip install kagglehub")
    except Exception as e:
        print(f"NASA data download failed: {e}")
        print("You can manually download the dataset from:")
        print("https://www.kaggle.com/datasets/sujaykapadnis/global-surface-temperatures")
        print("and place the CSV files in data/raw/")

#this function loads the solar nc file and converts it to a dataframe
def load_solar_data():
    print("Loading solar irradiance data")

    file_path = rawdata_directory / "solar_irradiance_yearly.nc"
    csv_path = processeddata_directory / "solar_irradiance_yearly.csv"

    #if we already have a csv version, just load that
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        print(f"Solar data loaded from csv: {len(df)} rows")
        return df
    if not file_path.exists():
        print("Solar data file not found")
        return pd.DataFrame()
    try:
        #try to import xarray, give error if not installed
        try:
            import xarray as xr
        except ImportError:
            print("xarray library not installed. please install it with: pip install xarray")
            return pd.DataFrame()
        #open the netcdf file with decode_times=False to get raw time values
        ds = xr.open_dataset(file_path, engine='h5netcdf', decode_times=False)

        #convert to dataframe
        df = ds.to_dataframe().reset_index()

        #manually convert the time values to years
        if 'time' in df.columns:
            #time is in days since 1610-01-01
            time_units = ds.time.units if hasattr(ds.time, 'units') else 'days since 1610-01-01'
            print(f"Time units: {time_units}")

            #convert days to years
            df['year'] = 1610 + df['time'] / 365.25
            df['date'] = pd.to_datetime(df['year'].astype(int).astype(str) + '-01-01')
        else:
            print("Could not find time column")
            return pd.DataFrame()
        #try to find the solar irradiance column
        tsi_col = None
        for col in df.columns:
            if 'tsi' in col.lower() or 'irradiance' in col.lower():
                tsi_col = col
                break
        if tsi_col:
            result = df[['date', tsi_col]].copy()
            result = result.rename(columns={tsi_col: 'solar_irradiance'})
            result = result.dropna().sort_values('date')

            #save as csv for next time
            result.to_csv(csv_path, index=False)
            print(
                f"Solar data loaded: {len(result)} rows from {result['date'].min().year} to {result['date'].max().year}")
            return result
        else:
            print(f"Could not find solar irradiance column. columns: {df.columns.tolist()}")
            return df
    except Exception as e:
        print(f"Error loading solar data: {e}")
        return pd.DataFrame()

#this function downloads the solar irradiance data from NOAA
def download_solar_data():
    print("Downloading solar irradiance data")
    url = "https://www.ncei.noaa.gov/data/total-solar-irradiance/access/yearly/tsi_v03r00_yearly_s1610_e2025_c20260305.nc"
    save_path = rawdata_directory / "solar_irradiance_yearly.nc"
    try:
        response = requests.get(url, timeout=30, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Solar data saved -> {save_path.name}")
    except Exception as e:
        print(f"Solar data download failed: {e}")


owid_url = "https://owid-public.owid.io/data/co2/owid-co2-data.csv"#owid dataset
#this function downloads the owid dataset
def download_owid_data():
    print("Downloading Our World in Data dataset")
    save_path = rawdata_directory / "owid_co2_data.csv"
    try:
        response = requests.get(owid_url, timeout=60)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"OWID dataset saved -> {save_path.name}")
        return True
    except Exception as e:
        print(f"OWID download failed: {e}")
        return False

#this function loads the owid dataset and filters it to global data
def load_owid_data():
    file_path = rawdata_directory / "owid_co2_data.csv"
    if not file_path.exists():
        print("OWID data file not found")
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path, low_memory=False)
        #only keep the rows that correspond to global data
        global_df = df[df['country'] == 'World'].copy()
        if len(global_df) == 0:
            global_df = df[df['country'].isin(['Global','WORLD'])].copy()
        #create a date column from the year
        global_df['date'] = pd.to_datetime(global_df['year'].astype(str) + '-01-01')
        #columns that are useful for our analysis
        key_cols = [
            'date','year','co2','co2_per_capita',
            'methane','nitrous_oxide','ghg_excluding_lucf'
        ]
        available_cols = [col for col in key_cols if col in global_df.columns]
        result = global_df[available_cols].copy()
        print(f"OWID data loaded: {len(result)} rows")
        return result
    except Exception as e:
        print(f"Error loading OWID dataset: {e}")
        return pd.DataFrame()

#this function downloads all datasets needed for the project
def download_all_data():
    print("Starting climate data downloads")
    download_noaa_data()
    download_owid_data()
    download_solar_data()
    download_nasa_data()
    print("All downloads finished")

#this function loads all datasets and stores them in a dictionary
def load_all_data():
    print("Loading all datasets")
    datasets = {}
    datasets['noaa'] = load_noaa_all()
    datasets['owid'] = load_owid_data()
    datasets['solar'] = load_solar_data()
    return datasets
