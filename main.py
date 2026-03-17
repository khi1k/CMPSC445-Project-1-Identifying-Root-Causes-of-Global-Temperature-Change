import os
import sys
from pathlib import Path

#add src to path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

#setup paths
rawdata_directory = Path("data/raw")
processeddata_directory = Path("data/processed")
results_directory = Path("results")

#create directories if they dont exist
rawdata_directory.mkdir(parents=True, exist_ok=True)
processeddata_directory.mkdir(parents=True, exist_ok=True)
results_directory.mkdir(parents=True, exist_ok=True)

#collecting the needed datasets
def run_data_collection():
    print("Step 1 - Data Collection")
    try:
        from data_collection import download_all_data
        download_all_data()
        print("\nData collection completed successfully")
    except ImportError as e:
        print(f"Error importing data_collection module: {e}")
        print("Make sure data_collection.py exists in the src folder")
        sys.exit(1)
    except Exception as e:
        print(f"Error during data collection: {e}")
        print("You may need to manually download some of these datasets")

#Cleaning and preprocessing datasets
def run_preprocessing():
    print("\nStep 2 - Preprocessing data")
    try:
        from preprocess_data import build_final_dataset
        final_df = build_final_dataset()
        print("\nPreprocessing completed")
        return final_df
    except ImportError as e:
        print(f"Error importing preprocess_data module: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        sys.exit(1)

#Step 3: Training the models and creating the visualizations
def run_modeling():
    print("\nStep 3 - Modeling and Visualization Creation")
    try:
        from model_training import run_modeling as run_models
        run_models()
        print("\nModeling and visualization completed")
    except ImportError as e:
        print(f"Error importing the modeling module: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during modeling the dataset: {e}")
        sys.exit(1)

#checking required packages are installed
def check_requirements():
    required_packages = ["pandas", "numpy", "matplotlib", "sklearn", "seaborn", "requests"]
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    if missing:
        print("Missing required packages to run this project:")
        for p in missing:
            print(f"   - {p}")
        print("\nPlease install them using the following command in terminal:")
        print("pip install -r requirements.txt")
        return False
    return True

#running the full analysis of the dataset
def main():
    print("\nClimate Route Cause Analysis:")
    print("Identifying drivers of global temperature change")

    #check requirements first
    if not check_requirements():
        sys.exit(1)
    #ask user what they want to run depending on their needs
    print("\nWhat would you like to do?")
    print("1. Run complete pipeline (download, preprocess, model)")
    print("2. Run preprocessing only (if data already downloaded)")
    print("3. Run modeling only (if data already preprocessed)")

    choice = input("\nEnter choice (1-3) -  ").strip()
    if choice == "1":
        #full pipeline
        run_data_collection()
        run_preprocessing()
        run_modeling()
    elif choice == "2":
        #preprocessing only
        run_preprocessing()
    elif choice == "3":
        #modeling only
        #check if processed data exists
        data_path = processeddata_directory / "final_climate_dataset.csv"
        if not data_path.exists():
            print(f"\nProcessed data not found at: {data_path}")
            print("Please run preprocessing first (option 2)")
        else:
            run_modeling()
    else:
        print("\nInvalid choice")
        sys.exit(1)

    print("\nAnalysis has been finished")
    print(f"\nResults saved to: {results_directory.absolute()}")

if __name__ == "__main__":
    main()