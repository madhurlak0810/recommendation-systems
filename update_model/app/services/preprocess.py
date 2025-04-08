import pandas as pd
import glob

def load_and_clean_product_data(path_pattern="data/*.csv"):
    # Get all CSV file paths
    files = glob.glob(path_pattern)

    # Fix the index for a known specific file
    amazon_file = "data/Amazon-Products.csv"
    if amazon_file in files:
        df_amazon = pd.read_csv(amazon_file, index_col=0)
        df_amazon.to_csv(amazon_file, index=False)

    cleaned_dfs = []

    for file in files:
        df = pd.read_csv(file)

        # Clean ratings and number of ratings
        df["ratings"] = pd.to_numeric(df.get("ratings"), downcast="float", errors="coerce")
        df["no_of_ratings"] = pd.to_numeric(df.get("no_of_ratings"), downcast="float", errors="coerce")

        # Save cleaned individual file back
        df.to_csv(file, index=False)

        cleaned_dfs.append(df)

    # Combine all DataFrames
    fl = pd.concat(cleaned_dfs, ignore_index=True)
    fl.dropna(inplace=True)
    fl.reset_index(drop=True, inplace=True)

    # Clean up price columns
    fl["discount_price"] = fl["discount_price"].apply(lambda x: float(str(x).strip("₹").replace(",", "")))
    fl["actual_price"] = fl["actual_price"].apply(lambda x: float(str(x).strip("₹").replace(",", "")))
    fl["ratings"] = fl["ratings"].astype(float)
    fl["no_of_ratings"] = fl["no_of_ratings"].astype(float)

    return fl
