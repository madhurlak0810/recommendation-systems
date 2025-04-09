import pandas as pd
import re

class PreprocessService:
    @staticmethod
    def clean_dataframe(df: pd.DataFrame):
        if "Unnamed: 0" in df.columns:
            df.drop("Unnamed: 0", axis=1, inplace=True)

        df["ratings"] = pd.to_numeric(df.get("ratings"), downcast="float", errors="coerce")
        df["no_of_ratings"] = pd.to_numeric(df.get("no_of_ratings"), downcast="float", errors="coerce")
        

        for price_col in ["discount_price", "actual_price"]:
            if price_col in df.columns:
                df[price_col] = df[price_col].astype(str).apply(lambda x: float(str(x).strip("₹").replace(",","")))

        return df

    @staticmethod
    def combine_and_clean(dfs):
        combined = pd.concat(dfs, ignore_index=True)
        combined.dropna(inplace=True)
        combined.reset_index(drop=True, inplace=True)
        return combined
