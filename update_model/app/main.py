import sys
import tqdm
from update_model.app.services.s3_service import S3Service
from update_model.app.services.preprocess import PreprocessService

def main():
    bucket_name = "your-recommender-data"
    prefix = "archive (1)/"
    output_key = "cleaned/models.csv"
    credentials_path = sys.argv[1] if len(sys.argv) > 1 else None

    s3_service = S3Service(credentials_path)
    preprocess = PreprocessService()

    csv_keys = s3_service.list_csv_keys(bucket_name, prefix)

    dfs = []
    for key in tqdm.tqdm(csv_keys, desc="Downloading and cleaning"):
        df = s3_service.read_csv_from_s3(bucket_name, key)
        cleaned_df = preprocess.clean_dataframe(df)
        dfs.append(cleaned_df)

    if dfs:
        final_df = preprocess.combine_and_clean(dfs)
        s3_service.upload_csv_to_s3(final_df, bucket_name, output_key)
        print(f"Cleaned data uploaded to s3://{bucket_name}/{output_key}")
    else:
        print("No valid CSV files were processed.")

if __name__ == "__main__":
    main()
