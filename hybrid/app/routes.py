from fastapi import APIRouter, Query,Request
import pandas as pd
import boto3
import joblib
import numpy as np

router = APIRouter()

class ProductHybridModel:
    """
    A class to handle the training and recommendation of products using a hybrid model based on SVD.
    """
    def __init__(self):
        """
        Initializes the ProductHybridModel with the path to the CSV file and the directory to save the model.

        Args:
            csv_path (str): The path to the CSV file containing reviews.
            save_model_dir (str): Directory to save the model components.
            n_reviews (int): The number of reviews to use from the dataset.
            n_components (int): The number of components for TruncatedSVD.
        """
        self.csv_path = None
        self.save_model_dir = None
        self.n_reviews = None
        self.n_components = None
        self.model_path = None
        self.decomposed_matrix = None
        self.product_names = None
        self.correlation_matrix = None


    def load_model(self,model_path):
        """Loads model data from the file if not already loaded."""
        if self.decomposed_matrix is None or self.product_names is None:
            try:
                self.model_path = model_path
                model_data = joblib.load(self.model_path)
                self.decomposed_matrix = model_data['decomposed_matrix']
                self.product_names = model_data['product_names']
                print(f"Model data loaded successfully from {self.model_path}")
                return True
            except FileNotFoundError:
                print(f"Error: Model file not found at {self.model_path}")
                return False
            except KeyError as e:
                print(f"Error: Model file at {self.model_path} is missing expected key: {e}")
                return False
            except Exception as e:
                print(f"Error loading model data from {self.model_path}: {e}")
                return False
        return True # Already loaded

    def _calculate_correlation_matrix(self):
        """Calculates and stores the correlation matrix."""
        if self.decomposed_matrix is None:
            print("Error: Decomposed matrix not available. Train or load the model first.")
            return False
        try:
            # Ensure the decomposed matrix is suitable for correlation calculation
            if self.decomposed_matrix.shape[0] <= 1:
                print(f"Error: Not enough products ({self.decomposed_matrix.shape[0]}) in the model to calculate correlation.")
                self.correlation_matrix = None
                return False
            self.correlation_matrix = np.corrcoef(self.decomposed_matrix)
            # Handle potential NaN values if a row in decomposed_matrix has zero variance
            self.correlation_matrix = np.nan_to_num(self.correlation_matrix)
            print("Correlation matrix calculated and stored.")
            return True
        except Exception as e:
            print(f"Error calculating correlation matrix: {e}")
            self.correlation_matrix = None
            return False


    def recommend_products(self, target_product_id, correlation_threshold=0.65, top_n=24):
        """
        Recommends products based on the loaded or trained model.

        Args:
            target_product_id (str): The ProductId for which to generate recommendations.
            correlation_threshold (float): The minimum correlation coefficient for a product to be recommended.
            top_n (int): The maximum number of recommendations to return.

        Returns:
            list: A list of recommended ProductIds.
            None: If the model cannot be loaded, the target product is not found, or an error occurs.
        """
        # Ensure model data is loaded
        if not self.load_model(self.model_path):
            return None

        # Ensure correlation matrix is calculated
        if self.correlation_matrix is None:
            if not self._calculate_correlation_matrix():
                 return None

        # Find the index of the target product
        try:
            product_ID_index = self.product_names.index(target_product_id)
        except ValueError:
            print(f"Error: ProductId '{target_product_id}' not found in the loaded model data.")
            return None
        except TypeError:
             print(f"Error: Product names list is not available or not a list.")
             return None


        # Get the correlation vector for the target product
        correlation_product_ID = self.correlation_matrix[product_ID_index]

        # Generate recommendations
        try:
            # Find indices where correlation exceeds the threshold
            recommend_indices = np.where(correlation_product_ID > correlation_threshold)[0]
            # Filter out the target product itself and map indices back to product names
            Recommend = [self.product_names[idx] for idx in recommend_indices if idx != product_ID_index]
            print(f"Found {len(Recommend)} recommendations for '{target_product_id}' with threshold {correlation_threshold}.")
        except Exception as e:
            print(f"Error generating recommendations from correlations: {e}")
            return None

        # Return the top recommendations
        return Recommend[:top_n]
    

def download_joblib_from_s3(bucket_name, s3_key, local_file_path):
    s3 = boto3.client('s3')
    try:
        with open(local_file_path, 'wb') as f:
            s3.download_fileobj(bucket_name, s3_key, f)
            print(f"Successfully downloaded {s3_key} from s3://{bucket_name} to {local_file_path}")
    except Exception as e:
        print(f"Error downloading {s3_key} from s3://{bucket_name}: {e}")

        
@router.get("/recommend_products")
def similar_products(request: Request,product_name: str = Query(..., description="Product name")):
    try:
        model=request.app.state.model
        recommendations = model.recommend_products(product_name)
        return {"recommended_products": recommendations}
    except Exception as e:
        return {"error": str(e)}