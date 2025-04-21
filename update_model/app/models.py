import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import joblib
import os # Import os for path joining



def train_and_save_model(csv_path, save_model_dir, n_reviews=10000, n_components=10):
    """
    Loads data, trains an SVD model, and saves the decomposed matrix and product names.

    Args:
        csv_path (str): The path to the CSV file containing reviews.
        save_model_dir (str): Directory to save the model components.
        n_reviews (int): The number of reviews to use from the dataset.
        n_components (int): The number of components for TruncatedSVD.

    Returns:
        bool: True if training and saving were successful, False otherwise.
    """
    # Load data
    try:
        df = pd.read_csv(csv_path)
        print(f"Data loaded successfully from {csv_path}")
    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
        return False
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return False

    # Select a subset of data
    new_df1 = df.head(n_reviews)
    print(f"Using the first {len(new_df1)} reviews for training.")

    # Create user-item matrix
    try:
        ratings_matrix = new_df1.pivot_table(values='Score', index='UserId', columns='ProductId', fill_value=0)
        print("Pivot table created successfully.")
    except Exception as e:
        print(f"Error creating pivot table: {e}")
        return False

    # Transpose the matrix for item-based analysis
    X = ratings_matrix.T
    product_names = list(X.index)
    print(f"Matrix transposed. Shape: {X.shape}")

    # Decompose the matrix using SVD
    try:
        SVD_model = TruncatedSVD(n_components=n_components, random_state=42)
        decomposed_matrix = SVD_model.fit_transform(X)
        print(f"SVD performed with {n_components} components. Decomposed matrix shape: {decomposed_matrix.shape}")
    except Exception as e:
        print(f"Error during SVD fitting/transformation: {e}")
        return False

    # Prepare data to save
    model_data = {
        'decomposed_matrix': decomposed_matrix,
        'product_names': product_names
    }

    # Ensure the save directory exists
    os.makedirs(save_model_dir, exist_ok=True)
    save_model_path = os.path.join(save_model_dir, 'svd_model_data.joblib')

    # Save the decomposed matrix and product names
    try:
        joblib.dump(model_data, save_model_path)
        print(f"Model data (decomposed matrix and product names) saved to {save_model_path}")
        return True
    except Exception as e:
        print(f"Error saving model data to {save_model_path}: {e}")
        return False


def recommend_products_from_model(model_path, target_product_id, correlation_threshold=0.65):
    """
    Loads a pre-trained model (decomposed matrix and product names) and recommends products.

    Args:
        model_path (str): Path to the saved model data file (.joblib).
        target_product_id (str): The ProductId for which to generate recommendations.
        correlation_threshold (float): The minimum correlation coefficient for a product to be recommended.

    Returns:
        list: A list of recommended ProductIds.
        None: If the model file is not found, the target product is not in the model, or an error occurs.
    """
    # Load the model data
    try:
        model_data = joblib.load(model_path)
        decomposed_matrix = model_data['decomposed_matrix']
        product_names = model_data['product_names']
        print(f"Model data loaded successfully from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None
    except KeyError as e:
        print(f"Error: Model file at {model_path} is missing expected key: {e}")
        return None
    except Exception as e:
        print(f"Error loading model data from {model_path}: {e}")
        return None

    # Find the index of the target product
    try:
        product_ID_index = product_names.index(target_product_id)
    except ValueError:
        print(f"Error: ProductId '{target_product_id}' not found in the loaded model data.")
        return None

    # Calculate the correlation matrix from the decomposed matrix
    try:
        # Ensure the decomposed matrix is suitable for correlation calculation
        if decomposed_matrix.shape[0] <= 1:
             print(f"Error: Not enough products ({decomposed_matrix.shape[0]}) in the model to calculate correlation.")
             return None
        correlation_matrix = np.corrcoef(decomposed_matrix)
        # Handle potential NaN values if a row in decomposed_matrix has zero variance
        correlation_matrix = np.nan_to_num(correlation_matrix)
        print("Correlation matrix calculated.")
    except Exception as e:
        print(f"Error calculating correlation matrix: {e}")
        return None

    # Get the correlation vector for the target product
    correlation_product_ID = correlation_matrix[product_ID_index]

    # Generate recommendations
    try:
        # Find indices where correlation exceeds the threshold
        recommend_indices = np.where(correlation_product_ID > correlation_threshold)[0]
        # Filter out the target product itself and map indices back to product names
        Recommend = [product_names[idx] for idx in recommend_indices if idx != product_ID_index]
        print(f"Found {len(Recommend)} recommendations for '{target_product_id}' with threshold {correlation_threshold}.")
    except Exception as e:
        print(f"Error generating recommendations from correlations: {e}")
        return None

    # Return the top recommendations (limited to 24 as before)
    return Recommend[:24]
