import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import joblib
from lightfm import LightFM
from lightfm.data import Dataset
import os
import os # Import os for path joining


class ProductHybridModel:
    """
    A class to handle the training and recommendation of products using a hybrid model based on SVD.
    """
    def __init__(self, csv_path, save_model_dir, n_reviews=10000, n_components=10):
        """
        Initializes the ProductHybridModel with the path to the CSV file and the directory to save the model.

        Args:
            csv_path (str): The path to the CSV file containing reviews.
            save_model_dir (str): Directory to save the model components.
            n_reviews (int): The number of reviews to use from the dataset.
            n_components (int): The number of components for TruncatedSVD.
        """
        self.csv_path = csv_path
        self.save_model_dir = save_model_dir
        self.n_reviews = n_reviews
        self.n_components = n_components
        self.model_path = os.path.join(self.save_model_dir, 'svd_model_user_data.joblib')
        self.decomposed_matrix = None
        self.product_names = None
        self.correlation_matrix = None
        
        self.train_and_save_model()

    def train_and_save_model(self):
        """
        Loads data, trains an SVD model, saves the decomposed matrix and product names,
        and stores model components within the instance.

        Returns:
            bool: True if training and saving were successful, False otherwise.
        """
        # Load data
        try:
            df = pd.read_csv(self.csv_path)
            print(f"Data loaded successfully from {self.csv_path}")
        except FileNotFoundError:
            print(f"Error: File not found at {self.csv_path}")
            return False
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return False

        # Select a subset of data
        new_df1 = df.head(self.n_reviews)
        print(f"Using the first {len(new_df1)} reviews for training.")

        # Create user-item matrix
        try:
            ratings_matrix = new_df1.pivot_table(values='rating', index='user', columns='name', fill_value=0)
            print("Pivot table created successfully.")
        except Exception as e:
            print(f"Error creating pivot table: {e}")
            return False

        # Transpose the matrix for item-based analysis
        X = ratings_matrix.T
        self.product_names = list(X.index)
        print(f"Matrix transposed. Shape: {X.shape}")

        # Decompose the matrix using SVD
        try:
            SVD_model = TruncatedSVD(n_components=self.n_components, random_state=42)
            self.decomposed_matrix = SVD_model.fit_transform(X)
            print(f"SVD performed with {self.n_components} components. Decomposed matrix shape: {self.decomposed_matrix.shape}")
        except Exception as e:
            print(f"Error during SVD fitting/transformation: {e}")
            return False

        # Prepare data to save
        model_data = {
            'decomposed_matrix': self.decomposed_matrix,
            'product_names': self.product_names
        }

        # Ensure the save directory exists
        os.makedirs(self.save_model_dir, exist_ok=True)

        # Save the decomposed matrix and product names
        try:
            joblib.dump(model_data, self.model_path)
            print(f"Model data (decomposed matrix and product names) saved to {self.model_path}")
            # Calculate correlation matrix after successful training
            self._calculate_correlation_matrix()
            return True
        except Exception as e:
            print(f"Error saving model data to {self.model_path}: {e}")
            return False

    def _load_model_data(self):
        """Loads model data from the file if not already loaded."""
        if self.decomposed_matrix is None or self.product_names is None:
            try:
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
        if not self._load_model_data():
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

class ProductCollabModel:
    """
    A class to handle the training and recommendation of products using a collaborative filtering model (LightFM).
    """
    def __init__(self, csv_path, save_model_dir, n_reviews=10000, n_components=10, loss='warp', epochs=30, num_threads=4):
        """
        Initializes the ProductCollabModel.

        Args:
            csv_path (str): The path to the CSV file containing reviews.
            save_model_dir (str): Directory to save the model components.
            n_reviews (int): The number of reviews to use from the dataset.
            n_components (int): The number of latent components for the LightFM model.
            loss (str): The loss function for LightFM ('warp', 'logistic', 'bpr', 'warp-kos').
            epochs (int): The number of training epochs.
            num_threads (int): The number of parallel threads to use for training.
        """
        self.csv_path = csv_path
        self.save_model_dir = save_model_dir
        self.n_reviews = n_reviews
        self.n_components = n_components
        self.loss = loss
        self.epochs = epochs
        self.num_threads = num_threads
        self.model_path = os.path.join(self.save_model_dir, 'lightfm_collab_model.joblib')
        self.model = None
        self.dataset = None
        self.interactions = None
        self.user_id_map = None
        self.item_id_map = None

        self.train_and_save_model()

    def train_and_save_model(self):
        """
        Loads data, trains a LightFM model, saves the model, dataset, and interactions,
        and stores model components within the instance.

        Returns:
            bool: True if training and saving were successful, False otherwise.
        """
        # Load data
        try:
            df = pd.read_csv(self.csv_path)
            print(f"Data loaded successfully from {self.csv_path}")
        except FileNotFoundError:
            print(f"Error: File not found at {self.csv_path}")
            return False
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return False

        # Select a subset of data
        df_subset = df.head(self.n_reviews).copy() # Use copy to avoid SettingWithCopyWarning
        print(f"Using the first {len(df_subset)} reviews for training.")

        # Prepare dataset
        try:
            self.dataset = Dataset()
            # Ensure user and name columns are suitable type (e.g., string)
            df_subset['user'] = df_subset['user'].astype(str)
            df_subset['name'] = df_subset['name'].astype(str)

            self.dataset.fit(df_subset['user'], df_subset['name'])
            self.user_id_map, _, self.item_id_map, _ = self.dataset.mapping()
            print("LightFM dataset fitted.")
        except Exception as e:
            print(f"Error fitting LightFM dataset: {e}")
            return False

        # Build interactions
        try:
            # Ensure rating is numeric
            df_subset['rating'] = pd.to_numeric(df_subset['rating'], errors='coerce')
            df_subset.dropna(subset=['rating'], inplace=True) # Drop rows where rating couldn't be converted

            (self.interactions, _) = self.dataset.build_interactions(
                (row['user'], row['name'], row['rating']) for _, row in df_subset.iterrows()
            )
            print(f"Interactions built. Shape: {self.interactions.shape}")
        except Exception as e:
            print(f"Error building LightFM interactions: {e}")
            return False

        # Train model
        try:
            self.model = LightFM(no_components=self.n_components, loss=self.loss, random_state=42)
            self.model.fit(self.interactions, epochs=self.epochs, num_threads=self.num_threads, verbose=True)
            print("LightFM model trained.")
        except Exception as e:
            print(f"Error training LightFM model: {e}")
            return False

        # Prepare data to save
        model_data = {
            'model': self.model,
            'dataset': self.dataset,
            'interactions': self.interactions,
            'user_id_map': self.user_id_map,
            'item_id_map': self.item_id_map
        }

        # Ensure the save directory exists
        os.makedirs(self.save_model_dir, exist_ok=True)

        # Save the model components
        try:
            joblib.dump(model_data, self.model_path)
            print(f"Collaborative model data saved to {self.model_path}")
            return True
        except Exception as e:
            print(f"Error saving collaborative model data to {self.model_path}: {e}")
            return False

    def _load_model_data(self):
        """Loads collaborative model data from the file if not already loaded."""
        if self.model is None or self.dataset is None or self.interactions is None:
            try:
                model_data = joblib.load(self.model_path)
                self.model = model_data['model']
                self.dataset = model_data['dataset']
                self.interactions = model_data['interactions']
                self.user_id_map = model_data.get('user_id_map') # Use .get for backward compatibility
                self.item_id_map = model_data.get('item_id_map')
                # Regenerate maps if they weren't saved (older versions)
                if self.user_id_map is None or self.item_id_map is None:
                        self.user_id_map, _, self.item_id_map, _ = self.dataset.mapping()
                print(f"Collaborative model data loaded successfully from {self.model_path}")
                return True
            except FileNotFoundError:
                print(f"Error: Collaborative model file not found at {self.model_path}")
                return False
            except KeyError as e:
                print(f"Error: Collaborative model file at {self.model_path} is missing expected key: {e}")
                return False
            except Exception as e:
                print(f"Error loading collaborative model data from {self.model_path}: {e}")
                return False
        return True # Already loaded

    def recommend_products(self, user_id, top_n=10):
        """
        Recommends products for a given user using the trained LightFM model.

        Args:
            user_id (str): The user ID for whom to generate recommendations.
            top_n (int): The maximum number of recommendations to return.

        Returns:
            list: A list of recommended product names (str).
            None: If the model cannot be loaded, the user is unknown, or an error occurs.
        """
        # Ensure model data is loaded
        if not self._load_model_data():
            return None

        if self.model is None or self.dataset is None or self.user_id_map is None or self.item_id_map is None:
                print("Error: Model components not properly loaded.")
                return None

        # Check if user_id exists in the training data
        if user_id not in self.user_id_map:
            print(f"Error: User '{user_id}' not found in the training data.")
            # Option: Return popular items or None
            return None # Or implement popular item fallback

        try:
            # Get the internal user index
            internal_user_id = self.user_id_map[user_id]

            # Get all item internal IDs
            n_users, n_items = self.interactions.shape
            item_indices = np.arange(n_items)

            # Predict scores for all items for the given user
            scores = self.model.predict(internal_user_id, item_indices, num_threads=self.num_threads)

            # Get item names back from internal IDs
            # Create reverse mapping from internal item ID to original item name
            reverse_item_map = {v: k for k, v in self.item_id_map.items()}

            # Combine scores with item names
            scored_items = sorted(zip(scores, item_indices), key=lambda x: x[0], reverse=True)

            # Get top N recommendations (excluding items the user already interacted with, if desired - requires interactions matrix)
            # For simplicity, we just return the top N scored items here.
            # A more robust implementation might filter out already rated items.
            recommended_item_indices = [item_idx for score, item_idx in scored_items[:top_n*2]] # Get more initially to allow filtering
            recommended_item_names = [reverse_item_map[idx] for idx in recommended_item_indices if idx in reverse_item_map]


            print(f"Generated {len(recommended_item_names[:top_n])} recommendations for user '{user_id}'.")
            return recommended_item_names[:top_n]

        except KeyError:
                print(f"Error: Internal mapping issue for user '{user_id}'.")
                return None
        except Exception as e:
            print(f"Error generating recommendations for user '{user_id}': {e}")
            return None
