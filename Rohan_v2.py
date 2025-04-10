import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from IPython.display import display, HTML
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

class ProductRecommender:
    def __init__(self):
        self.df = None
        self.nlp = spacy.load("en_core_web_md", disable=["parser", "ner", "lemmatizer"])
        self.cosine_sim = None

    def load_and_clean_data(self, file_path):
        """Load and clean the dataset"""
        # Load full dataset
        self.df = pd.read_csv(file_path)
        
        # Sample 10000 random entries
        if len(self.df) > 10000:
            self.df = self.df.sample(n=10000, random_state=42)
        
        # Drop unnecessary columns
        self.df = self.df.drop(['link', 'image'], axis=1)
        
        # Clean ratings
        self.df['ratings'] = pd.to_numeric(self.df['ratings'], errors='coerce')
        self.df['no_of_ratings'] = self.df['no_of_ratings'].str.replace(',', '').apply(pd.to_numeric, errors='coerce')
        self.df = self.df.dropna(subset=['ratings', 'no_of_ratings'])
        self.df['ratings'] = self.df['ratings'].astype(float)
        self.df['no_of_ratings'] = self.df['no_of_ratings'].astype(float)

        # Clean prices
        self.df['discount_price'] = self.df['discount_price'].str.replace('₹', '').str.replace(',', '').apply(pd.to_numeric, errors='coerce')
        self.df['actual_price'] = self.df['actual_price'].str.replace('₹', '').str.replace(',', '').apply(pd.to_numeric, errors='coerce')
        self.df = self.df.dropna(subset=['discount_price', 'actual_price'])
        self.df['discount_price'] = self.df['discount_price'].astype(int)
        self.df['actual_price'] = self.df['actual_price'].astype(int)

        # Remove invalid values
        self.df = self.df[self.df['ratings'].between(1, 5)]
        self.df = self.df[self.df['no_of_ratings'] > 0]
        self.df = self.df[(self.df['discount_price'] > 0) & (self.df['actual_price'] > 0)]
        
        # Remove duplicates and missing values
        self.df = self.df.drop_duplicates()
        self.df = self.df.dropna()

    def clean_text_fields(self):
        """Clean text fields in the dataset"""
        def clean_product_names(name):
            if isinstance(name, str):
                name = re.sub(r'[^a-zA-Z0-9\s]+', '', name)
                name = name.lower().strip()
            return name

        def clean_category(category):
            category = category.lower().strip()
            category = category.replace('&', 'and').replace(',', '')
            if "home" in category:
                category = 'home and kitchen'
            return category

        def clean_sub_category(category):
            category = category.lower().strip()
            category = category.replace('&', 'and').replace(',', '').replace("'", "").replace('-','')
            return category.title()

        self.df['name'] = self.df['name'].apply(clean_product_names)
        self.df['main_category'] = self.df['main_category'].apply(clean_category)
        self.df['sub_category'] = self.df['sub_category'].apply(clean_sub_category)

    def get_top_rated_products(self, n=10):
        """Get top rated products"""
        average_rating = self.df.groupby(['name','main_category', 'sub_category','no_of_ratings',
                                        'discount_price','actual_price'])['ratings'].mean().reset_index()
        top_rated = average_rating.sort_values(by=['ratings', 'no_of_ratings'], ascending=False)
        return top_rated[['name','main_category', 'sub_category', 'no_of_ratings', 
                         'discount_price', 'actual_price','ratings']].head(n)

    def preprocess_text(self):
        """Preprocess text data for recommendations"""
        columns_to_extract_tags_from = ['name', 'main_category', 'sub_category']
        
        for column in columns_to_extract_tags_from:
            texts = self.df[column].astype(str).tolist()
            self.df[column] = [' '.join([token.text for token in doc 
                                       if token.text.isalnum() and token.text not in STOP_WORDS]) 
                              for doc in self.nlp.pipe(texts, batch_size=50)]
        
        self.df['Tags'] = self.df[columns_to_extract_tags_from].apply(lambda row: ','.join(row), axis=1)

    def build_recommendation_engine(self):
        """Build the recommendation engine using cosine similarity"""
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(self.df['Tags'])
        self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    def recommend_products(self, product_name, n_recommendations=10):
        """Get product recommendations based on product name"""
        def clean_product_names(name):
            if isinstance(name, str):
                name = re.sub(r'[^a-zA-Z0-9\s]+', '', name)
                return name.lower().strip()
            return name

        cleaned_name = clean_product_names(product_name)
        matching_mask = self.df['name'].str.contains(cleaned_name, case=False, na=False)
        matching_products = self.df[matching_mask].copy()

        if matching_products.empty:
            search_terms = cleaned_name.split()
            matching_mask = self.df['name'].apply(lambda x: any(term in x.lower() for term in search_terms))
            matching_products = self.df[matching_mask].copy()

        if matching_products.empty:
            print(f"No products found matching '{product_name}'. Try a different search term.")
            return None

        matching_products.loc[:, 'match_score'] = matching_products['name'].apply(
            lambda x: sum(term in x.lower() for term in cleaned_name.split())
        )
        matching_products = matching_products.sort_values('match_score', ascending=False)
        best_match_idx = matching_products.index[0]
        idx_position = self.df.index.get_loc(best_match_idx)

        print(f"\nFound matching product: {self.df.iloc[idx_position]['name']}")
        
        sim_scores = list(enumerate(self.cosine_sim[idx_position]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:n_recommendations + 1]
        
        product_indices = [i[0] for i in sim_scores]
        similarity_scores = [i[1] for i in sim_scores]
        
        recommendations = self.df.iloc[product_indices].copy()
        recommendations.loc[:, 'similarity_score'] = similarity_scores
        recommendations.loc[:, 'similarity_score'] = recommendations['similarity_score'].round(3)
        
        return recommendations[['name', 'main_category', 'sub_category', 'similarity_score']]

# Usage example
if __name__ == "__main__":
    # Use absolute path to the CSV file
    file_path = '/Users/rohanjain/Desktop/UMD - MSML/Sem 2/605_Project//Rohan Trial/Amazon-Products.csv'
    recommender = ProductRecommender()
    recommender.load_and_clean_data(file_path)
    recommender.clean_text_fields()
    
    # Get top rated products
    print("\nTop 10 rated products:")
    print(recommender.get_top_rated_products())
    
    # Build recommendation engine
    recommender.preprocess_text()
    recommender.build_recommendation_engine()
    
    # Get recommendations for a specific product
    product_name = 'tarnkash slim fit stylish stretchable washed jeans'
    recommendations = recommender.recommend_products(product_name)
    
    if recommendations is not None:
        print("\nRecommended products based on cosine similarity:")
        print(recommendations.to_string(index=False))