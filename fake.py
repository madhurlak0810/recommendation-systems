import numpy as np
import pandas as pd
import random
import tqdm

# GLOBAL POPULAR/UNIVERSAL ITEMS (reuse these across all users)
def get_global_items(df, n=10):
    return df['name'].value_counts().head(n).index.tolist()  # top N popular items

# NICHE ITEMS (least popular in the dataset)
def get_niche_items(df, n=10):
    return df['name'].value_counts().tail(n).index.tolist()

# Biased sampling function to reuse popular items more
def sample_with_popularity(df_cat: pd.DataFrame, n):
    item_counts = df_cat['name'].value_counts()
    weights = df_cat['name'].map(item_counts).fillna(1)
    return df_cat.sample(n=n, weights=weights.values, replace=True)

def create_category_user(categories, item_count=100, category_range=(4,5), other_range=(1,3)):
    user = {}

    # pick random category
    category = random.choice(categories)

    # sample
    df_cat: pd.DataFrame = df[df["main_category"] == category]
    df_other: pd.DataFrame = df[df["main_category"] != category]
    # cat_sample = df_cat.sample(n=item_count//2, replace=True) # old
    # other_sample = df_other.sample(n=item_count//2, replace=True) # old
    cat_sample = sample_with_popularity(df_cat, item_count // 2) # new 
    other_sample = sample_with_popularity(df_other, item_count // 2) # new

    # ratings
    cat_ratings = np.random.randint(category_range[0], category_range[1] + 1, size=cat_sample.shape[0])
    other_ratings = np.random.randint(other_range[0], other_range[1] + 1, size=other_sample.shape[0])

    # update user
    user.update(dict(zip(cat_sample["name"], cat_ratings)))
    user.update(dict(zip(other_sample["name"], other_ratings)))

    # Add global items (universal to all users)
    for item in global_items: # new
        if item not in user:  # prevent duplicate entries
            user[item] = random.randint(3, 5)
    
    # Add niche items (random few per user)
    for item in niche_items:
        if item not in user:
            user[item] = random.randint(1, 2)
    
    return user

def generate_user_dataset(user_count=1000, clusters=10):
    user_ids = []
    names = []
    ratings = []

    categories = np.unique(df["main_category"])
    cluster_categories = np.random.choice(categories, size=clusters, replace=True) # new


    for i in tqdm.tqdm(range(user_count)):
        user_id = "user" + str(i)
        cluster_id = i % clusters # new 
        preferred_category = cluster_categories[cluster_id] # new

        # user_data = create_category_user(categories) # old
        user_data = create_category_user(
            categories=[preferred_category],  # limit category choices
            category_range=(4, 5),
            other_range=(1, 2)
        ) # new

        user_ids.extend([user_id] * len(user_data))
        names.extend(user_data.keys())
        ratings.extend(user_data.values())
        # print(f"done {user_id}")

    fake_df = pd.DataFrame({
        "user": user_ids,
        "name": names,
        "rating": ratings
    })
    return fake_df

df = pd.read_csv("data/models.csv")
global_items = get_global_items(df)
niche_items = get_niche_items(df)

if __name__ == "__main__":
    # users = generate_user_dataset()
    # users.to_csv("temp.csv")
    
    item_counts = df['name'].value_counts()
    print(item_counts)
    print(np.unique(item_counts))
    weights = df['name'].map(item_counts).fillna(1)
    print(np.unique(weights))