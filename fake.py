import numpy as np
import pandas as pd
import random
import tqdm

# GLOBAL POPULAR/UNIVERSAL ITEMS (reuse these across all users)
def get_global_items(df: pd.DataFrame, n=10):
    return df.sort_values("no_of_ratings", ascending=False).head(n)

# NICHE ITEMS (least popular in the dataset)
def get_niche_items(df: pd.DataFrame, n=10):
    return df.sort_values("no_of_ratings", ascending=True).head(n)

# Biased sampling function to reuse popular items more
def sample_with_popularity(df: pd.DataFrame, n):
    weights = df["no_of_ratings"].fillna(1)
    if len(df) < n:
        return df.sample(n=n, weights=weights, replace=True)
    else:
        return df.sample(n=n, weights=weights, replace=False)

def create_category_user(df: pd.DataFrame, categories, global_items: pd.DataFrame, niche_items: pd.DataFrame, item_count=100, category_range=(4,5), other_range=(1,3)):
    name = []
    rating = []
    main_category = []
    sub_category = []

    # pick random category
    category = random.choice(categories)

    # sample
    df_cat: pd.DataFrame = df[df["main_category"] == category]
    cat_sample = sample_with_popularity(df_cat, item_count // 2)
    df_other = df[~df["name"].isin(cat_sample["name"])]
    other_sample = sample_with_popularity(df_other, item_count // 2)

    # ratings
    cat_ratings = np.random.randint(category_range[0], category_range[1] + 1, size=cat_sample.shape[0])
    other_ratings = np.random.randint(other_range[0], other_range[1] + 1, size=other_sample.shape[0])

    # update user
    name.extend(cat_sample["name"])
    rating.extend(cat_ratings)
    main_category.extend(cat_sample["main_category"])
    sub_category.extend(cat_sample["sub_category"])
    name.extend(other_sample["name"])
    rating.extend(other_ratings)
    main_category.extend(other_sample["main_category"])
    sub_category.extend(other_sample["sub_category"])

    # Add global items (reuse for all users)
    for _, row in global_items.iterrows():
        if row["name"] not in name:
            name.append(row["name"])
            rating.append(random.randint(3, 5))
            main_category.append(row["main_category"])
            sub_category.append(row["sub_category"])

    # Add a few niche items (per user, low ratings)
    niche_sample = niche_items.sample(n=min(5, len(niche_items)))  # choose a few
    for _, row in niche_sample.iterrows():
        if row["name"] not in name:
            name.append(row["name"])
            rating.append(random.randint(1, 2))
            main_category.append(row["main_category"])
            sub_category.append(row["sub_category"])

    return name, rating, main_category, sub_category

def generate_user_dataset(df: pd.DataFrame, user_count=1000, clusters=10):
    user_ids = []
    names = []
    ratings = []
    main_categories = []
    sub_categories = []

    global_items = get_global_items(df)
    niche_items = get_niche_items(df)

    categories = np.unique(df["main_category"])
    cluster_categories = np.random.choice(categories, size=clusters, replace=True) # new

    for i in tqdm.tqdm(range(user_count)):
        user_id = "user" + str(i)
        cluster_id = i % clusters # new 
        preferred_category = cluster_categories[cluster_id] # new

        name, rating, main_category, sub_category = create_category_user(
            df,
            categories=[preferred_category],  # limit category choices
            global_items=global_items,
            niche_items=niche_items,
            category_range=(4, 5),
            other_range=(1, 2)
        ) # new

        user_ids.extend([user_id] * len(name))
        names.extend(name)
        ratings.extend(rating)
        main_categories.extend(main_category)
        sub_categories.extend(sub_category)

    fake_df = pd.DataFrame({
        "user": user_ids,
        "name": names,
        "rating": ratings,
        "main_category": main_categories,
        "sub_category": sub_categories,
    })
    return fake_df

if __name__ == "__main__":
    df = pd.read_csv("data/models.csv")

    users = generate_user_dataset(df, 100)
    users.to_csv("temp.csv")