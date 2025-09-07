# URLs for downloading MovieLens datasets
ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
ML25M_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

# Genres in MovieLens datasets
mainGenres = ["Action", "Comedy", "Drama", "Horror"]
allGenres = [
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]

# Columns for MovieLens datasets (for 100k, the structure is different)
itemCols = ["item_id", "title", "genres"]
itemCols_100k = [
    "item_id",
    "title",
    "release_date",
    "video_release_date",
    "IMDb_URL",
] + allGenres

userCols = ["user_id", "age", "gender", "occupation", "zip_code"]

ratingCols = ["user_id", "item_id", "rating", "timestamp"]
