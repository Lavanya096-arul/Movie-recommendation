import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "tmdb/tmdb-movie-metadata",
    "tmdb_5000_movies.csv",
    pandas_kwargs={
        "encoding": "latin1",
        "engine": "python",
        "sep": ",",
        "quotechar": '"',
        "on_bad_lines": "skip"
    }
)

print("Shape:", df.shape)
print(df.head())





# Missing values per column
print(df.isna().sum())

# Drop completely empty rows (if any)
df = df.dropna(how='all')

# Reset index
df.reset_index(drop=True, inplace=True)

# ----------------------------------------------------------

  # convert string representation of list/dict to real list/dict

def parse_genres(genres_str):
    try:
        genres_list = ast.literal_eval(genres_str)
        return [g['name'] for g in genres_list]
    except:
        return []

df['genres'] = df['genres'].apply(parse_genres)
