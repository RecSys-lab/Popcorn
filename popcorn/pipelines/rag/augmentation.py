# import openai
# import pandas as pd
# import numpy as np
# import time

# def augment_user_profile(
#     user_id,
#     df_ratings,
#     df_items,
#     method="manual",
#     llm_model="gpt-4o-mini",
#     api_key=None,
#     top_genre_limit=5,  # Limit top genres to the most frequent 5
#     top_tag_limit=15    # Limit top tags to the most frequent 15
# ):
#     """
#     Build a structured user profile for a single user, limiting the top genres
#     and tags to specified numbers.

#     Parameters:
#     -----------
#     - user_id: int
#         The user we are analyzing.

#     - df_ratings: pd.DataFrame
#         Contains [userId, itemId, rating, timestamp] at minimum.

#     - df_items: pd.DataFrame
#         Contains [itemId, title, genres, tag, ...].

#     - method: str, default="manual"
#         - "manual": Summarize user tastes in code.
#         - "llm": Summarize user tastes using LLM.

#     - llm_model: str, default="gpt-4o-mini"
#         The OpenAI model name to use if method="llm".

#     - api_key: str or None
#         The OpenAI API key. Required if method="llm".

#     - top_genre_limit: int, default=5
#         The number of top genres to include in the user profile.

#     - top_tag_limit: int, default=15
#         The number of top tags to include in the user profile.

#     Returns:
#     --------
#     - user_profile: dict
#         A structured user profile containing fields like:
#         {
#           "user_id": ...,
#           "favorite_genres": [...],
#           "favorite_tags": [...],
#           "top_items": [
#              { "itemId": ..., "title": "..." }, ...
#           ],
#           "average_rating": ...,
#           "overall_taste_summary": "..."
#         }
#     """

#     # Filter only the user’s ratings
#     user_df = df_ratings[df_ratings["userId"] == user_id].copy()
#     if user_df.empty:
#         return {
#             "user_id": user_id,
#             "favorite_genres": [],
#             "favorite_tags": [],
#             "top_items": [],
#             "average_rating": 0.0,
#             "overall_taste_summary": "No data"
#         }

#     # Merge with item info
#     user_merged = user_df.merge(df_items, on="itemId", how="left")

#     # Compute average rating
#     avg_rating = user_merged["rating"].mean()

#     # Sort items by rating desc
#     top_items = user_merged.sort_values("rating", ascending=False).head(5)

#     # Collect top item info
#     top_items_list = []
#     for _, row in top_items.iterrows():
#         top_items_list.append({
#             "itemId": int(row["itemId"]),
#             "title": str(row.get("title", "")),
#             "rating": float(row["rating"])
#         })

#     # For genres
#     if "genres" in user_merged.columns:
#         genre_counts = {}
#         for _, row in user_merged.iterrows():
#             genres_str = row.get("genres", "")
#             if not isinstance(genres_str, str):
#                 continue
#             genres_list = genres_str.split("|")  # Adjust delimiter as necessary
#             for g in genres_list:
#                 g = g.strip()
#                 if g:
#                     genre_counts[g] = genre_counts.get(g, 0) + 1
#         # Sort by frequency and limit to the top genres
#         favorite_genres = sorted(genre_counts.keys(), key=lambda g: genre_counts[g], reverse=True)[:top_genre_limit]
#     else:
#         favorite_genres = []

#     # For tags
#     if "tag" in user_merged.columns:
#         tag_counts = {}
#         for _, row in user_merged.iterrows():
#             tags_value = row.get("tag", [])
#             if not isinstance(tags_value, list):
#                 continue
#             for t in tags_value:
#                 t = t.strip()
#                 if t:
#                     tag_counts[t] = tag_counts.get(t, 0) + 1
#         # Sort by frequency and limit to the top tags
#         favorite_tags = sorted(tag_counts.keys(), key=lambda t: tag_counts[t], reverse=True)[:top_tag_limit]
#     else:
#         favorite_tags = []

#     # Prepare a basic "manual" summary
#     if method == "manual":
#         overall_taste_summary = (
#             f"This user has rated {len(user_df)} items with an average rating of {avg_rating:.2f}. "
#             f"The top genres seem to be {favorite_genres}. "
#             f"Frequent tags: {favorite_tags}. "
#             "They appear to enjoy these top 5 items: " +
#             ", ".join([x["title"] for x in top_items_list]) + "."
#         )
#     else:
#         # method == "llm"
#         if api_key is None:
#             raise ValueError("api_key is required for LLM-based user profile augmentation.")
#         openai.api_key = api_key

#         raw_text = f"""
# User {user_id} has the following data:
# - Average rating: {avg_rating:.2f}
# - Favorite genres (by frequency): {favorite_genres}
# - Favorite tags (by frequency): {favorite_tags}
# - Top 5 highest rated items:
# """

#         for x in top_items_list:
#             raw_text += f"  - {x['title']} (rating {x['rating']:.1f})\n"

#         system_msg = (
#             "You are a helpful assistant that summarizes a user's taste in a short paragraph. "
#             "Focus on the user's overall preferences and style, then produce a single summary string."
#         )
#         user_msg = (
#             "Here is the raw data for a user's preferences:\n"
#             f"{raw_text}\n\n"
#             "Please write a concise summary of this user's overall taste. Keep it short but meaningful."
#         )

#         try:
#             response = openai.ChatCompletion.create(
#                 model=llm_model,
#                 messages=[
#                     {"role": "system", "content": system_msg},
#                     {"role": "user", "content": user_msg}
#                 ],
#                 max_tokens=200,
#                 temperature=0.7
#             )
#             overall_taste_summary = response["choices"][0]["message"]["content"].strip()
#         except Exception as e:
#             print(f"LLM error for user {user_id}: {e}")
#             overall_taste_summary = "LLM summarization failed."

#     # Assemble the structured user profile
#     user_profile = {
#         "user_id": int(user_id),
#         "favorite_genres": favorite_genres,
#         "favorite_tags": favorite_tags,
#         "top_items": top_items_list,
#         "average_rating": float(avg_rating),
#         "overall_taste_summary": overall_taste_summary
#     }

#     return user_profile


# profile_manual = augment_user_profile(
#     user_id=1,
#     df_ratings=df_ratings,
#     df_items=df_items,
#     method="manual",
#     top_genre_limit=5,  # Top 5 genres
#     top_tag_limit=15    # Top 15 tags
# )
# display("Manual profile:", profile_manual)


# profile_llm = augment_user_profile(
#     user_id=1,
#     df_ratings=df_ratings,
#     df_items=df_items,
#     method="llm",
#     llm_model="gpt-4o-mini",
#     api_key=os.environ.get("OPENAI_API_KEY"),
#     top_genre_limit=5,  # Top 5 genres
#     top_tag_limit=15    # Top 15 tags
# )
# display("LLM profile:", profile_llm)

