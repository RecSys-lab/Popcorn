# import json
# import openai
# import inflect


# def method_A_generation(
#     user_profile,
#     ranked_item_list,
#     df_items=None,
#     llm_model="gpt-4o-mini",
#     api_key=None,
#     top_N_retrieval=50,
#     top_k_rec=10,
#     debug=True,
#     return_prompt=False,
#     chain_of_thought=False
# ):
#     """
#     Takes a user profile + a previously ranked list of items,
#     asks the LLM to produce a final recommendation in JSON.

#     Behavior controlled by `chain_of_thought`:
#       - If chain_of_thought=True:
#           The final JSON must include:
#              "recommended_item_ids"
#              "explanation"
#              "chain_of_thought"
#       - If chain_of_thought=False:
#           The final JSON must ONLY have:
#              "recommended_item_ids"
#           (No explanation of any kind.)

#     Parameters
#     ----------
#     user_profile : dict
#         Contains fields like "user_id", "overall_taste_summary", etc.

#     ranked_item_list : list
#         A pre-ranked list of items, e.g. [(itemId, distance), ...]

#     df_items : pd.DataFrame or None
#         If provided, used to look up 'title' or other fields.

#     llm_model : str, default="gpt-4o-mini"
#         OpenAI model name.

#     api_key : str, optional
#         OpenAI API key. Required if calling the OpenAI API.

#     top_k : int, default=5
#         Number of items to present to the LLM as candidates.

#     debug : bool, default=True
#         Whether to print debug information.

#     return_prompt : bool, default=False
#         If True, returns the final prompt structure in the output dict.

#     chain_of_thought : bool, default=False
#         - If True, the LLM must return a "chain_of_thought" AND an "explanation".
#         - If False, no explanation or chain-of-thought is returned at all.

#     Returns
#     -------
#     dict
#         {
#             "model_output": <raw LLM text>,
#             "parsed_json": { ... final JSON as a Python dict ... },
#             "prompt": { ... }  # only if return_prompt=True
#         }
#     """

#     top_N_retrieval = min(top_N_retrieval, len(ranked_item_list))

#     # Create an inflect engine
#     p = inflect.engine()

#     # Convert the number to words
#     top_k_rec_text = p.number_to_words(top_k_rec).upper()
#     #print(top_k_rec_text)

#     if api_key is None:
#         raise ValueError("An OpenAI API key is required.")
#     openai.api_key = api_key

#     # 1) Slice the top-ranked items we will feed to the LLM
#     top_items = ranked_item_list[:top_N_retrieval]

#     # 2) Build a text snippet describing the items
#     items_text = ""
#     for entry in top_items:
#         if isinstance(entry, tuple) and len(entry) == 2:
#             item_id, score = entry
#         else:
#             item_id = entry
#             score = None

#         # Optionally fetch the title from df_items
#         title_str = f"Item {item_id}"
#         if df_items is not None:
#             row = df_items[df_items["itemId"] == item_id]
#             if not row.empty:
#                 title_val = row.iloc[0].get("title", "")
#                 if title_val:
#                     title_str = title_val

#         items_text += f"- ID: {item_id}, Title: {title_str}"
#         if score is not None:
#             items_text += f", Score: {score:.4f}"
#         items_text += "\n"

#     # 3) Create system & user messages,
#     #    based on whether chain_of_thought is True/False.
#     if chain_of_thought:
#         # Prompt that REQUIRES explanation + chain_of_thought in final JSON
#         system_msg = (
#             "You are a helpful recommendation assistant. "
#             "You have a user profile plus some candidate items. "
#             f"You MUST recommend at least {top_k_rec_text} items from the provided list. "
#             "Output must be valid JSON with all of these keys:\n"
#             "  'recommended_item_ids': array of at least 5 item IDs\n"
#             "  'explanation': a short summary\n"
#             "  'chain_of_thought': your step-by-step reasoning\n"
#             "For example:\n"
#             "{\n"
#             "  \"recommended_item_ids\": [111, 222, 333, 444, 555],\n"
#             "  \"explanation\": \"short reasoning...\",\n"
#             "  \"chain_of_thought\": \"step by step reasoning...\"\n"
#             "}\n"
#             "Make sure the JSON is valid and includes these exact keys."
#         )
#     else:
#         # Prompt that says ONLY produce recommended_item_ids
#         system_msg = (
#             "You are a helpful recommendation assistant. "
#             "You have a user profile plus some candidate items. "
#             f"You MUST recommend at least {top_k_rec_text} items from the provided list. "
#             "Output must be valid JSON with ONLY one key:\n"
#             "  'recommended_item_ids': array of at least 5 item IDs\n"
#             "Do NOT include any explanation or chain-of-thought.\n"
#             "For example:\n"
#             "{\n"
#             "  \"recommended_item_ids\": [111, 222, 333, 444, 555]\n"
#             "}\n"
#             "Make sure the JSON is valid, with no extra keys."
#         )

#     user_msg = f"""
# User Profile for user {user_profile.get('user_id')}:
# - Favorite Genres: {user_profile.get('favorite_genres')}
# - Favorite Tags: {user_profile.get('favorite_tags')}
# - Average Rating: {user_profile.get('average_rating')}
# - Top Items: {user_profile.get('top_items')}

# Overall Taste Summary:
# "{user_profile.get('overall_taste_summary')}"

# Candidate items to potentially recommend (already pre-ranked):
# {items_text}

# Please provide your final JSON as requested above.
# """

#     if debug:
#         print("=== DEBUG: System Message ===")
#         print(system_msg)
#         print("=== DEBUG: User Message ===")
#         print(user_msg)

#     # Keep the prompt in case we want to return it
#     prompt = {"system_message": system_msg, "user_message": user_msg}

#     # 4) Call the ChatCompletion
#     try:
#         response = openai.ChatCompletion.create(
#             model=llm_model,
#             messages=[
#                 {"role": "system", "content": system_msg},
#                 {"role": "user", "content": user_msg},
#             ],
#             max_tokens=1000,
#             temperature=0.7
#         )
#         model_output = response["choices"][0]["message"]["content"].strip()
#     except Exception as e:
#         print(f"Error calling the LLM: {e}")
#         return {
#             "model_output": "",
#             "parsed_json": {},
#             **({"prompt": prompt} if return_prompt else {})
#         }

#     if debug:
#         print("\n=== DEBUG: Raw Model Output ===")
#         print(model_output)

#     # 5) Parse JSON from model output
#     parsed_json = {}
#     try:
#         first_brace = model_output.find("{")
#         last_brace = model_output.rfind("}")
#         if first_brace != -1 and last_brace != -1:
#             json_str = model_output[first_brace:last_brace + 1]
#             parsed_json = json.loads(json_str)
#     except Exception as e:
#         print(f"JSON parsing error: {e}")
#         parsed_json = {}

#     # 6) If "recommended_item_ids" is empty or missing, fallback to top-3
#     if "recommended_item_ids" not in parsed_json or not parsed_json["recommended_item_ids"]:
#         fallback_item_ids = []
#         for entry in top_items:
#             if isinstance(entry, tuple):
#                 fallback_item_ids.append(entry[0])
#             else:
#                 fallback_item_ids.append(entry)
#         fallback_item_ids = fallback_item_ids[:top_k_rec]
#         parsed_json["recommended_item_ids"] = fallback_item_ids

#         # If chain_of_thought=True, fill them in blank if missing
#         # Otherwise skip them entirely if chain_of_thought=False
#         if chain_of_thought:
#             if "explanation" not in parsed_json:
#                 parsed_json["explanation"] = "LLM did not provide an explanation."
#             if "chain_of_thought" not in parsed_json:
#                 parsed_json["chain_of_thought"] = "LLM did not provide chain-of-thought."

#     # 7) If chain_of_thought=False, ensure we do NOT leak "explanation" or "chain_of_thought"
#     #    in case the model incorrectly included them.
#     if not chain_of_thought:
#         keys_to_remove = ["explanation", "chain_of_thought"]
#         for key in keys_to_remove:
#             if key in parsed_json:
#                 del parsed_json[key]

#     # Return final result
#     result = {
#         "model_output": model_output,
#         "parsed_json": parsed_json,
#     }
#     if return_prompt:
#         result["prompt"] = prompt

#     return result
