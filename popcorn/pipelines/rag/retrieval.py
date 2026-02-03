# import numpy as np
# from sklearn.impute import KNNImputer
# from sklearn.neighbors import NearestNeighbors

# #########################
# # Part 3: Retrieval
# #########################

# def build_item_matrix(embedding_dict):
#     """
#     Convert embedding_dict -> (item_matrix, item_ids).
#     item_matrix shape: (num_items, emb_dim)
#     """
#     item_ids = list(embedding_dict.keys())
#     item_matrix = np.array([embedding_dict[i] for i in item_ids], dtype='float32')
#     return item_matrix, item_ids


# def retrieve_top_N_items(user_emb, item_matrix, item_ids, N=5):
#     """
#     Uses scikit-learn's NearestNeighbors for retrieval with cosine distance.
#     Imputes missing values using KNNImputer (n_neighbors=3) before fitting.
#     Returns a list of (item_id, distance), sorted by ascending distance.
#     """
#     # 1) KNN-based imputation of NaN values in item_matrix and user_emb
#     imputer = KNNImputer(n_neighbors=3)

#     # Fit on the item_matrix (which can contain multiple items)
#     item_matrix_imputed = imputer.fit_transform(item_matrix)

#     # Transform user_emb (reshape to 2D first so imputer can handle a single row)
#     user_emb_imputed = imputer.transform(user_emb.reshape(1, -1))[0]

#     # 2) Fit NearestNeighbors on the imputed item_matrix
#     nn = NearestNeighbors(n_neighbors=N, metric='cosine')
#     nn.fit(item_matrix_imputed)

#     # 3) Retrieve neighbors for user_emb_imputed
#     distances, indices = nn.kneighbors([user_emb_imputed])

#     # 4) Build and return sorted results [(item_id, distance), ...]
#     results = [(item_ids[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
#     return sorted(results, key=lambda x: x[1])

# #########################
# # Part 4: Visualization
# #########################

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from sklearn.neighbors import NearestNeighbors

# def visualize_embeddings_2d(user_embeddings_dict, embedding_dict, item_ids_subset, fileName):
#     """
#     Create three 2D scatter plots showing the user embeddings (random, average, temporal)
#     and zoom in for the average and temporal embeddings.
#     """
#     # Extract item embeddings and labels for the selected items
#     item_embs = []
#     item_labels = []
#     for iid in item_ids_subset:
#         if iid in embedding_dict:
#             item_embs.append(embedding_dict[iid])
#             item_labels.append(str(iid))

#     item_embs = np.array(item_embs)

#     if item_embs.shape[1] < 2:
#         print("Item embeddings have fewer than 2 dimensions; cannot plot in 2D.")
#         return

#     # Restrict to first two dimensions for plotting
#     item_embs_2d = item_embs[:, :2]
#     user_embs_2d = {method: emb[:2] for method, emb in user_embeddings_dict.items()}

#     # Create 3 plots: Random, Average, Temporal + 1 Zoomed plot for Average & Temporal
#     fig, axes = plt.subplots(1, 4, figsize=(24, 6), gridspec_kw={'width_ratios': [1, 1, 1, 1.5]})
#     methods = list(user_embeddings_dict.keys())

#     # Plot individual methods
#     for i, method in enumerate(methods):
#         ax = axes[i]
#         user_emb = user_embs_2d[method]

#         # Scatter plot of items
#         ax.scatter(item_embs_2d[:, 0], item_embs_2d[:, 1], c='b', label='Items', alpha=0.7)
#         for j, label in enumerate(item_labels):
#             ax.text(item_embs_2d[j, 0], item_embs_2d[j, 1], f' {label}', fontsize=8, color='blue')

#         # Plot user embedding
#         ax.scatter(user_emb[0], user_emb[1], c='r', marker='X', s=150, label=f'User ({method})')
#         ax.text(user_emb[0], user_emb[1], f' User', fontsize=10, color='red')

#         # Titles and labels
#         ax.set_title(f'User Embedding: {method}', fontsize=12)
#         ax.set_xlabel('Dim 1')
#         ax.set_ylabel('Dim 2')
#         ax.legend(loc='best')

#     # Plot Zoomed-in View for Average and Temporal
#     ax_zoom = axes[3]
#     zoom_methods = ['average', 'temporal']
#     for method in zoom_methods:
#         user_emb = user_embs_2d[method]
#         ax_zoom.scatter(user_emb[0], user_emb[1], label=f'User ({method})', s=150, marker='X')
#         ax_zoom.text(user_emb[0], user_emb[1], f' User ({method})', fontsize=10)

#     # Add nearby item embeddings to zoomed plot
#     zoom_threshold = 0.05  # Distance threshold for nearby items
#     for method in zoom_methods:
#         user_emb = user_embs_2d[method]
#         distances = np.linalg.norm(item_embs_2d - user_emb, axis=1)
#         nearby_items = item_embs_2d[distances < zoom_threshold]
#         ax_zoom.scatter(nearby_items[:, 0], nearby_items[:, 1], label=f'Nearby Items ({method})', alpha=0.7)

#     ax_zoom.set_title("Zoomed-In View (Average & Temporal)", fontsize=12)
#     ax_zoom.set_xlabel('Dim 1')
#     ax_zoom.set_ylabel('Dim 2')
#     ax_zoom.legend(loc='best')

#     plt.tight_layout()

#     # Save as PDF
#     plt.savefig(f"{fileName}.pdf", format="pdf", bbox_inches='tight')
#     # Download
#     files.download(f"{fileName}.pdf")

#     plt.show()