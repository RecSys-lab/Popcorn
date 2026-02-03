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