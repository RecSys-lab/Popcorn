# 🍿 Popcorn - Configurations

The first step to get started with Popcorn is setting up your configuration.
All the key parameters can be easily adjusted in the [`config.yml`](/popcorn/config/config.yml) file. Tweak them to match your experiment needs 🎛️!

| Category             | Sub-Category                        | Inner-level Option    | Description                                                                 |
| -------------------- | ----------------------------------- | --------------------- | --------------------------------------------------------------------------- |
| General              | `root_path`                         | -                     | the root location of the framework (`Popcorn` for using in Google Colab)    |
| General              | `output_path`                       | -                     | the output location of the framework for saving output data                 |
| General              | `verbose`                           | -                     | the detailed logs flag                                                      |
| Dataset (Unimodal)   | `movielens`                         | `name`                | the standard name of the dataset (mainly for logging)                       |
| Dataset (Unimodal)   | `movielens`                         | `version`             | the demanded version (supported: `100k`, `1m`, `25m`)                       |
| Dataset (Unimodal)   | `movielens`                         | `download_path`       | the root path to download the dataset (if exists, will be skipped)          |
| Dataset (Unimodal)   | `poison_rag_plus`                   | `name`                | the standard name of the dataset (mainly for logging)                       |
| Dataset (Unimodal)   | `poison_rag_plus`                   | `llm`                 | the demanded llm backbone (supported: `openai`, `llama`, `st`)              |
| Dataset (Unimodal)   | `poison_rag_plus`                   | `augmented`           | the boolean to choose original or augmented (enriched) variants             |
| Dataset (Unimodal)   | `poison_rag_plus`                   | `max_parts`           | the maximum number of textual embedding files parts                         |
| Dataset (Multimodal) | `popcorn`                           | `name`                | the standard name of the dataset (mainly for logging)                       |
| Dataset (Multimodal) | `popcorn`                           | `path_metadata`       | the path to the metadata json file of Popcorn dataset                       |
| Dataset (Multimodal) | `popcorn`                           | `path_raw`            | the path to the raw packets of the dataset, containing visual features      |
| Dataset (Multimodal) | `popcorn`                           | `feature_sources`     | features extracted from which **sources** should be used?                   |
| Dataset (Multimodal) | `popcorn`                           | `feature_models`      | features extracted from which **models** should be used?                    |
| Dataset (Multimodal) | `popcorn`                           | `agg_feature_sources` | aggregated features extracted from which **sources** should be used?        |
| Pipeline             | `trailer_fetch`                     | `download_path`       | the path to downloaded trailers saved                                       |
| Pipeline             | `frame_extractor`                   | `movies_path`         | the path to the movies directory                                            |
| Pipeline             | `frame_extractor`                   | `frames_path`         | the path to save the extracted frames                                       |
| Pipeline             | `frame_extractor`                   | `frame_format`        | the format of the extracted frames (e.g., jpg, png)                         |
| Pipeline             | `frame_extractor`                   | `frequency`           | the frequency of frames extraction (picking 'n' frames every second)        |
| Pipeline             | `frame_extractor`                   | `frame_width`         | the width of the extracted frames (in pixels)                               |
| Pipeline             | `visual_embedding_extractor`        | `frames_path`         | the path to the frames directory                                            |
| Pipeline             | `visual_embedding_extractor`        | `features_path`       | the path to save the extracted visual features                              |
| Pipeline             | `visual_embedding_extractor`        | `cnn`                 | the CNN model to use for feature extraction                                 |
| Pipeline             | `visual_embedding_extractor`        | `packet_size`         | the packets size (number of frames in each packet)                          |
| Pipeline             | `visual_embedding_aggregator`       | `features_path`       | the path to the extracted visual features                                   |
| Pipeline             | `visual_embedding_aggregator`       | `agg_features_path`   | the path to save the aggregated visual features                             |
| Pipeline             | `visual_embedding_aggregator`       | `aggregation_methods` | the aggregation model to use (supported: `Max`, `Mean`)                     |
| Pipeline             | `shot_detector`                     | `shots_path`          | the path to save the detected shots (frames or embeddings)                  |
| Pipeline             | `shot_detector`                     | `threshold`           | the shot boundaries detection threshold (between 0.1 to 1.0, default: 0.7)  |
| Pipeline             | `shot_detector` / `from_frames`     | `frames_path`         | the path to the extracted movie frames                                      |
| Pipeline             | `shot_detector` / `from_frames`     | `frame_format`        | the format of the extracted frames (e.g., jpg, png)                         |
| Pipeline             | `shot_detector` / `from_embeddings` | `features_path`       | the path to the extracted movie visual embeddings                           |
| Pipeline             | `shot_detector` / `from_embeddings` | `packet_size`         | the packets size (number of frames in each packet, between 1 and 50)        |
| Modalities           | `output_path`                       | -                     | the root path to save fused datasets                                        |
| Modalities           | `selected`                          | -                     | the list of selected modalities to be used                                  |
| Modalities           | `fusion_methods`                    | `selected`            | the list of fusion methods to be used                                       |
| Modalities           | `fusion_methods`                    | `cca_components`      | the CCA components (only for 'cca' fusion method)                           |
| Modalities           | `fusion_methods`                    | `pca_variance`        | the PCA variance to retain (only for 'pca' fusion method, between 0 and 1)  |
| Setup                | `seed`                              | -                     | the seed for reproducibility purposes                                       |
| Setup                | `k_core`                            | -                     | the number of cores for k-core filtering                                    |
| Setup                | `split`                             | `mode`                | the train/test splitting mode (supported: `random`, `temporal`, `per_user`) |
| Setup                | `split`                             | `test_ratio`          | the test ratio (between 0.1 to 1.0, otherwise is set to 0.2)                |
