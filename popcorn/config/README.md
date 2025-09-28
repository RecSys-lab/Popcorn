# 🍿 Popcorn - Configurations

The first step to get started with Popcorn is setting up your configuration.
All the key parameters can be easily adjusted in the [`config.yml`](/popcorn/config/config.yml) file. Tweak them to match your experiment needs 🎛️!

| Category             | Sub-Category      | Inner-level Option    | Description                                                                 |
| -------------------- | ----------------- | --------------------- | --------------------------------------------------------------------------- |
| General              | `root_path`       | -                     | the root location of the framework (`Popcorn` for using in Google Colab)    |
| General              | `output_path`     | -                     | the output location of the framework for saving output data                 |
| General              | `verbose`         | -                     | the detailed logs flag                                                      |
| Dataset (Unimodal)   | `movielens`       | `name`                | the standard name of the dataset (mainly for logging)                       |
| Dataset (Unimodal)   | `movielens`       | `version`             | the demanded version (supported: `100k`, `1m`, `25m`)                       |
| Dataset (Unimodal)   | `movielens`       | `download_path`       | the root path to download the dataset (if exists, will be skipped)          |
| Dataset (Unimodal)   | `poison_rag_plus` | `name`                | the standard name of the dataset (mainly for logging)                       |
| Dataset (Unimodal)   | `poison_rag_plus` | `llm`                 | the demanded llm backbone (supported: `openai`, `llama`, `st`)              |
| Dataset (Unimodal)   | `poison_rag_plus` | `augmented`           | the boolean to choose original or augmented (enriched) variants             |
| Dataset (Unimodal)   | `poison_rag_plus` | `max_parts`           | the maximum number of textual embedding files parts                         |
| Dataset (Multimodal) | `popcorn`         | `name`                | the standard name of the dataset (mainly for logging)                       |
| Dataset (Multimodal) | `popcorn`         | `path_metadata`       | the path to the metadata json file of Popcorn dataset                       |
| Dataset (Multimodal) | `popcorn`         | `path_raw`            | the path to the raw packets of the dataset, containing visual features      |
| Dataset (Multimodal) | `popcorn`         | `feature_sources`     | features extracted from which **sources** should be used?                   |
| Dataset (Multimodal) | `popcorn`         | `feature_models`      | features extracted from which **models** should be used?                    |
| Dataset (Multimodal) | `popcorn`         | `agg_feature_sources` | aggregated features extracted from which **sources** should be used?        |
| Dataset (Multimodal) | `popcorn`         | `aggregation_models`  | which **aggregation** models should be used?                                |
| Pipeline             | `trailers`        | `mode`                | the path where found and downloaded trailers will be saved                  |
| Setup                | `seed`            | -                     | the seed for reproducibility purposes                                       |
| Setup                | `k_core`          | -                     | the number of cores for k-core filtering                                    |
| Setup                | `split`           | `mode`                | the train/test splitting mode (supported: `random`, `temporal`, `per_user`) |
| Setup                | `split`           | `test_ratio`          | the test ratio (between 0.1 to 1.0, otherwise is set to 0.2)                |

## III. Pipelines

It covers the pipelines designed for the framework, including the followings:

| Sub-Category                     | Options                   | Description                                                          |
| -------------------------------- | ------------------------- | -------------------------------------------------------------------- |
| `movie_frames`                   | `name`                    | the name of the pipeline to extract frames from videos               |
| `movie_frames`                   | `movies_path`             | the path of the videos to read from                                  |
| `movie_frames`                   | `frames_path`             | the path of the frames to be saved                                   |
| `movie_frames`                   | `video_formats`           | the supported video franes ["mp4", "avi", "mkv"]                     |
| `movie_frames`                   | `frequency`               | the frequency of frames extraction (picking 'n' frames every second) |
| `movie_frames`                   | `output_format`           | the output saved frames format (e.g., jpg)                           |
| `movie_frames`                   | `model_input_size`        | the input size (width) of the saved frame                            |
| `movie_frames_visual_features`   | `name`                    | the name of the pipeline to extract visual features from frames      |
| `movie_frames_visual_features`   | `frames_path`             | the path to the root directory containing the frames in folders      |
| `movie_frames_visual_features`   | `features_path`           | the generated output features path                                   |
| `movie_frames_visual_features`   | `image_formats`           | the supported image formats (["png", "jpg", "jpeg"])                 |
| `movie_frames_visual_features`   | `feature_extractor_model` | feature extraction models (pick from ["incp3", "vgg19"])             |
| `movie_frames_visual_features`   | `packet_size`             | the packets size (number of frames in each packet)                   |
| `movie_shots`                    | `name`                    | the name of the pipeline to extract shots from frames/features       |
| `movie_shots`                    | `variants`                | the variants of shot detection (from frame or from feature)          |
| `movie_shots`-`variants`         | `from_frames`             | parameters to extract shots from frames (image files)                |
| `movie_shots`-`variants`-`frame` | `frames_path`             | the path to read frames from                                         |
| `movie_shots`-`variants`-`frame` | `shot_frames_path`        | the output movie shots saved as images                               |
| `movie_shots`-`variants`-`frame` | `image_formats`           | the supported input frames format                                    |
| `movie_shots`-`variants`-`frame` | `output_format`           | the output frame format                                              |
| `movie_shots`-`variants`-`frame` | `threshold`               | the shot boundaries detection threshold                              |
| `movie_shots`-`variants`         | `from_features`           | parameters to extract shots from features (json files)               |
| `movie_shots`-`variants`-`feat`  | `features_path`           | the path to read features from                                       |
| `movie_shots`-`variants`-`feat`  | `shot_features_path`      | the output movie shots saved as json packets                         |
| `movie_shots`-`variants`-`feat`  | `threshold`               | the shot boundaries detection threshold                              |
| `movie_shots`-`variants`-`feat`  | `packet_size`             | the packets size (number of frames in each packet)                   |
