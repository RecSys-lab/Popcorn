# MoViFex - Configurations

The first recommended step to use the framework is setting the configuration parameters. You can find below the configurations modifiable in [`config.yml`](/movifex/config/config.yml):

## I. General

It covers the general configurations of the system, including the followings:

- **mode**: what is the expected functionality of the framework? It can be `ds` to run dataset-related modules, `pipeline` to run an available pipeline, or `recsys` to run the recommender system.
- **sub_mode_ds**: if `ds` is selected as the `mode`, which dataset-related functionality is expected? Choose from `movifex_meta` to process based on **MoViFex**'s metadata, or `movifex_visual` to process its visual features.
- **sub_mode_pipeline**: if `pipeline` is selected as the `mode`, which pipeline functionality is desired? Choose from `dl_trailers` to download trailers for a list of given movies from YouTube, `frame_extractor` to extract frames of a video, `feat_extractor` to extract features of a frame, `shot_from_frame` to extract shots from a set of frames, or `shot_from_feat` to extract shots from features.
- **sub_mode_recsys**: if `recsys` is selected as the `mode`, which recommender system is expected?

## II. Datasets

It covers the datasets usable in the framework, including the followings:

| Sub-Category     | Options           | Description                                                                       |
| ---------------- | ----------------- | --------------------------------------------------------------------------------- |
| `unimodal_dataset`   | `name`            | the name of the text dataset (e.g., `movielens-25m`)                              |
| `unimodal_dataset`   | `need_download`   | the flag to check whether to download the text dataset or read from file          |
| `unimodal_dataset`   | `url`             | the url of the text dataset                                                       |
| `unimodal_dataset`   | `download_path`   | the path to download the text dataset                                             |
| `multimodal_dataset` | `name`            | the name of the visual dataset (e.g., `MoViFex`)                                  |
| `multimodal_dataset` | `url`             | the url of the visual dataset                                                     |
| `multimodal_dataset` | `path_metadata`   | the path to the metadata json file of SceneNet dataset                            |
| `multimodal_dataset` | `path_raw`        | the path to the raw packets of the dataset, containing visual features            |
| `multimodal_dataset` | `feature_sources` | features extracted from which **sources** should be used? (e.g., [`full_movies`]) |
| `multimodal_dataset` | `feature_models`  | features extracted from which **models** should be used? (e.g., [`vgg19`])        |

## III. Pipelines

It covers the pipelines designed for the framework, including the followings:

| Sub-Category                     | Options                   | Description                                                          |
| -------------------------------- | ------------------------- | -------------------------------------------------------------------- |
| `movie_trailers`                 | `name`                    | the name of the pipeline to download trailers                        |
| `movie_trailers`                 | `download_path`           | the path in which downloaded files will be saved                     |
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
