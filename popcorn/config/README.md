# 🍿 Popcorn - Configurations

The first step to get started with Popcorn is setting up your configuration.
All the key parameters can be easily adjusted in the [`config.yml`](/popcorn/config/config.yml) file. Tweak them to match your experiment needs 🎛️!

| Category | Sub-Category | Inner-level Option | Description                                                              |
| -------- | ------------ | ------------------ | ------------------------------------------------------------------------ |
| General  | `root_path`  | -                  | the root location of the framework (`Popcorn` for using in Google Colab) |

## II. Datasets

It covers the datasets usable in the framework, including the followings:

| Sub-Category         | Options           | Description                                                                       |
| -------------------- | ----------------- | --------------------------------------------------------------------------------- |
| `unimodal`   | `name`            | the name of the text dataset (e.g., `movielens-25m`)                              |
| `unimodal`   | `url`             | the url of the text dataset                                                       |
| `unimodal`   | `download_path`   | the path to download the text dataset                                             |
| `multimodal` | `name`            | the name of the visual dataset (e.g., `MoViFex`)                                  |
| `multimodal` | `url`             | the url of the visual dataset                                                     |
| `multimodal` | `path_metadata`   | the path to the metadata json file of SceneNet dataset                            |
| `multimodal` | `path_raw`        | the path to the raw packets of the dataset, containing visual features            |
| `multimodal` | `feature_sources` | features extracted from which **sources** should be used? (e.g., [`full_movies`]) |
| `multimodal` | `feature_models`  | features extracted from which **models** should be used? (e.g., [`vgg19`])        |

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
