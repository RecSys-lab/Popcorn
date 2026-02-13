# 🚀 Quick Start

Popcorn exposes a minimal, end-to-end workflow from configuration to exported artifacts.  
A reproduction notebook is also provided in the repository:

👉 https://github.com/RecSys-lab/Popcorn/blob/main/examples/colab/popcorn_tool.ipynb

A typical run:

```bash
python examples/python/quick_test.py
```

The preset shows a simple workflow for loading Popcorn’s abstracted modules:

```python
from popcorn.utils import readConfigs
from popcorn.optimizers.grid_search import gridSearch
from popcorn.recommenders.reclist import generateLists
from popcorn.recommenders.assembler import assembleModality

# Step-0: Read the configuration file
configs = readConfigs("popcorn/config/config.yml")

# Step-1: Data ingestion and modality assembly
trainDF, testDF, trainSet, modalitiesDict, genreDict = assembleModality(configs)

# Step-2: Apply grid search to find the best model configurations
finalModels = gridSearch(configs, trainDF, trainSet, modalitiesDict)

# Step-3: Generate recommendation lists
generateLists(configs, trainDF, trainSet, testDF, genreDict, finalModels)
```

Upon successful completion, the framework exports reclist.csv and metrics.csv, enabling comparable evaluation across evidence sources, granularities, fusion methods, and GenAI settings:

```bash
| Model | Scenario | NDCG@2 | Novelty | Recall@2 | Diversity | ... |
|-------|----------|--------|---------|----------|-----------|-----|
| VBPR  | text     | 0.421  | 11.819  | 0.558    | 0.588     | ... |
| VBPR  | visual   | 0.421  | 11.819  | 0.558    | 0.588     | ... |
| VBPR  | cca_40   | 0.141  | 12.272  | 0.117    | 0.635     | ... |
| VBPR  | pca_40   | 0.154  | 12.144  | 0.131    | 0.709     | ... |
```

A reproduction notebook is also provided in the repository:

[https://github.com/RecSys-lab/Popcorn/blob/main/examples/colab/popcorn_tool.ipynb](https://github.com/RecSys-lab/Popcorn/blob/main/examples/colab/popcorn_tool.ipynb)