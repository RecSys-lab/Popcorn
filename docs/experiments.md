# 🔎 Experiments

## **I. Best-Performing Results**

- **Settings:** `n_epochs=10`, `test_ratio=0.2`, `k_core=10`, `top_n ∈ {2,10}`  
- **Visual**: Popcorn-Visual (Inception-v3 + max pooling)  
- **Text**: Text-RAG-Plus (`text_aug ∈ {true,false}`)
- **MovieLens**: `1M` version

---

### 🧩 **n = 2**

| **Method** | **Modality** | **Params** | **NDCG (Trailer)** | **Recall (Trailer)** | **NDCG (Full Movie)** | **Recall (Full Movie)** |
|--------|----------|--------|--------|----------|--------|----------|
| VBPR | Text | llm=llama, text_aug=false | 0.222 | 0.195 | 0.224 | 0.194 |
| VBPR | Visual | cnn=inception3 | 0.206 | 0.177 | 0.207 | 0.179 |
| VBPR | CCA | llama,false, cca=8 | 0.230 | 0.202 | 0.225 | 0.194 |
| VBPR | CCA | llama,true, cca=8 | 0.230 | 0.199 | 0.228 | 0.198 |
| VBPR | CCA | llama,true, cca=40 | 0.240 | 0.207 | 0.233 | 0.202 |
| VBPR | CCA | openai,true, cca=40 | 0.234 | 0.201 | 0.235 | 0.203 |
| VBPR | PCA | llama,false, pca=0.9 | 0.200 | 0.170 | 0.211 | 0.180 |
| AMR | Text | openai,false | 0.211 | 0.182 | 0.195 | 0.168 |
| AMR | Visual | inception3 | 0.142 | 0.122 | 0.166 | 0.143 |
| AMR | CCA | llama,true,8 | 0.223 | 0.191 | 0.229 | 0.196 |
| AMR | CCA | openai,false,40 | 0.227 | 0.195 | 0.232 | 0.199 |
| VMF | Text | openai,false | 0.129 | 0.108 | 0.130 | 0.111 |
| VMF | Visual | inception3 | 0.121 | 0.101 | 0.105 | 0.086 |
| VMF | CCA | llama,false,40 | 0.133 | 0.113 | 0.134 | 0.114 |

---

### 🧩 **n = 10**

| **Method** | **Modality** | **Params** | **NDCG (Trailer)** | **Recall (Trailer)** | **NDCG (Full Movie)** | **Recall (Full Movie)** |
|--------|----------|--------|--------|----------|--------|----------|
| VBPR | Text | llama,true | 0.399 | 0.535 | 0.416 | 0.557 |
| VBPR | Visual | inception3 | 0.433 | 0.575 | 0.413 | 0.552 |
| VBPR | CCA | llama,true,40 | 0.444 | 0.579 | 0.436 | 0.573 |
| VBPR | PCA | llama,true,0.8 | 0.428 | 0.565 | 0.413 | 0.553 |
| AMR | Text | openai,false | 0.401 | 0.531 | 0.378 | 0.506 |
| AMR | Visual | inception3 | 0.339 | 0.468 | 0.298 | 0.411 |
| AMR | CCA | openai,false,40 | 0.425 | 0.555 | 0.434 | 0.564 |
| VMF | Text | openai,false | 0.280 | 0.390 | 0.280 | 0.390 |
| VMF | Visual | inception3 | 0.281 | 0.395 | 0.266 | 0.382 |
| VMF | CCA | openai,false,40 | 0.275 | 0.385 | 0.285 | 0.391 |


## **II. Visual-RAG Results**

### 🧩 **Accuracy Metrics**

#### **Retrieval Stage**

| Name | Visual | Textual | PCA | CCA | Recal (temp) | NDCG (temp) | Recal (avg) | NDCG (avg) |
|------|--------|---------|-----|-----|-----|-----|-----|-----|
| ST Unimodal |  | * |  |  | 0.160541 | 0.159737 | 0.116791 | 0.089738 |
| OpenAI Unimodal |  | * |  |  | 0.176216 | 0.181286 | 0.116251 | 0.101413 |
| Llama3.0 Unimodal |  | * |  |  | 0.093495 | 0.096862 | 0.070260 | 0.055245 |
| Visual Unimodal | * |  |  |  | 0.036688 | 0.037534 | 0.026457 | 0.025499 |
| Visual + Textual (ST) | * | * | * |  | 0.104882 | 0.122977 | 0.092985 | 0.096264 |
| Visual + Textual (OpenAI) | * | * | * |  | 0.105041 | 0.123262 | 0.092985 | 0.096195 |
| Visual + Textual (Llama3.0) | * | * | * |  | 0.108416 | 0.139363 | 0.089009 | 0.099592 |
| Visual + Textual (ST) | * | * |  | * | 0.205637 | 0.252080 | 0.180762 | 0.174588 |
| Visual + Textual (OpenAI) | * | * |  | * | 0.087140 | 0.080328 | 0.072103 | 0.055052 |
| Visual + Textual (Llama3.0) | * | * |  | * | 0.119989 | 0.148211 | 0.099557 | 0.095724 |

#### **Recommender (Manual) Stage**

| Name | Visual | Textual | PCA | CCA | Recal (temp) | NDCG (temp) | Recal (avg) | NDCG (avg) |
|------|--------|---------|-----|-----|-----|-----|-----|-----|
| ST Unimodal |  | * |  |  | 0.084321 | 0.167488 | 0.050541 | 0.079941 |
| OpenAI Unimodal |  | * |  |  | 0.075686 | 0.166709 | 0.059188 | 0.104077 |
| Llama3.0 Unimodal |  | * |  |  | 0.039820 | 0.091084 | 0.034442 | 0.062854 |
| Visual Unimodal | * |  |  |  | 0.016110 | 0.045019 | 0.011980 | 0.030501 |
| Visual + Textual (ST) | * | * | * |  | 0.043374 | 0.124830 | 0.035712 | 0.080947 |
| Visual + Textual (OpenAI) | * | * | * |  | 0.037374 | 0.115342 | 0.036908 | 0.078362 |
| Visual + Textual (Llama3.0) | * | * | * |  | 0.037333 | 0.114743 | 0.032583 | 0.075561 |
| Visual + Textual (ST) | * | * |  | * | 0.107084 | 0.268102 | 0.095418 | 0.174363 |
| Visual + Textual (OpenAI) | * | * |  | * | 0.028301 | 0.078006 | 0.025079 | 0.059463 |
| Visual + Textual (Llama3.0) | * | * |  | * | 0.047614 | 0.145944 | 0.041977 | 0.092620 |

#### **Recommender (LLM) Stage**

| Name | Visual | Textual | PCA | CCA | Recal (temp) | NDCG (temp) | Recal (avg) | NDCG (avg) |
|------|--------|---------|-----|-----|-----|-----|-----|-----|
| ST Unimodal |  | * |  |  | 0.077300 | 0.169313 | 0.051284 | 0.086847 |
| OpenAI Unimodal |  | * |  |  | 0.089671 | 0.179811 | 0.057714 | 0.101169 |
| Llama3.0 Unimodal |  | * |  |  | 0.033492 | 0.083699 | 0.032395 | 0.066655 |
| Visual Unimodal | * |  |  |  | 0.017340 | 0.046774 | 0.008470 | 0.027337 |
| Visual + Textual (ST) | * | * | * |  | 0.035130 | 0.120745 | 0.030523 | 0.066660 |
| Visual + Textual (OpenAI) | * | * | * |  | 0.036987 | 0.101464 | 0.034161 | 0.072735 |
| Visual + Textual (Llama3.0) | * | * | * |  | 0.038048 | 0.118645 | 0.028749 | 0.078329 |
| Visual + Textual (ST) | * | * |  | * | 0.097654 | 0.246567 | 0.089355 | 0.159758 |
| Visual + Textual (OpenAI) | * | * |  | * | 0.040250 | 0.084531 | 0.029190 | 0.063593 |
| Visual + Textual (Llama3.0) | * | * |  | * | 0.048338 | 0.141869 | 0.034597 | 0.079539 |
