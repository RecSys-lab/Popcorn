# 🔎 Experiments (Full Version)

## I. Selected Best-Performing Results

- **Settings:** `n_epochs=10`, `test_ratio=0.2`, `k_core=10`, `top_n ∈ {2,10}`  
- **Visual**: Popcorn-Visual (Inception-v3 + max pooling)  
- **Text**: Text-RAG-Plus (`text_aug ∈ {true,false}`)
- **MovieLens**: `1M` version

---

## 🧩 **n = 2**

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

## 🧩 **n = 10**

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

