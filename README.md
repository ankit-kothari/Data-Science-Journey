# Ankit Kothari — Personalization, Recommendations & Applied AI

Hands-on projects in recommendation systems, LLM engineering, and applied machine learning — from matrix factorization on 20M ratings to distributed transformer training with FSDP.

[LinkedIn](https://www.linkedin.com/in/ankit-kothari-510a9623) | [Substack](https://kots256.substack.com) | ankit256@gmail.com

---

## Recommendation & Personalization Systems

| Project | What it does | Key detail |
|---|---|---|
| [**Two-Tower Retrieval & Ranking Pipeline**](Recommendations/Two_Tower_Model) | End-to-end candidate generation → ranking → multi-task learning → FAISS serving on H&M fashion data | 4-stage pipeline: two-tower retrieval with in-batch negatives, single-task ranker, multi-task ranker (engagement + conversion), and ANN serving benchmarks |
| [**Matrix Factorization from Scratch**](Recommendations/recommendation_matrix_factorization) | Collaborative filtering on 20M MovieLens ratings using SGD with L2 regularization, sparse matrices, no GPU | Reduced 3.4B parameters to 2.4M (0.07%) via latent factor decomposition; explicit user/item bias modeling; memory-optimized with downcasting and sparse storage |
| [**Contextual Bandits for Explore/Exploit**](Recommendations/Multi-Bandit-Arm) | Epsilon-Greedy vs UCB vs Thompson Sampling on Yahoo! Front Page click logs | Rigorous offline evaluation via policy replay; Thompson Sampling achieves 4.11% CTR vs 2.9% uniform baseline |
| [**Image-to-Product Recommendations**](Recommendations/image2product_recs) | Visual similarity search using CNN embeddings over 15K fashion product images | Transfer learning for visual embeddings + cosine similarity retrieval bridging image features and product metadata |
| [**Upsell Recommendations with AI Agents**](Recommendations/upsell_recommendation) | LangGraph agentic pipeline: attribute extraction → alternative identification → comparison → recommendation synthesis | Multi-node state machine with Pydantic-validated structured outputs at each step |

## LLM & Transformer Engineering

| Project | What it does | Key detail |
|---|---|---|
| [**Distributed Training: DDP → FSDP ZeRO-2/3**](llm_projects/scratch2scale) | 5 progressive scripts showing exactly what breaks at each scale and how the next strategy fixes it | DDP baseline (OOM) → ZeRO-2 (4x savings) → ZeRO-3 (full sharding) → mixed precision + activation checkpointing |
| [**Transformer Decoder from Scratch**](Natural-Language-Processing/Transformers/transformers_from_scratch/decoder_only_from_scratch.ipynb) | Full decoder-only transformer: masked multi-head attention, feed-forward blocks, positional encoding | No `nn.Transformer` — every component built and shape-traced by hand |
| [**Transformer Encoder from Scratch**](Natural-Language-Processing/Transformers/transformers_from_scratch/encoder_only_from_scratch.ipynb) | Encoder-only transformer with self-attention and layer normalization | Complements the decoder — both architectures implemented from first principles |
| [**Text-to-Image with CGAN + BERT Embeddings**](Natural-Language-Processing/T2I-with-quantitative-embeddings) | Conditional GAN conditioned on BERT text embeddings for text-to-image generation | Custom CGAN architecture trained on NVIDIA HPC cluster |
| [**LLM Distillation**](Natural-Language-Processing/random_experiments/distillation_large_language_model.ipynb) | Knowledge distillation from large language models to smaller student models | Practical model compression for production deployment |
| [**Research Paper to Podcast**](llm_projects/research_paper2podcast) | End-to-end LLM pipeline: PDF parsing → section summarization → conversational script → audio | Full document-to-audio transformation |

## Experimentation & Causal Inference

| Project | What it does | Key detail |
|---|---|---|
| [**A/B Testing Framework**](Probablity-and-Statistics/AB-Testing) | End-to-end experiment design: power analysis, sample sizing, hypothesis testing, significance interpretation | [Cookie Cats case study](Probablity-and-Statistics/AB-Testing/ab-testing-cookie-cat-dataset.ipynb) — real mobile game retention A/B test with day-1 and day-7 retention analysis |
| [**Credit Risk Modeling**](Machine-Learning/Credit-Risk-Analysis-master) | Predict loan default comparing Logistic Regression, XGBoost, and ANN on LendingClub data | Fintech domain — hyperparameter tuning, class imbalance handling, feature engineering at scale |

## NLP & Text Understanding

| Project | What it does | Key detail |
|---|---|---|
| [**Text Classification Benchmark**](Deep-Learning/Text%20Classification) | Multi-label classification on Amazon reviews: parallel ConvNet vs stacked BiLSTM vs baseline | Same dataset, three architectures — understanding which representation wins and why |
| [**Chatbots: 4 Retrieval Architectures**](Natural-Language-Processing/chatbots) | TF-IDF vs Word Embeddings vs Sentence Embeddings vs TF-Hub Encoders for FAQ retrieval | Progressive complexity: sparse → dense → contextual representations |
| [**Named Entity Recognition**](Natural-Language-Processing/Named%20Entity%20Recognition) | NER fine-tuning + POS tagging pipelines | Entity extraction for structured information retrieval |
| [**Topic Modeling**](Natural-Language-Processing/Topic-Modeling) | LDA and NMF for unsupervised topic extraction | Document clustering and theme discovery |

## Foundations

| Project | What it does |
|---|---|
| [**Gradient Descent from Scratch**](Machine-Learning/GD%20and%20SGD%20from%20scratch) | GD and SGD on linear and logistic regression — no sklearn, understanding optimization at the gradient level |
| [**Shapes in Deep Learning**](Deep-Learning/Concepts-Deep%20Learning/Shapes%20in%20Deep%20Learning.ipynb) | Tensor shape reference for ANN, RNN, LSTM, CNN, BiLSTM, MaxPooling — the mental model that prevents shape bugs |
| [**Large Dataset Optimization**](Big%20Data/optimizing-large-datasets) | Process pools, threading, downcasting — practical techniques for data that doesn't fit in memory |
| [**Spotify 1M Songs with Dask**](Big%20Data/Spotify%201M%20Songs%20Analysis%20with%20Dask) | Out-of-core distributed analysis on 1M tracks using Dask |
| [**PySpark & MapReduce**](Big%20Data/Pyspark-and-MapReduce) | Spark fundamentals: transformations, actions, distributed data manipulation |
| [**Optimization Problems**](optimization_projects) | Job scheduling, route optimization, power generation — linear programming and constraint satisfaction |

---

**Tools:** Python, PyTorch, TensorFlow, NumPy, pandas, scikit-learn, FAISS, PySpark, Dask, LangGraph, LangChain, Hugging Face Transformers, spaCy, OpenCV, Plotly, Docker, FSDP
