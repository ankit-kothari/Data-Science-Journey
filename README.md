# Ankit Kothari — Personalization, Recommendations & Applied AI

Hands-on projects in recommendation systems, LLM engineering, and applied machine learning — from matrix factorization on 20M ratings to distributed transformer training with FSDP.

[LinkedIn](https://www.linkedin.com/in/ankit-kothari-510a9623) | [Substack](https://kots256.substack.com) | ankit256@gmail.com

---

## Recommendation & Personalization Systems

| Project | What it does | Key detail |
|---|---|---|
| [**Two-Tower Retrieval Model**](Recommendations/Two_Tower_Model) | Full candidate generation + ranking pipeline with multi-task learning | Separate candidate gen and ranker notebooks; multi-task objective balances engagement and conversion |
| [**Matrix Factorization from Scratch**](Recommendations/recommendation_matrix_factorization) | Collaborative filtering on 20M MovieLens ratings using SGD, sparse matrices, no GPU | Reduced 3.4B parameters to 2.4M (0.07%) via latent factor decomposition; models user and item bias explicitly |
| [**Contextual Bandits for Explore/Exploit**](Recommendations/Multi-Bandit-Arm) | Online learning on Yahoo! Front Page click logs — balancing exploration vs exploitation in real time | Real-world logged bandit data with unbiased offline evaluation |
| [**Image-to-Product Recommendations**](Recommendations/image2product_recs) | Visual similarity search: given a product image, retrieve visually similar items from catalog | Multimodal embeddings bridging vision and product metadata |
| [**Upsell Recommendations with AI Agents**](Recommendations/upsell_recommendation) | Agentic recommendation system using LangGraph — autonomous multi-step reasoning for cross-sell/upsell | LLM agent with tool use for personalized product recommendations |

## LLM & Transformer Engineering

| Project | What it does | Key detail |
|---|---|---|
| [**Transformer Decoder from Scratch**](Natural-Language-Processing/Transformers/transformers_from_scratch/decoder_only_from_scratch.ipynb) | Full decoder-only transformer: masked multi-head attention, feed-forward blocks, positional encoding | Built block by block — no `nn.Transformer`, every shape traced |
| [**Transformer Encoder from Scratch**](Natural-Language-Processing/Transformers/transformers_from_scratch/encoder_only_from_scratch.ipynb) | Encoder-only transformer with self-attention and layer normalization | Complements the decoder — full understanding of both architectures |
| [**Distributed Training: DDP to FSDP ZeRO-2/3**](llm_projects/scratch2scale) | Progressive scaling: single GPU baseline (OOM) → DDP → FSDP ZeRO-2 → ZeRO-3 with advanced sharding | 5 scripts showing exactly what breaks at each scale and how to fix it |
| [**Text-to-Image with CGAN + BERT Embeddings**](Natural-Language-Processing/T2I-with-quantitative-embeddings) | Conditional GAN conditioned on BERT text embeddings for text-to-image generation | Custom CGAN architecture with quantitative embedding conditioning |
| [**Research Paper to Podcast**](llm_projects/research_paper2podcast) | LLM pipeline that converts academic papers into podcast-style audio | End-to-end: PDF parsing → summarization → conversational script → audio |
| [**LLM Distillation**](Natural-Language-Processing/random_experiments/distillation_large_language_model.ipynb) | Knowledge distillation from large language models to smaller student models | Practical compression techniques for production deployment |

## Experimentation & Measurement

| Project | What it does | Key detail |
|---|---|---|
| [**A/B Testing Framework**](Probablity-and-Statistics/AB-Testing) | End-to-end experiment design: power analysis, sample size, hypothesis testing, p-value interpretation | Includes [Cookie Cats case study](Probablity-and-Statistics/AB-Testing/ab-testing-cookie-cat-dataset.ipynb) — real mobile game retention experiment |
| [**Evaluation Metrics from Scratch**](Natural-Language-Processing/helper_functions/metric_evaluation.ipynb) | Classification and ranking metrics implemented from numpy | Precision, recall, F1, AUC — understanding the math, not just the API |
| [**Credit Risk Modeling**](Machine-Learning/Credit-Risk-Analysis-master) | Predict loan default: Logistic Regression vs XGBoost vs ANN on LendingClub data | Fintech domain — hyperparameter tuning, class imbalance, feature engineering at scale |

## Deep Learning & NLP

| Project | What it does | Key detail |
|---|---|---|
| [**Shapes in Deep Learning**](Deep-Learning/Concepts-Deep%20Learning/Shapes%20in%20Deep%20Learning.ipynb) | Tensor shape reference for ANN, RNN, LSTM, CNN, BiLSTM, MaxPooling layers | The mental model that prevents shape bugs |
| [**Text Classification Benchmark**](Natural-Language-Processing/Transformers/classification_tasks) | Multi-class and multi-label classification across architectures | [distilBERT](Natural-Language-Processing/Transformers/classification_tasks), [Pytorch](Deep-Learning/Text%20Classification), [Sentiment](Natural-Language-Processing/Transformers/classification_tasks) |
| [**Chatbots: 4 Architectures Compared**](Natural-Language-Processing/chatbots) | TF-IDF vs Word Embeddings vs Sentence Embeddings vs TF-Hub Encoders | Same task, four approaches — understanding which representation wins and why |
| [**Topic Modeling**](Natural-Language-Processing/Topic-Modeling) | LDA and NMF for unsupervised topic extraction | Document clustering and theme discovery |
| [**Named Entity Recognition**](Natural-Language-Processing/Named%20Entity%20Recognition) | NER pipeline with spaCy | Entity extraction for structured information retrieval |
| [**Style Transfer**](Machine-Vison/Style%20Transfer) | Neural style transfer using VGG19 in PyTorch | Content + style loss optimization |
| [**Gradient Descent from Scratch**](Machine-Learning/GD%20and%20SGD%20from%20scratch) | GD and SGD on linear and logistic regression — no sklearn | Understanding optimization at the gradient level |

## Data Engineering & Scale

| Project | What it does | Key detail |
|---|---|---|
| [**Large Dataset Optimization**](Big%20Data/optimizing-large-datasets) | Process pools, threading, downcasting, memory optimization | Practical techniques for working with data that doesn't fit in memory |
| [**Spotify 1M Songs with Dask**](Big%20Data/Spotify%201M%20Songs%20Analysis%20with%20Dask) | Distributed data analysis on 1M tracks using Dask | Out-of-core computation on a single machine |
| [**PySpark & MapReduce**](Big%20Data/Pyspark-and-MapReduce) | Spark fundamentals: transformations, actions, distributed data manipulation | MapReduce mental model for distributed processing |
| [**Optimization Problems**](optimization_projects) | Job scheduling, construction planning, power generation, route optimization | Linear programming and constraint satisfaction |

## Data Analysis & Visualization

| Project | What it does |
|---|---|
| [**H1B Visa Trend Analysis**](Data_Cleaning_Analysis_and_Visualization/H1B_Visa_Analysis) | Employer and state trends from USCIS data — [interactive Plotly visualizations](https://colab.research.google.com/drive/1BREsuISGVMJiQrdBH03KlO3OpMyzqqbN?usp=sharing) |
| [**Customer Segmentation for Ad Targeting**](Data_Cleaning_Analysis_and_Visualization/Identify%20Customers%20for%20Promotion) | Identifying high-value customer segments for social media ad campaigns |
| [**App Store Revenue Optimization**](Data_Cleaning_Analysis_and_Visualization/Identify%20Undervalued%20Apps%20on%20Google%20Store) | Finding undervalued apps to improve revenue |
| [**Bike Rental Prediction**](Machine-Learning/predicting_bike_rentals) | Decision trees vs Random Forest for hourly demand prediction |

---

**Tools:** Python, PyTorch, TensorFlow, NumPy, pandas, scikit-learn, PySpark, Dask, LangGraph, LangChain, Hugging Face Transformers, spaCy, OpenCV, Plotly, Docker, FSDP
