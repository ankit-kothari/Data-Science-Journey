# Data Science Journey

This repo is roughly chronological. It's how I actually learned — one question leading to the next, each project existing because the previous one left something unanswered. The through-line, if there is one: I kept ending up back at the same problem — *how do you figure out what a user wants and surface the right thing?* — just with better tools each time.

[LinkedIn](https://www.linkedin.com/in/ankit-kothari-510a9623) · [Substack](https://kots256.substack.com) · ankit256@gmail.com

```
 ┌───────────────────────────────────────────────────────────────────────────────┐
 │                                                                               │
 │   gradient       "is this       bag-of-words     what do CNNs    collaborative│
 │   descent,       result even     → embeddings     actually see?   filtering on│
 │   tensor shapes  real?"          → attention                      20M ratings │
 │       │               │              │                │               │       │
 │       ▼               ▼              ▼                ▼               ▼       │
 │   FOUNDATIONS → EXPERIMENTATION → NLP → VISION → RECOMMENDATIONS             │
 │                                                       │                       │
 │                                          two-tower ◂──┘                       │
 │                                          bandits                              │
 │                                          visual search                        │
 │                                          LLM agents                           │
 │                                               │                               │
 │                                               ▼                               │
 │                                     TRANSFORMERS & LLMs                       │
 │                                     encoder/decoder from scratch              │
 │                                     DDP → FSDP distributed training           │
 │                                     distillation, serving                     │
 │                                                                               │
 └───────────────────────────────────────────────────────────────────────────────┘
```

---

## It started with gradient descent

I wrote [gradient descent from scratch](Machine-Learning/GD%20and%20SGD%20from%20scratch) — GD and SGD on linear and logistic regression, no sklearn — because I wanted to see the loss surface move. That sounds simple, but watching how learning rate and batch size change the optimization trajectory built an intuition I still rely on when a training run goes sideways.

Around the same time I made a [tensor shape reference](Deep-Learning/Concepts-Deep%20Learning/Shapes%20in%20Deep%20Learning.ipynb) for myself: what are the shapes at every layer for ANN, RNN, LSTM, CNN, BiLSTM, pooling? I got tired of debugging shape mismatches by guessing.

Then a dataset didn't fit in memory, and I had to learn [process pools, threading, and downcasting](Big%20Data/optimizing-large-datasets). That was the first time it clicked that engineering and data science aren't separate things.

---

## Then I realized: a model is only as good as the experiment behind it

I built an [A/B testing framework](Probablity-and-Statistics/AB-Testing) — power analysis, sample sizing, hypothesis testing — and applied it to a [real mobile game retention study](Probablity-and-Statistics/AB-Testing/ab-testing-cookie-cat-dataset.ipynb) (Cookie Cats, day-1 vs day-7 retention). Side work on [Bayesian stats](Probablity-and-Statistics/bayesian_statistics.ipynb) and [posterior updating](Probablity-and-Statistics/updating_posterior_distribution_demo.ipynb) filled in the probabilistic thinking.

[Credit risk modeling](Machine-Learning/Credit-Risk-Analysis-master) (Logistic Regression vs XGBoost vs ANN on LendingClub data) taught me more about class imbalance than about model selection — and that a more complex model doesn't always buy you anything.

Before all of this, there were the [unglamorous projects](Data_Cleaning_Analysis_and_Visualization): H-1B visa trends, Google Play Store apps, traffic regression, currency analysis. They taught me to find signal before reaching for a model.

---

## NLP: I didn't jump to transformers — I built up to them

The project that made embeddings click was building [the same chatbot four different ways](Natural-Language-Processing/chatbots): TF-IDF, word embeddings, sentence embeddings, and TF-Hub encoders — same FAQ dataset, same queries. Watching a query go from "no match" to "perfect match" as the representation got richer was more convincing than any paper.

Then a [text classification benchmark](Deep-Learning/Text%20Classification) — parallel ConvNet vs stacked BiLSTM vs baseline on Amazon reviews. Same data, three architectures. The takeaway wasn't "BiLSTM wins" — it was that architecture choice depends on the structure of the signal, not the trend cycle.

[NER fine-tuning](Natural-Language-Processing/Named%20Entity%20Recognition), [topic modeling](Natural-Language-Processing/Topic-Modeling) (LDA and NMF), and a [text-to-image project with CGAN + BERT embeddings](Natural-Language-Processing/T2I-with-quantitative-embeddings) (trained on NVIDIA HPC) filled in the rest. The CGAN project was the first time I bridged language and vision representations — it didn't work great, but it showed me where the two modalities connect.

---

## A detour through vision that turned out not to be a detour

[Style transfer](Machine-Vison/Style%20Transfer) showed me what CNN feature maps encode at different depths — early layers capture edges, deeper layers capture texture and style. [Digit recognition with OpenCV](Machine-Vison/Handwrittent%20Digit%20Recognition%20using%20OpenCV%20and%20Keras) was the full pipeline from raw pixels to prediction. [Seam carving](Machine-Vison/seam_carving) (content-aware image resizing) was just a beautiful algorithm.

The reason this wasn't a detour: understanding what a network learns *layer by layer* is the same intuition you need to understand what attention heads do in a transformer. I didn't know that at the time.

---

## Recommendations: where everything converged

This is the part of the repo that has the most depth, because it's the problem I kept circling back to.

It started with [matrix factorization from scratch](Recommendations/recommendation_matrix_factorization) — collaborative filtering on 20M MovieLens ratings. SGD with L2 regularization, explicit user/item biases, sparse matrices. 3.4B parameters compressed to 2.4M via latent factor decomposition. No GPU — the memory constraints forced me to think about sparsity from the start.

That naturally led to the question: *how do production systems actually do this?* So I built a [full two-tower retrieval and ranking pipeline](Recommendations/Two_Tower_Model) on H&M fashion data. Four stages: two-tower retrieval with in-batch negatives, single-task ranker, multi-task ranker (engagement + conversion), and FAISS ANN serving with latency benchmarks. This is the architecture pattern that runs at Netflix, YouTube, Airbnb — I wanted to build every piece of it myself.

Then [contextual bandits](Recommendations/Multi-Bandit-Arm) — Epsilon-Greedy vs UCB vs Thompson Sampling on Yahoo! Front Page click logs. Offline evaluation via policy replay. Thompson Sampling hit 4.11% CTR vs 2.9% uniform baseline. The explore/exploit tradeoff is the part of personalization that doesn't go away no matter how good your model gets.

[Image-to-product recommendations](Recommendations/image2product_recs) bridged vision and recsys — visual similarity search using CNN embeddings over 15K fashion images. Users don't always search with words.

Most recently, [upsell recommendations with LLM agents](Recommendations/upsell_recommendation) — a LangGraph pipeline: attribute extraction → alternative identification → comparison → recommendation synthesis. Classic recsys meets LLM-era tooling.

---

## Transformers and LLMs: I wanted to understand every matrix multiply

I built both a [transformer encoder](Natural-Language-Processing/Transformers/transformers_from_scratch/encoder_only_from_scratch.ipynb) and [decoder](Natural-Language-Processing/Transformers/transformers_from_scratch/decoder_only_from_scratch.ipynb) from scratch — no `nn.Transformer`, every component (multi-head attention, positional encoding, feed-forward blocks, layer norm) implemented and shape-traced by hand. The encoder made bidirectional context concrete; the decoder made causal masking and autoregression concrete.

Then the question became: *what happens when the model doesn't fit on one GPU?* I wrote [5 progressive distributed training scripts](llm_projects/scratch2scale) that show exactly what breaks at each scale and how the next strategy fixes it: DDP baseline (OOM) → ZeRO-2 (4x memory savings) → ZeRO-3 (full sharding) → mixed precision + activation checkpointing.

[Distillation](Natural-Language-Processing/random_experiments/distillation_large_language_model.ipynb) came from a practical question: your best model is too expensive to serve — can you teach a smaller one to approximate it?

[Research paper to podcast](llm_projects/research_paper2podcast) was a personal project — PDF parsing → section summarization → conversational script → audio. I wanted to absorb papers during my commute.

---

## Data engineering ran through everything

These weren't a separate stage — they came up whenever something didn't fit on one machine.

[PySpark and MapReduce](Big%20Data/Pyspark-and-MapReduce) for the Spark mental model (lazy evaluation, shuffle costs). [Spotify 1M songs with Dask](Big%20Data/Spotify%201M%20Songs%20Analysis%20with%20Dask) for out-of-core analysis when pandas breaks. [Optimization problems](optimization_projects) (job scheduling, route optimization, power generation) for the mathematical thinking — constraints, objectives, feasible regions — that shows up everywhere.

---

## What connects all of this

If I look at this repo as a whole, the question that kept pulling me forward was: **how do you represent users, items, and context as vectors, and turn those representations into good decisions at scale?**

That question looks different at each stage — it's a loss surface in the foundations stage, an embedding space in the NLP stage, a retrieval-ranking pipeline in the recommendations stage, an attention mechanism in the transformer stage. But it's the same question.

The part I'm most interested in now is where LLMs and recommendation systems meet: using language understanding to build richer representations of what users want, and using recommendation principles (retrieval, ranking, explore/exploit) to make LLM-powered products that actually surface the right thing.

---

**Built with:** Python · PyTorch · TensorFlow · NumPy · pandas · scikit-learn · FAISS · PySpark · Dask · LangGraph · LangChain · Hugging Face Transformers · spaCy · OpenCV · Docker · FSDP
