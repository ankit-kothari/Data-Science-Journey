# Two-Tower Retrieval & Ranking Pipeline

End-to-end recommendation system on H&M fashion data: candidate generation via two-tower retrieval, single-task and multi-task ranking, and approximate nearest neighbor serving with FAISS.

## Objective

Build a production-style recommendation pipeline from scratch using TensorFlow Recommenders on real e-commerce data (H&M Personalized Fashion Recommendations dataset).

## Pipeline

```
User/Item Features → Two-Tower Retrieval → Candidate Set → Ranking Model → Final Recommendations
                                                              ↓
                                                     Multi-Task Learning
                                                   (engagement + conversion)
```

## Project Structure

| Notebook | Stage | What it does |
|---|---|---|
| `candidate_generator_submission.ipynb` | **Retrieval** | Two-tower model (user tower + item tower) trained with in-batch negatives. Outputs user and item embeddings for ANN lookup. Uses Dask for large-scale data processing. |
| `rankers_hmdata.ipynb` | **Ranking** | Single-objective ranking model using feature crosses and deep layers. Scores candidate items for a given user. |
| `rankers_hmdata_multi_task_learning.ipynb` | **Ranking (multi-task)** | Joint optimization of click prediction and purchase prediction. Balances engagement vs conversion with task-specific heads sharing a common backbone. |
| `user_representation.ipynb` | **Serving** | FAISS approximate nearest neighbor index for real-time retrieval. Benchmarks FAISS vs brute-force sklearn on embedding lookup latency and recall. |

## Key Techniques

- **Two-tower architecture** with separate user and item embedding networks
- **In-batch negative sampling** for efficient contrastive learning
- **Multi-task learning** with shared-bottom architecture and task-specific heads
- **FAISS ANN indexing** for sub-millisecond retrieval at scale
- **Dask** for out-of-core data processing on large transaction logs

## Dataset

H&M Personalized Fashion Recommendations (Kaggle) — ~31M transactions, 1.3M customers, 105K articles.
