# Image-to-Product Recommendations

Visual similarity search: given a product image, retrieve the most visually similar items from a fashion catalog using CNN embeddings.

## Objective

Build a content-based recommendation system that uses visual features (not just metadata) to find similar products — bridging the gap between what a user *sees* and what the catalog *contains*.

## Approach

1. **Feature extraction** — Pre-trained CNN (transfer learning) extracts embedding vectors from product images
2. **Similarity search** — Cosine similarity over embedding space retrieves nearest neighbors
3. **Evaluation** — Visual inspection of query → retrieved items across product categories

## Dataset

~15,000 fashion product images with metadata (article type, color, season) from the Kaggle Fashion Product Images dataset.

## Key Techniques

- Transfer learning for visual embeddings
- Cosine similarity retrieval
- Multimodal signals: image features + product metadata
