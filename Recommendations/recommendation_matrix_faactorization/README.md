# Movie Recommendation

Ankit Kothari 

## `Introduction`

The dataset contains `138493` users and `26744 movies`. Each user has rated at least 20 movies. This dataset has `~20 Million user-rating pairs`. **The problem that makes it challenging is that we are trying to train the model to be able to estimate the ratings for roughly 3703856792 user-rating combinations without the use of cloud computing or GPUs.**

- **Motivation**
    - The primary motivation behind solving this problem was to use efficient data compression techniques like `sparse matrix methods` and `efficient data structures` like hash maps to store and iterate over the data. These techniques are handy with memory constraints given **a dataset as massive as this one with 20 Million ratings**.
    - Use of `Matrix - Factorization` to represent the most critical latent features in a compressed format to reduce the computation.
    - One of the goals is to **understand user bias and movie_bias** and its impact on the movie recommendation system.
    - Understand the function of gradient descent algorithm along with regularization terms.

### **Challanges and Methods used to Address them**

### Datatype Downcasting

The dataset was gigantic to fit in the computer's memory (no GPUs were used). The initial memory usage was 1.9GB. Downcasting of float and integer values reduced the memory usage **to 228 MB**

- Int64 was converted to int32
- float64 was converted to float32
- removed the timestamp column, which was not used.
- The following is the final list of conversions from original to new data types.
    
    
    |  | Before | After |
    | --- | --- | --- |
    | int32 | 0 | 2 |
    | int64 | 2 | 0 |
    | float 32 | 0 | 1 |
    | float 64 | 1 | 0 |

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/mu.png" width="80%">

Comparison of Memory Usage before and after Datatype Downcasting

### Matrix Factorization

**The shrink in parameters is from 3418838198 to 2447685 due to MF, which  is 0.07% of the original size (using K=15)**

`After Matrix Factorization`: N X M = **(NXK) * (K*M)**

- Here K is the latent features; the `NXK user matrix` tries to capture the **importance of user features for each N user** by iterating through the data and how the user has rated movies. The underlying assumption is that the model learns, for example, how much the user like different aspects of the film like genres : [action, comedy, suspense, dark, anime]. A vector-like this might be represented by [1,0.1,-1,1.9,4]. Here the user likes Animated movies a lot and followed by dark and action, but it seems like the user does not like suspense and comedy. While this embedding is not directly apparent, the primary goal of matrix-factorization is to create a dense representation of the most useful features.
- Like above, here, `MXK represents the movie matrix`. The goal is to create an embedding for each movie depicting its distinct features.

### Sparse Matrix

- Since there will be a lot of cells with zero, a user will not rate all the ~28K movies. This type of matrix where there are a lot of 0's is called a dense matrix, and it is computationally expensive; on a regular Mac, this computation runs out of memory. Sparse-matrix was used to avoid running into this issue, which stores the (user_id,moveie_id) rating like a hash-map. It makes the computation go a lot faster.
- Element Wise Multiplication
- to vectorize the computation and instead of looping through the dataset using,
- Embedding Matrix was created using the Element-wise between user and movie embeddings to avoid creating a dense matrix (NXM), which was one of the project's goals.

### Cold Start Problem

- One of the drawbacks of this method of recommendation system is it creates recommendations and embeddings only for the user, movie it has seen in the training in the dataset. It cannot generate an embedding for a user it has not seen in the dataset. Similarly, it cannot create an embedding for a movie it has not seen in the dataset. For this reason, all the users and movies which were not part of the training data had to be removed from the testing dataset.

## `FORMULATION`

### Mathematical Model

- **Optimization variables**
    
    
    | Optimization Variables | Variable Name | Shape of the Matrix | Experimntal Initializaation  |
    | --- | --- | --- | --- |
    | User matrix | W | [NXK] | 12*np.random.random((N, K))/K |
    | User bias |  b | [N,1] | 10*np.ones(N) |
    | Movie matrix | U | [M X K] | 12*np.random.random((M, K))/K |
    | Movie bias | c | [M,1] | 10*np.ones(M) |
    | Global mean | mu | mu (Scaler) |  |
    
    <aside>
    💡 `N` is number of unique users and 
    `M` is the number of unique movies and 
    `K` is the latent dimension after Matrix Factorization
    
    </aside>
    
- **Cost Function**
    
    <img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/cost.png" width="80%">
    

**The descriptive version of the cost function.**

`**J=cost**` = **$(1/total\_ratings)$ *$sum( [(actual\_rating – predicted\_rating)^2] )$+ $`lambda* (regularization terms)`$**

`**Regularization terms**` = $norm (user\_matrix)^2 + norm(movie\_matrix)^2 + norm (user\_bias)^2 + norm (movie\_bias)^2$

<aside>
💡 In the code I am printing the non-regularized loss while ***using the regularization for gradient which is needed for traning.***

</aside>

**Prediction Function**

`**Predicted Rating**` = $user\_matrix.(movie\_matrix) + user\_bias + movie\_bias + global\_mean$

### Optimization Model (Gradient Descent with Ridge Regularization)

Alternating Least Squares method is used, First we update the W and b matrix and then update the U and c matrix. This method is known to converge and get close to local mimimum faster. Gradient Descent algorithm is used to update the weights at each iteration. 

- **Gradient of the User Matrix**
    
    <img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/w.png" width="80%">
    
    - updated `**user_matrix**` = $user\_matrix – eta* user\_matrix\_gradient$ **where,**
        
        **`user_matrix_gradient`** = $**(-2/total\_rows) *(actual\_rating - predicted\_rating – user\_bias-movie\_bias– global\_mean)* movie\_matrix + 2*reg*user\_matrix**$
        
- **Gradient of the User Bias**
    
    <img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/b.png" width="80%">
    
    - updated : **`user_bias` = $user\_bias – eta * user\_bias\_gradient$  where,**
        - **`user_bias_gradient` =$(-2/user\_mean\_vector) * (actual – predicted – movie\_bias – global\_mean)+ 2* lambda*user\_bias$**
    
    where `**user_mean_vector**` is the number of movies each user has watched, so we divide the sum of total ratings from the movies by a particular user divided by the total movies the user has rated, to give an individual user bias. 
    
- **Gradient of the Movie Matrix**
    
    <img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/U.png" width="80%">
    
    - updated `**movie_matrix**` = $movie\_matrix - eta*movie\_matrix\_gradient$ **where,**
        - **movie_matrix_gradient = $(-2/total\_rows) *(actual\_rating - predicted\_rating – user\_bias-movie\_bias– global\_mean).T * user\_matrix + 2* lambda* movie\_matrix$**
- **Gradient of the Movie Bias**
    
    <img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/c.png" width="80%">
    
    - updated **`movie_bias` = movie_bias – eta * movie_bias_gradient, where**
        - `**movie_bias_gradient` =$(-2/ movie\_mean\_vector ) * (actual – predicted – user\_bias – global\_mean) + 2*lambda* movie\_bias$**
    
    where  `**movie_mean_vector**` is the number of users  watched each of the movies, so we divide the sum of total ratings from the all the ratings given by users to  a particular movie divided by the total number of users that has rated the movie, to give an individual movie bias. 
    

## `Numerical Studies`

**Data Set** : Movie Lens Data set with 20 M ratings ; Please find the link to get the `ratings.csv` from Kaggles Movie Lens Data Set

[MovieLens 20M Dataset](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset?select=rating.csv)

**Tools used**: Matrix Factorization, Sparse Matrix, Vectorization

### Training Parameters

- Epochs = 110
- K (Latent Dimension) = 15
- Learning Rate = 0.01
- Regularization Parameter = 0.01
- Train Data Size = `12000157` ratings
- Test Data Size = `7997412` ratings

### Numerical results

- Initial Train Cost =502.91, Initial Test Cost = 502.91
- Final Train Cost = 0.960, Final Test Cost=.998

### Training Loss

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/training_loss_plot.png" width="80%">

### Testing Loss
<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/test_loss_plot.png" width="80%">

### Results and Discussion

- **The data was split into Train, and Test data, with 60% used on training and the remaining 40% on test data. The gradient descent optimization algorithm trained the weights for the User Matrix, Movie Matrix, User Bias Matrix, and Movie Bias Matrix. It took about 110 epochs at a learning rate of 0.01 and a regularization parameter of 0.01. After all the vectorization and data compression techniques, it took each epoch about 3 minutes to iterate over the entire training dataset and compute Train and test loss, which is excellent given no cloud computing was used.**
- **The results on the prediction set were very encouraging. The below graph shows that if a user rated a particular movie 0.5, it got the least predicted rating on an average. In contrast, if the user rated a movie 5, it got the highest predicted rating on an average. So in this scenario, it will be able to rank the movies in order of a user preference which is essential. This will work great for a recommendation use case since the relative ranking is more important than the actual ranking itself, and this model can do it.**

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/actual_rating_vs_avg_predicted_rating_plot.png" width="80%">


### Future Directions

- In the future I would like to try different optimization algorithm for faster convergence.
- Leverage parallel computing to get performace gains.
- Explore other data structures and compression techniques
- Use of SVD and SVD++ instead of Matrix Factorization.
- Explore other  research papers in the area of recommendation.

## `References`

- [https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset?select=rating.csv](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset?select=rating.csv)
- [https://mc-stan.org/docs/2_19/functions-reference/CSR.html](https://mc-stan.org/docs/2_19/functions-reference/CSR.html)
- [https://hippocampus-garden.com/pandas_sparse/#converting-to-csr-matrix](https://hippocampus-garden.com/pandas_sparse/#converting-to-csr-matrix)
- [https://towardsdatascience.com/recommender-systems-matrix-factorization-using-pytorch-bd52f46aa199](https://towardsdatascience.com/recommender-systems-matrix-factorization-using-pytorch-bd52f46aa199)
- [https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778](https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778)
- [https://www.dataquest.io/blog/pandas-big-data/](https://www.dataquest.io/blog/pandas-big-data/)
