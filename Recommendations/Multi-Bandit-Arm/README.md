

## Multi-Arm Bandit

### Dataset: R6A - Yahoo! Front Page Today Module User Click Log Dataset, version 1.0 (1.1 GB)

- Selected dataset contains a fraction of user click log for news articles displayed in the Featured Tab of the Today   Module on Yahoo! Front Page (http://www.yahoo.com) during the first ten days in May 2009. The articles were chosen uniformly at random from a hand-picked pool of high-quality articles, which allows one to use a recently developed method to obtain an unbiased evaluation of an arbitrary bandit algorithm.

- [**Projects with Huggingface and Transformers**](https://github.com/ankit-kothari/Data-Science-Journey/tree/master/Natural-Language-Processing/Transformers)
    - Named Entity Recognition
    - POS Tagging
    - Classification Probelms
    - Textual Entailment
    - Creating BERT Transformer Encoder from scratch 
    
### Traing a conteextual multi-arm bandits
- In the learning phase, the algorithm,
        - Step 1 it selects an action and
        - Step 2 obtains a reward,
        - Step 3 updates its estimates of expected rewards of actions given the context.

- This is usually done via feeding a single example consisting of x,a,r (context, action, reward) to an estimator model doing a single update step (batch size 1).

- The learn phase is usually common for most algorithms and the main differences come from the exploration phase. Different Exploring-Exploitation Strategies could be
    - Epsilon Greedy
    - UCB
    - Thomson Sampling

## Evaluation



## Evaluation Results

### Epsilon Gredy
<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/eg.png" width="90%">

### UCB
<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/ucb.png" width="90%">

### Thompson Sampling
<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/ts.png" width="90%">