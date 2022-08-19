

## Multi-Arm Bandit

### Dataset: R6A - Yahoo! Front Page Today Module User Click Log Dataset, version 1.0 

- Selected dataset contains a fraction of user click log for news articles displayed in the Featured Tab of the Today   Module on Yahoo! Front Page (http://www.yahoo.com) during the first ten days in May 2009. The articles were chosen uniformly at random from a hand-picked pool of high-quality articles, which allows one to use a recently developed method to obtain an unbiased evaluation of an arbitrary bandit algorithm.

- Columns
  * timestamp: e.g., 1241160900
  * displayed_article_id: e.g., 109513
  * user_click (0 for no-click and 1 for click): e.g., 0
  * strings "|user" and "|{article_id}" indicate the start of user
  and article features
  * features are encoded as "feature_id:feature_value" pairs, and feature_id starts from 1.

- The pool of available articles for recommendation for each user visit is the set of articles that appear in that line of data. All user IDs (specifically, bcookies) are replaced by a common string 'user' so that no user information can be identified from this data.

- Each user or article is associated with six features. Feature #1 is the constant (always 1) feature, and features #2-6 correspond to the 5 membership features constructed via conjoint analysis with a bilinear model.



### Traing a multi-arm bandits
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
- I have based the evaluation on the seminal work on offline evaluation of bandits as presented in:

    [WSDM 2011] Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms https://arxiv.org/pdf/1003.5956.pdf

- In this method  input is  a bandit algorithm A and a desired number of “valid” events T on which to base the evaluation. We then step through the stream of logged events one by one. If, given the current history ht−1, it happens that the policy A chooses the same arm a as the one that was selected by the logging policy, then the event is retained (that is, added to the history), and the total payoff updated. Otherwise, if the policy A selects a different arm from the one that was taken by the logging policy, then the event is entirely ignored, and the algorithm proceeds to the next event without any change in its state.

## Evaluation Results
### Evaluation

| Sampling Strateegy | CTR (Estimated) |
| --- | --- |
| Uniform Sampling | 2.9% |
| Epsilon Gredy | 3.14% |
| UCB | 2.16% |
| Thomson Sampling | 4.11% |

### Epsilon Gredy
<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/eg.png" width="60%">

### UCB
<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/ucb.png" width="60%">

### Thompson Sampling
<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/ts.png" width="60%">

References:

    [WSDM 2011] Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms https://arxiv.org/pdf/1003.5956.pdf

"""

- [**Beta Distribution — Intuition, Examples, and Derivation**](https://towardsdatascience.com/beta-distribution-intuition-examples-and-derivation-cf00f4db57af)
- [**Bayesian Machine Learning in Python: A/B Testing**](https://www.udemy.com/course/bayesian-machine-learning-in-python-ab-testing/learn/lecture/32195648?start=0#announcements)

