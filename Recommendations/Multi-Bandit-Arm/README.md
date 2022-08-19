

## Multi-Arm Bandit

### Dataset: R6A - Yahoo! Front Page Today Module User Click Log Dataset, version 1.0 

- Selected dataset contains a fraction of user click log for news articles displayed in the Featured Tab of the Today   Module on Yahoo! Front Page (http://www.yahoo.com) during the first ten days in May 2009. The articles were chosen uniformly at random from a hand-picked pool of high-quality articles, which allows one to use a recently developed method to obtain an unbiased evaluation of an arbitrary bandit algorithm.


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
| Epsilon Gredy | 1 % |
| UCB | 2.18% |
| Thomson Sampling | 4% |

### Epsilon Gredy
<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/eg.png" width="60%">

### UCB
<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/ucb.png" width="60%">

### Thompson Sampling
<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/ts.png" width="60%">