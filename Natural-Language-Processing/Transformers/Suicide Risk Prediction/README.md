# Suicide Risk Prediction

## About the Dataset and Credits

### How to obtain this dataset?

```markdown
# The University of Maryland Reddit Suicidality Dataset Version 2
   
********************************************************************************
THIS DATASET IS NOT PUBLICLY AVAILABLE. PLEASE DO NOT USE IT WITHOUT PERMISSION.

To request permission to use to the dataset, please see 
http://users.umiacs.umd.edu/~resnik/umd_reddit_suicidality_dataset.html or
contact Philip Resnik, resnik@umd.edu
********************************************************************************
```

### Credits for the Dataset

```markdown
## Reference:   
* Han-Chin Shing, Suraj Nair, Ayah Zirikly, Meir Friedenberg, Hal Daumé III, and
Philip Resnik, "Expert, Crowdsourced, and Machine Assessment of Suicide Risk via
Online Postings", Proceedings of the Fifth Workshop on Computational Linguistics
and Clinical Psychology: From Keyboard to Clinic, pages 25–36, New Orleans,
Louisiana, June 5, 2018.

* Ayah Zirikly, Philip Resnik, Özlem Uzuner, and Kristy Hollingshead. CLPsych
2019 Shared Task: Predicting the Degree of Suicide Risk in Reddit Posts.
Proceedings of the Sixth Workshop on Computational Linguistics and Clinical
Psychology, pages 24–33, Minneapolis, Minnesota, June 6, 2019.
```

## Task and Data

### Task

- **`Risk Assessment for SuicideWatch posters based *only* on their SuicideWatch postings.`**

### Data

- Annotations, possible values include `a`, `b`, `c`, `d`, or `None`.
    - a means No Risk,
    - b means Low-Risk,
    - c means  Moderate Risk,
    - and d means Severe Risk.
- The dataset consists of two files.
    - Two columns of CSV file (user_id and label) that contains the gold annotations for SuicideWatch and control users
    - All posts for users in *expert* with the following columns:
        - post_id
        - user_id
        - timestamp
        - subreddit
        - post_title
        - post_body

## Feature Engineering

### Length of Posts, Length of Title, Number of Posts by User_id

- The average length of text **(not cleaned)** for the post with suicide risk was around 232 words compared to 201 words for posts with no risk.
- High standard deviation was observed
- Interestingly title length was marginally longer for people with no suicide risk. The mean is 9 and 7 words for no-risk and risk, respectively.
- After analyzing, we ***filtered posts that were less than 30 words***



<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/pl1.png" width="40%">

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/tl2.png" width="40%">

### Sentence Polarity (Positive, Negative)

- Every sentence counts the number of lexica based on their category. Positive or Negative. This is taken from Stanford University Lexicon Data.
- Both the features were noisy, so we removed the positive count.
- Trying to measure the negativity in the sentence, After looking at the data, we concluded that negative (normalized) between .42 and .58 makes the model noisy and performed poorly.
- Performed a log transformation of the values and used them as a feature.
- Performed the t-test and got a p-value below 0.05, indicating the feature is significantly essential to separate the two classes.

![Before and After Transformation of Negative Count Feature ](https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/sp1.png)

Before and After Transformation of Negative Count Feature 

### Emotional Lexicon (Joy, Anger, Disgust, Sadness, etc.)

- Every sentence counts the number of lexica based on their category. This word to emotion mapping is taken from Stanford University Lexicon Data.
- According to the NHS website, the significant causes of suicide are the following.
    - Feeling of hopelessness
    - Anger
    - Depression
    - Sadness
- Individual Emotions were very noisy, so we aggregated Anger, Sadness, Fear, and Disgust as one feature.
- Squared the value and used it as the feature
- Performed t-test and got the p-value of 0.012.

![The difference in Distribution in the measure of  Aggregated Sadness Value between the two labels]

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/el.png" width="40%">


The difference in Distribution in the measure of  Aggregated Sadness Value between the two labels

|  | Experiment | Recall  | Precision | Accuracy |
| --- | --- | --- | --- | --- |
| Four Labels + No Stopwords + Only Last Comment  + Fine Tuning + Text Only  | exp1 | .46 | .38 | 0.46 |
| Four Labels + No Stopwords + Last Two Comments  + Fine Tuning + Text Only  | exp2 | .53 | .37 | 0.53 |
| Binary Labels + No Stopwords + Last Two comments + Fine Tuning + Text Only  | exp3 | .76 | .75 | .77 |
| Binary Labels + No Stopwords + Last Two comments + Fine Tuning + Text Only + 2X tokens | exp4 | .77 | .80 | .78 |
| Binary Labels + With Stopwords + Last Two Comments + Fine Tuning + Text  Only | exp5 | .85 | .86 | .85 |

## Conclusion

- When all four labels were used, The results showed that the model was not able to distinguish between `low-risk`, `moderate-risk`, and `high`-`risk` labels, and most of the labels were classified as `high risk`
- After that iteration, the problem was formulated as a binary classification problem where `low-risk`, `moderate-risk`, and `high`-`risk`  were combined to give a single label as `Severe Risk`  and `No Risk`
- It was also important to note that not only the user’s last comment on the channel but its past comment also helped improve the model.
- The results were significantly better with `STOPWORDS`  with an accuracy of `85%` as and precision of `86%` and recall of `85%`

![Confusion Matrix of the best performing Model on the test data](https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/cf.png)

Confusion Matrix of the best performing Model on the test data

## Future Work

- Experiments with Transfer Learning and trying tree-based methods
- Add other numerical feature data
- Get more data to better generalize the model 
