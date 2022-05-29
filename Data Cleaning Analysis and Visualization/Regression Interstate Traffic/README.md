## Objective

- The goal of the project is to analyse `**Metro Interstate Traffic Minneapolis-St Paul, MN  westbound I-94 Volume Data**`  and perform  `**Regression analysis**` and produce a predictive model that can predict the number of cars crossing the interstate highway at any hour during the day.

## Dataset

- The dataset is taken from [Kaggle](https://www.kaggle.com/code/ramyahr/metro-interstate-traffic-volume/data)
- The data has `~48000` rows which has an hourly data and has the following attributes
    1. `holiday` Categorical US National holidays plus regional holiday, Minnesota State Fair
    2. `temp` Numeric Average temp in kelvin
    3. `rain_1h` Numeric Amount in mm of rain that occurred in the hour
    4. `snow_1h` Numeric Amount in mm of snow that occurred in the hour
    5. `clouds_all` Numeric Percentage of cloud cover
    6. `weather_main` Categorical Short textual description of the current weather
    7. `weather_description` Categorical Longer textual description of the current weather
    8. `date_time` DateTime Hour of the data collected in local CST time
    9. `traffic_volume` Numeric Hourly I-94 ATR 301 reported westbound traffic volume



## Data Exploration, Data Cleaning and Feature Engineering
    - Traffic Patterns on different Holiday and Holiday Eve’s
    - Traffic Pattern on based on Varying Temprature
    - Traffic Pattern Based on Weather
    - Traffic Patterns Based on Time Feature Engineering
    - Feature importance
    - Feature Creation that will be used for Modeling Modeling


## Modeling

### Preprocessing

- The data is divided into `Train and Test Data`
- Data is scaled using `Standard Scaler`  ***using the Train Data*** and same ***scaling is implemented on the Test Data*** which kind of gives the more symetry to data.
- Categorical Features are `one-hot` encoded
- There were three different Models
    - Random Forest Regressor
    - Poison Regressor
    - XGBoost Regressor

### Model 1: `RandomForestRegressor`

```
R2 Score: 93.42%
Mean Absolute Error: 312.9
RMSE: 511.3407
```

### Model 2: `PoissonRegressor`

```
R2 Score: 85.782%
Mean Absolute Error: 529.8141
RMSE: 752.2109
```

### Model 3: `XGBRegressor`

```
R2 Score: 93.79
RMSE: 496.61843473752737
Mean Absolute Error: 317.3705
```

## Recommendation

- The recommendation will be to use `XGBRegressor` because it has the `lowset RMSE of 496` among the three models, This says how far the predicted value is from the actual data on an `average` , It also has the `highest R2` value which explaines what %percentage of variance is expalied by the model. So given these metrics `XGBoost Regressor` fits the data the best.
- Also `XGBoost` algorithm can utilize the power of paralleel proceessing and is a versatile model which can handle missing values as compared to other algorithms which cannot proceess due to missing values.
- Sample of comparison of Actual Data vs Prediction


<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/result_regression.png" height="60%" width="60%">