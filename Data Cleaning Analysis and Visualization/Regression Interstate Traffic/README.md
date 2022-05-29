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

## Data Exploration and Data Cleaning

### Traffic Patterns on different Holiday and Holiday Eve’s

- The `Holiday` column had was transfomred into `Holidays`(named as Individual  Public Holiday) , `Holiday_eve's` and `no_holiday`
- A `feature` was added to mark for `Holiday eve's` as we saw differnt traffic patterns on the eve of a pubilc holiday.
    - The ***mean*** Traffic Volume on **Public Holidays** ***865 cars per hour***
    - The ***mean*** Traffic Volume on **Public Holidays EVE** ***2699 cars per hou***r
    - The ***mean*** Traffic Volume on **Regular Days *3279 cars per hour***

![Average Traffic Volume for Public Holidays Starting of the calendar year to the end](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/8741e7e9-a8aa-44d8-bc63-93f5e3e7b3c1/Screen_Shot_2022-04-25_at_3.34.00_PM.png)

Average Traffic Volume for Public Holidays Starting of the calendar year to the end

### ****Traffic Pattern on based on Varying Temprature****

- `Cleaned up` the **outlier** which was skewing the distribution, It was possible due to a manual error.

![Distribution with the outlier ](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/db786be5-96d7-4f7a-819a-c2f382d2a92e/Screen_Shot_2022-04-25_at_3.41.44_PM.png)

Distribution with the outlier 

- The `Mean` Temprature on Interstate highway was `281 K` and the `max`  was `310 K`
    
    ![Distribution without the outlier](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/f58e6d99-72f9-4498-aeec-1bf04fc5ca81/Screen_Shot_2022-04-25_at_3.41.55_PM.png)
    
    Distribution without the outlier
    

- The following is the `temprature distribution` , Its peaking around `274 K` and `290K` thats where the peak or the avergae temprature around on the interstate which attracts the `high-volume` traffic as high as `4500-4999` cars per hour.
- The `median` traffic volume on the Interstate Highway is `3380` cars per hour.
- The `75% quartile` is `4933` cars per hour
- So when the temp is between 274 and 275.999 ,
    - `22%` of the time  traffic volume will be `higher than`  4999 cars per hour.
    - `13%` of the time  traffic volume will be between 4500 and 4999 cars per hour.
    - `9%` of the time traffic volume will be 
    etween   4000 and 4500 cars per hour.

![Screen Shot 2022-04-25 at 3.53.17 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/76c92855-8982-440f-ac12-85bea0502c5c/Screen_Shot_2022-04-25_at_3.53.17_PM.png)

### Traffic Pattern Based on Weather

- Both `snow` and `rain` features were have a very `skewed`  distribution and even after trying `cube root` and `sqare root`, `exp` transformations, the data has most of the rows as `0` , So I decided to `drop` these features
- Also  the `weatheer_main` column and `weather_description` column are redundant so I have decided to `drop` the `weather_description` column

![Histogram for `snow` feature ](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2f89af29-dc9e-42b3-bc20-00d8f2455f4e/Screen_Shot_2022-04-25_at_4.31.50_PM.png)

Histogram for `snow` feature 

- Average Traffic Volume During Different Weather Conditions
    
    ![Screen Shot 2022-04-25 at 4.35.12 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9e5cf050-a57a-4360-a2fb-3afde2dc1324/Screen_Shot_2022-04-25_at_4.35.12_PM.png)
    

![Histogram for `rain` feature ](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/45ab196e-97fb-4aa6-bda0-c03d4d791aa8/Screen_Shot_2022-04-25_at_4.32.00_PM.png)

Histogram for `rain` feature 

- Table for average volume with weather conditions
    
    ![Screen Shot 2022-04-25 at 4.37.06 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e7077eb2-c0d8-4208-a749-c3754d19c1d8/Screen_Shot_2022-04-25_at_4.37.06_PM.png)
    

- It can be seen that during `Squall` and `Fog` the average traffic is significantly less than the average.
- Surprisingly during `Thunderstorm` and `Rain` and `Clouds` traffic volume is higher than average
- The below chart shows the `percentage` weather conditions that prevailed on the Interstate.
    - `31.4 %` of the time it was `cloudy`
    - followed by `27.4%` it was `clear`
    - and `squall` was witnessed only on `0.01%`

![Screen Shot 2022-04-25 at 4.41.37 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d4da585a-8ae1-4365-9ec5-091d187195e4/Screen_Shot_2022-04-25_at_4.41.37_PM.png)

### Traffic Patterns Based on Time

- The `hour` was divided into five timezone
    - `Morning` : 5 am to 11:59 am
    - `Afternoon` : 12 pm to 3:59 pm
    - `Evening` : 4pm to 7:59pm
    - `Night` : 8 pm to 11:59 pm
    - `Late_Night` : midnight to 4.59 am
- Plot of traffic volume in different time-zones also comparing regular days vs holiday and holiday eves.
- Traffic during `Morning` and `Evening` , `afternoon` seems to be the most.
- It is evident; during holidays in each of the time-zone is lesser than `regular days` except for `Late_night` where traffic on holidays increases which is quite expected.
- The graph on the (right) compares the `traffic volume` across days; Also evident, traffic on `Saturday and Sunday` seems to be significantly lower.

![Comparing Traffic Volume across `different time-zone`](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ad72bce1-256a-4b6d-bb9c-c0e8f2d19806/Screen_Shot_2022-04-25_at_4.46.21_PM.png)

Comparing Traffic Volume across `different time-zone`

![Comparing Traffic Volume across different `days of the week`](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ed8f417b-786a-4791-bb0a-56368ee55880/Screen_Shot_2022-04-25_at_4.46.35_PM.png)

Comparing Traffic Volume across different `days of the week`

- Another Interesting aspect is that the `traffic volume increases in durinng the year` and it has been seen across the year; This is potentially the last quarter has a lot of holidays due to thanksgiving and christmas having more people using the interstate.
- This has remained consistent across 5 years.

![Average Volume by Quarter and Year ](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1c79924d-ea27-4f65-99ee-1cfd00ac27f3/Screen_Shot_2022-04-25_at_4.57.22_PM.png)

Average Volume by Quarter and Year 

## Feature Engineering

### Feature importance

- Using Label Encoder encode the data and using Random Forest ideentified the feature Importance
- It can be seen that `hour` , `weekend_saturday` and `weekend_sunday` along with temprature are the most important features
    
    ![Screen Shot 2022-04-25 at 5.03.55 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/9631d963-40cd-4dce-b1aa-da01e23eb44c/Screen_Shot_2022-04-25_at_5.03.55_PM.png)
    
- Using One Hot Encoder and using Random Forest ideentified the feature Importance in a more `Exploded View`
- This also shows that importance of `hour of the day` and `weekend`, and `temprature`  is th major factors contributing to the traffic volume.

![Screen Shot 2022-04-25 at 5.04.13 PM.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2f9ea32b-eb31-4d2e-8acc-5ce374d2b039/Screen_Shot_2022-04-25_at_5.04.13_PM.png)

### Feature Creation that will be used for Modeling

- `target` = traffic_volume
- `categorical variables`
    - ***hour*** - one hot encoding of the hour column,
    - ***month*** - one hot encoding of the month column,
    - ***weather_main*** - one hot encoding of weather column  (ex: fog, cloud, clear ..etc)
- `numerical variables`
    - ***temp*** -  Temprature in Kelvin
    - ***clouds_all*** - Numeric Perceentage of Cloud Cover
    - ***is_weekend*** - Binary Column (Feature Created)
    - ***is_public_holiday*** - Binary Column (Feature Created)
    - ***is_public_holiday_eve*** - Binary Column (Feature Created)

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

![Comparison of Actual and Predicted Traffic Volume](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/12ca4f13-788a-46c2-ad93-2a398423f0eb/Screen_Shot_2022-04-25_at_5.25.56_PM.png)

Comparison of Actual and Predicted Traffic Volume

## Summary

- Current Statistics
    - The ***mean*** Traffic Volume on **Public Holidays** ***865 cars per hour***
    - The ***mean*** Traffic Volume on **Public Holidays EVE** ***2699 cars per hou***r
    - The ***mean*** Traffic Volume on **Regular Days *3279 cars per hour***
- The `temprature distribution` , Its peaking around `274 K` and `290K` thats where the peak or the avergae temprature around on the interstate which attracts the `high-volume` traffic as high as `4500-4999` cars per hour.
- The `traffic volume increases in durinng the year` and it has been seen across multiple year; This is potentially the last quarter has a lot of holidays due to thanksgiving and christmas having more people using the interstate.
- The ***four*** biggest factors that affect the traffic on the Interstate highway
    - ***Hour of the day***
    - ***Weekends tend to be lower***
    - ***temprature***
    - ***and month*** (relatively ress important but might )
- The `prediction model` we have developed is robust and predict to the tune of average error of +-496.
    - This can further be improved by using more sophisticated time series model or deep leearning.

## Bibliography

[https://jpvt.github.io/post/metro_traffic_volume/](https://jpvt.github.io/post/metro_traffic_volume/)

[https://www.kaggle.com/code/ramyahr/metro-interstate-traffic-volume/data](https://www.kaggle.com/code/ramyahr/metro-interstate-traffic-volume/data)

[https://www.kaggle.com/code/ramyahr/metro-interstate-traffic-volume/notebook](https://www.kaggle.com/code/ramyahr/metro-interstate-traffic-volume/notebook)