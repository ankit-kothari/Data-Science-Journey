# H1B Data Analysis (2009-2018)

- The key goal of this analysis is to identify trends among H1B data.
- How is it used by States, Employers in the Unitesd States.
- Use of H1B visa by the Inidan IT companies.
- How has policies impacted visa allocations.


N.B. This data is complied from the data available on the uscis website from 2009 to 2019, The data for 2019 is not complete so I have excluded it in most of my analysis

## [Part 1: Loading and Optimization of H1B Data](https://github.com/ankit-kothari/Data-Science-Journey/blob/master/Data_Cleaning_Analysis_and_Visualization/H1B_Visa_Analysis/Loading_and_Optimization_of_H1B_data_file.ipynb)
- data profiling,downcasting, converting columns to the most optimized data type, droping unused columns among other techniques. 
- In the context of this file, it doesn't matter but this can be applicable and hugely benifitial for larger datasets. 

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/newplot.png" height="60%" width="60%">

## [Part 2: Data Profiling, Cleaning, Manipulations and loading into a SQL Database](https://github.com/ankit-kothari/Data-Science-Journey/blob/master/Data_Cleaning_Analysis_and_Visualization/H1B_Visa_Analysis/Data%20profiling_cleaning_manipulation_and_loading_into_a_SQL_database.ipynb)
- Checked for null values, unused columns, performed general agagregate analysis on the data. 
- Added Longitude and Latitude colums using the google API for each of the states
- Cleaned duplicate name convention for same employer using pandas vectorized functions and regex.
- converted from pandas dataframe and exported into a SQL Database with the following schema

``` 
CREATE TABLE h1b (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      Fiscal_Year datetime,
      Employer text,
      Initial_Approvals INT,
      Initial_Denials int,
      Continuing_Approvals int,
      Continuing_Denials int,
      NAICS int,
      Tax_ID float,
      State  text,
      City  text,
      ZIP  int,
      long  float,
      lang  float
  );
sqlite> 
```

## [Part 3: H1B Data Analysis using SQL queries](https://github.com/ankit-kothari/Data-Science-Journey/blob/master/Data%20Exploration%20Analysis%20and%20Visualization/H1B-Data-Analysis-master/H1B_Data_Analysis_Using_MySQL.ipynb)
Exploring SQL window based functions like DENSE_RANK(), ROW_NUMBER(), Partition, CASE statements, Views and Joins to explore interesting trends in the H1B data provided by USCIS:

- Explored H1B usage across states, employers
- Trends in denial rates pre 2016 and post 2016
- Denial rates by US Employer and Indian IT companies
- Employers dominating by State

## [Part 4: H1B Data Analysis Visualization using SQL queries and plotly](https://colab.research.google.com/drive/1BREsuISGVMJiQrdBH03KlO3OpMyzqqbN?usp=sharing)

[Code for Plots and SQL queries](https://colab.research.google.com/drive/1BREsuISGVMJiQrdBH03KlO3OpMyzqqbN?usp=sharing)

[My Blog Article](https://www.linkedin.com/pulse/some-interesting-h1b-trends-insights-ankit-kothari)

- There has been a clear increase in denial rates post 2016. The increase in the denial rate is about 8% for Indian IT giants (average of top 5), whereas it was only about 1.5% for the US tech giants.
- The 5 States of California, Texas, New Jersy, New York, Illinois used 57% of the approved H1B visas. I think the policies should be made to have a fair distribution of these visas among the States, which will give a boost to the economy of smaller states. 39 of the 58 States+Islands have used less than 1% each of the total approved H1Bs from 2008-2019.
- There are ~265149 unique Employers but the top 20 employers used up to 25% of the total H1B visas from 2009-2018. I think salary should not be the only criteria to award an H1B, This is kind of unfair for smaller and medium scale enterprises who cannot afford 100k + or 150k+ salaries. They will not be able to use the so-called "Speciality Occupation" This will result in the H1B program being dominated by the top 20-50 tech giants in the US.

## [Part 5: Plotly Dash app/Dashboard](https://dash-app-h1bvisa.herokuapp.com/) 

[**Code**](https://github.com/ankit-kothari/Data-Science-Journey/blob/master/Data%20Exploration%20Analysis%20and%20Visualization/H1B-Data-Analysis-master/H1B_Dash_Dashboard.ipynb)

[**Dashboard**](https://dash-app-h1bvisa.herokuapp.com/) 

- A Dash plotly app/Dashboard which is hosted on heroku and can be used to determine various trends and visualization regarding H1B visa by selecting the Employer from the drop down visa.

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/dash.png" height="80%" width="80%">
