# H1B Data Analysis (2009-2018)

- The key goal of this analysis is to identify trends among H1B data.
- How is it used by States, Employers in the Unitesd States.
- Use of H1B visa by the Inidan IT companies.
- How has policies impacted visa allocations.


N.B. This data is complied from the data available on the uscis website from 2009 to 2019, The data for 2019 is not complete so I have excluded it in most of my analysis

## Part 1: Loading and Optimization of H1B Data
- data profiling,downcasting, converting columns to the most optimized data type, droping unused columns among other techniques. 
- In the context of this file, it doesn't matter but this can be applicable and hugely benifitial for larger datasets. 

<img src="https://github.com/ankit-kothari/data_science_journey/blob/master/github_images/newplot.png" height="60%" width="60%">

## Part 2: Data Profiling, Cleaning, Manipulations and loading into a SQL Database
- Checked for null values
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

