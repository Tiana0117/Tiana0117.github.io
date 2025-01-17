---
layout: post
title: "Blog Post 1 about Climate"
date: 2021-09-05
---
# Climate Change Visualization

Climate change is a hot topic throughout the time. Climate change is harming the planet we human beings are living on. Besides, animals living environement are also threatened, thus negatively influencing their life. So, we have to do something.

We will get started by seeing the climate change data.

## Access Data and Create database

Here we access three datasets: `temperature`,`stations`, and `countries`. We need to incorporate these datasets. 


```python
# import necessary libraries
import pandas as pd
import sqlite3
import numpy as np
from matplotlib import pyplot as plt
from plotly.io import write_html
```


```python
conn = sqlite3.connect("temps.db") 
```

Sort out the data. In this step, we want to present the dataframe in a more reasonable format. We convert the columns of temperature in 12 months to a single column with `df.stack`. 


```python
def prepare_df(df):
    """
    This function serves to clean the data. 
    We stack the columns representing 12 months into one column.
    """
    
    df["FIPS"] = df["ID"].str[0:2]
    df = df.set_index(keys=["ID", "Year","FIPS"])
    df = df.stack()
    df = df.reset_index()
    df = df.rename(columns = {"level_3"  : "Month" , 0 : "Temp" })
    df["Month"] = df["Month"].str[5:].astype(int)
    df["Temp"]  = df["Temp"] / 100
    return(df)
```

We use SQLite3 to create a database for better manipulation of data. In this step, we create a database and populate it with three tables.

**Converting Dataframe**


```python
# convert temperature data
temp_iter = pd.read_csv("temps.csv", chunksize = 100000)

for temps in temp_iter:
    temps = prepare_df(temps)
    temps.to_sql("temperatures", conn, if_exists = "append", index = False)
```


```python
# get stations data
stations_url = "https://raw.githubusercontent.com/PhilChodrow/PIC16B/master/datasets/noaa-ghcn/station-metadata.csv"
stations = pd.read_csv(stations_url)
```


```python
# convert stations to sql
stations.to_sql("stations", conn, if_exists = "replace", index = False)
```


```python
# get countries data
countries_url = "https://raw.githubusercontent.com/mysociety/gaze/master/data/fips-10-4-to-iso-country-codes.csv"
countries = pd.read_csv(countries_url)
```


```python
# rename several columns of countries to make column names more informative
countries = countries.rename(columns={"FIPS 10-4": "FIPS", 
                                      "Name": "Country"})
```


```python
# convert countries to sql
countries.to_sql("countries",conn,if_exists = "replace", index = False)
```

    /Users/yiningliang/anaconda3/envs/PIC16B/lib/python3.7/site-packages/pandas/core/generic.py:2882: UserWarning:
    
    The spaces in these column names will not be changed. In pandas versions < 0.14, spaces were converted to underscores.
    


**Getting Detailed Info of Tables**


```python
# get the detailed info of tables
cursor = conn.cursor()
cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")

for result in cursor.fetchall():
    print(result[0])
```

    CREATE TABLE "temperatures" (
    "ID" TEXT,
      "Year" INTEGER,
      "FIPS" TEXT,
      "Month" INTEGER,
      "Temp" REAL
    )
    CREATE TABLE "stations" (
    "ID" TEXT,
      "LATITUDE" REAL,
      "LONGITUDE" REAL,
      "STNELEV" REAL,
      "NAME" TEXT
    )
    CREATE TABLE "countries" (
    "FIPS" TEXT,
      "ISO 3166" TEXT,
      "Country" TEXT
    )


### Perform a SQL Query
We use a SQL Query to fetch the required data from our database. Not knowing about the SQL query? Here is a brief syntax specification.
* `SELECT` is to select the columns we want to fetch. 
* `FROM` specifies the table we fetch the columns from. 
* `LEFT JOIN` joins two tables based on a common column. 
* `ON` specifies the common column.


```python
cmd = \
"""
SELECT T.ID, T.Year, T.Month, T.Temp, T.FIPS,
S.ID, S.LATITUDE, S.LONGITUDE, S.STNELEV, S.NAME,
C.FIPS,C.Country
FROM temperatures T
LEFT JOIN stations S
ON S.ID = T.ID
LEFT JOIN countries C 
on C.FIPS = T.FIPS
"""

df_1 = pd.read_sql_query(cmd, conn)
df_1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Year</th>
      <th>Month</th>
      <th>Temp</th>
      <th>FIPS</th>
      <th>ID</th>
      <th>LATITUDE</th>
      <th>LONGITUDE</th>
      <th>STNELEV</th>
      <th>NAME</th>
      <th>FIPS</th>
      <th>Country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>1</td>
      <td>-0.89</td>
      <td>AC</td>
      <td>ACW00011604</td>
      <td>57.7667</td>
      <td>11.8667</td>
      <td>18.0</td>
      <td>SAVE</td>
      <td>AC</td>
      <td>Antigua and Barbuda</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>2</td>
      <td>2.36</td>
      <td>AC</td>
      <td>ACW00011604</td>
      <td>57.7667</td>
      <td>11.8667</td>
      <td>18.0</td>
      <td>SAVE</td>
      <td>AC</td>
      <td>Antigua and Barbuda</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>3</td>
      <td>4.72</td>
      <td>AC</td>
      <td>ACW00011604</td>
      <td>57.7667</td>
      <td>11.8667</td>
      <td>18.0</td>
      <td>SAVE</td>
      <td>AC</td>
      <td>Antigua and Barbuda</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>4</td>
      <td>7.73</td>
      <td>AC</td>
      <td>ACW00011604</td>
      <td>57.7667</td>
      <td>11.8667</td>
      <td>18.0</td>
      <td>SAVE</td>
      <td>AC</td>
      <td>Antigua and Barbuda</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACW00011604</td>
      <td>1961</td>
      <td>5</td>
      <td>11.28</td>
      <td>AC</td>
      <td>ACW00011604</td>
      <td>57.7667</td>
      <td>11.8667</td>
      <td>18.0</td>
      <td>SAVE</td>
      <td>AC</td>
      <td>Antigua and Barbuda</td>
    </tr>
  </tbody>
</table>
</div>



Make sure to close the database connection after you are finished constructing it.


```python
conn.close()
```

## Plotting A - Geographic Scatter Function
Question:
**How does the average yearly change in temperature vary within a given country?**


First, we need to write a function to prepare for the plotting. The function returns a dataframe subset from the table above based on user-specified features.

This dataframe have columns for:

* The station name.

* The latitude of the station.

* The longitude of the station.

* The name of the country in which the station is located.

* The year in which the reading was taken.

* The month in which the reading was taken.

* The average temperature at the specified station during the specified year and month. 


```python
def query_climate_database(country, year_begin, year_end, month):
    """
    The function accepts four arguments. 
    - country, a string giving the name of a country 
    for which data should be returned.
    - year_begin and year_end, two integers 
    giving the earliest and latest years 
    for which should be returned.
    - month, an integer giving the month of the year 
    for which should be returned.
    
    The return value is a Pandas dataframe of temperature readings 
    for the specified country, in the specified date range, 
    in the specified month of the year. 
    """
    prep = df_1[["NAME", "LATITUDE","LONGITUDE",
                 "Country","Year","Month","Temp"]]
    climate = prep[(prep["Country"] == country) & 
                   (prep["Year"]>=year_begin) &
                   (prep["Year"]<=year_end) & 
                   (prep["Month"] == month)]  
    return(climate)
```

Have a try on the function.


```python
india_try = query_climate_database(country = "India", 
                       year_begin = 1980, 
                       year_end = 2020,
                       month = 12)
india_try.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>NAME</th>
      <th>LATITUDE</th>
      <th>LONGITUDE</th>
      <th>Country</th>
      <th>Year</th>
      <th>Month</th>
      <th>Temp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>51329848</th>
      <td>SHILONG</td>
      <td>25.60</td>
      <td>91.89</td>
      <td>India</td>
      <td>2011</td>
      <td>12</td>
      <td>12.5</td>
    </tr>
    <tr>
      <th>51332811</th>
      <td>DARJEELING</td>
      <td>27.05</td>
      <td>88.27</td>
      <td>India</td>
      <td>1980</td>
      <td>12</td>
      <td>9.8</td>
    </tr>
    <tr>
      <th>51332821</th>
      <td>DARJEELING</td>
      <td>27.05</td>
      <td>88.27</td>
      <td>India</td>
      <td>1981</td>
      <td>12</td>
      <td>9.3</td>
    </tr>
    <tr>
      <th>51332849</th>
      <td>DARJEELING</td>
      <td>27.05</td>
      <td>88.27</td>
      <td>India</td>
      <td>1993</td>
      <td>12</td>
      <td>9.1</td>
    </tr>
    <tr>
      <th>51332869</th>
      <td>DARJEELING</td>
      <td>27.05</td>
      <td>88.27</td>
      <td>India</td>
      <td>1996</td>
      <td>12</td>
      <td>8.8</td>
    </tr>
  </tbody>
</table>
</div>



Firstly, in the plot, we want to have colored points reflecting the estimate of the yearly change in temperature during the specified month and time period at a station. To acheive this task, we need to compute the first coefficient of a linear regression model at the station. 

**So we need to build a helper function of linear regression model now.**


```python
import sklearn
from sklearn.linear_model import LinearRegression
def coef(data_group):
    x = data_group[["Year"]] # 2 brackets because X should be a df
    y = data_group["Temp"]   # 1 bracket because y should be a series
    LR = LinearRegression()
    LR.fit(x, y)
    return LR.coef_[0]
```


```python
# import necessary libraries
from plotly import express as px
from matplotlib import pyplot as plt
```

The function below outputs an interactive geographic scatterplot, constructed using Plotly Express, with a point for each station, such that the color of the point reflects an estimate of the yearly change in temperature during the specified month and time period at that station.


```python
def temperature_coefficient_plot(country, year_begin, year_end, 
                                 month, min_obs, **kwargs):
    
    # filter the dataframe according to the arguments
    prep2 = query_climate_database(country, year_begin, 
                                   year_end, month)
    count = prep2.groupby(["NAME"])[["Temp"]].aggregate(len)
    count = count[count["Temp"] >= min_obs].reset_index()
    filtered_names = count["NAME"]
    prep3 = prep2[prep2["NAME"].isin(np.array(filtered_names))]
    
    coef_df = prep3.groupby("NAME").apply(coef)
    coef_df = coef_df.reset_index()
    
    # merge the coefficient result with the filtered dataframe.
    merged_df = pd.merge(coef_df, prep3, on = ["NAME"])
    merged_df[0] = merged_df[0].round(5) # round the coefficients
    to_plot = merged_df.rename(columns = 
                               {0:"Estimated Yearly Increase in Temperature"})
    to_plot = to_plot.drop(["Year","Temp","Month"], axis = 1)
    
    # prepare for the title name
    month_dict = { 1: "January", 2: "February", 3: "March", 
                  4: "April", 
                  5: "May", 6: "June", 7: "July", 
                  8: "August", 
                  9: "September", 10: "October", 11: "November", 
                  12: "December"}
    
    # make the plot
    fig = px.scatter_mapbox(to_plot,
                            lat = "LATITUDE",
                            lon = "LONGITUDE",
                            hover_name = "NAME",
                            color = "Estimated Yearly Increase in Temperature",
                            color_continuous_midpoint = 0,
                            title = str(country)+
                            " Estimated Yearly Increase in Temperature in "+
                            str(month_dict[month]),
                            **kwargs
                            )
    return(fig)
```

Apply the function.


```python
color_map = px.colors.diverging.RdGy_r
fig = temperature_coefficient_plot("India", 1980, 2020, 1, 
                                   min_obs = 10,
                                   zoom = 2,
                                   mapbox_style="carto-positron",
                                   color_continuous_scale=color_map)

write_html(fig, "coefplot.html")
```
{% include coefplot.html %}

> Through the figure, we could get some detailed information. For instance, we could know the stations that have experienced more serious warming. When hover around the dots, we could see detailed info (name, latitude, longitude) of the station and also the estimated yearly increase in temperature.

## Plotting B - Scatter Plot of average temp
Question: 
* **What is the trend of the yearly average temperature in a country throughout a specific time period?**
* **Could I see two countries' changing trend changing in the same plot?**

We want to create a line plot in this section. This plot aims to show the average month temperature of the countries within a time period. 

First, we need to subset the dataframe according to the `country 1`, `country 2`, `year_begin`, `year_end`.


```python
def plot2_sql_query(country1, country2, year_begin, year_end):
    
    plot2 = df_1[["Country","Year","Month","Temp"]]
    filtered2 = plot2[((plot2["Country"] == country1) | 
                      (plot2["Country"] == country2)) & 
                   (plot2["Year"]>=year_begin) &
                   (plot2["Year"]<=year_end)]  
    return(filtered2)
```

Then we start to plot the scatter plot illustrating the yearly average temperature of two countries throughout a time period.

Hint: 
* Use `aggregate()` to calculate the average temperature by group of `Year` and `Country`.


```python
import seaborn as sns
def scatter_average(country1,country2, year_begin, year_end):
    myscatter = plot2_sql_query(country1, country2, 
                                year_begin, year_end)
    myscatter = myscatter.groupby(["Year","Country"])["Temp"].aggregate(np.mean)
    myscatter = myscatter.reset_index()
    
    # plot the scatterplot
    fig = sns.scatterplot(data = myscatter, x = "Year",y = "Temp", hue = "Country")
    title = "Scatterplot showing "+str(country1)+" and " +str(country2)+ " Temperature throughout"+ str(year_begin)+" and "+str(year_end)
    fig.set_title(title)
    
    return (fig)
```

Run the function.


```python
scatter_average("India", "Iceland", 1980,2020)
write_html(fig, "scatter1.html")
```


    
![output_38_1.png](/images/output_38_1.png)
   


> We could see from the plot that the average temperature in Iceland is way lower than that in India.


```python
scatter_average("China", "Japan", 1980,2020)
write_html(fig, "scatter2.html")
```


    
![output_40_1.png](/images/output_40_1.png)


> Compared with the plot of India and Iceland, we could see that the average temperature in China and Japan has a larger fluctuation. Generally, the yearly average temperature in China has increased a lot. We could also catch some detailed info from this scatter plot. For instance, in 2000, Japan presents an extremely high average temperature.

## Plotting C - Heatmap of z-scores
Question: 
* **How did the temperature every month change in a country by years?**
* **Are there any anomalities regarding the average temperature?**

First, we need to construt a list of unusually hot or unusually cold months in our dataset. According to the statistics knowledge, we need to calculate z-score, and compare its absolute value with 2. Here, we define a z-score whose aboslute value is greater than 2 as a sign of anomalies.

To calculate the z-score, we need to define a helper function.


```python
def z_score(x):
    m = np.mean(x)
    s = np.std(x)
    return (x - m)/s
```

Then, we need to subset the dataframe according to the function arguments. To acheive this task, we define another helper function, which helps to subset the dataframe.


```python
def to_heat(country, year_begin, year_end):
    filtered_heat = df_1[["NAME", "Country","Year","Month","Temp"]]
    filtered_heat = filtered_heat[(filtered_heat["Country"] == country) & 
                   (filtered_heat["Year"]>=year_begin) &
                   (filtered_heat["Year"]<=year_end)]  
    return(filtered_heat)
```

Now, we start to write the functiion for plotting the anomalies on the heatmap. In this function, data was filtered according to the comparison between z_score and +/- 2.

Hint:
* `transform()` is a good option for embedding the z_score function and generating a new column for z-scores in our dataframe.
* `pivot()` can be used to shape the dataframe with organized index/columns/values. Shaping the dataframe helps to plot the data better.


```python
def heatmap (country,year_begin, year_end):
    prep_z = to_heat(country, year_begin, year_end)
    prep_z["z"] = prep_z.groupby(["Year","Month"])["Temp"].transform(z_score)
    heat = prep_z.groupby(["Year","Month"])["z"].aggregate(np.mean)
    heat = heat.reset_index()
    heat = heat.pivot(index = "Year",
                      columns = "Month",
                      values = "z")
    
    # plot heatmap with dataframe
    fig, ax = plt.subplots(figsize= (17,13))
    title = "Heatmap of "+ str(country)+" Temperature z-scores"
    ax.set_title(title)
    fig = sns.heatmap(heat, ax = ax,linewidths=.5, 
                      cmap = "YlGnBu",center = 0)
    return (fig)
```


```python
heatmap("China",2000,2020)
write_html(fig, "heat.html")
```


    
![output_50_1.png](/images/output_50_1.png)




```python

```
