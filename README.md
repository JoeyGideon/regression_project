
Final Project for Regression


Project Goals
Construct an ML Regression model that predicts propery tax assessed values ('taxvaluedollarcnt') of Single Family Properties using attributes of the properties.

Find the key drivers of property value for single family properties. Some questions that come to mind are:

Why do some properties have a much higher value than others when they are located so close to each other?
Why are some properties valued so differently from others when they have nearly the same physical attributes but only differ in location?
Is having 1 bathroom worse for property value than having 2 bedrooms?
Deliver a report that the data science team can read through and replicate, understand what steps were taken, why and what the outcome was.

Make recommendations on what works or doesn't work in predicting these homes' values.

Project Description

Find what drives property value and how to best predict propery values via zillow data.

Project planning (lay out your process through the data science pipeline)
Aquire data from zillow dataframe make a local csv file.
View the data to see what I am working with.
Prep the data to get rid of nulls and such.
Split the data into train validate and test.
Scale the data if needed.
Explore the train dataset using visuals and stats tests to see if my hypothosis are true or not.
Evaluate the data using regression models and feature engineering.
Model the data and use it on the test dataset.
Create the presentation I am going to give.

Initial hypotheses and/or questions you have of the data, ideas
Fips has a direct correlation with property value.
calculatedfinishedsquarefeet has a direct correlation with property value.

Data dictionary
| Feature | Definition | Data Type | 
| --- | --- | --- |
| id | row index number, range: 0 - 2985216 | int64 |
| parcelid | Unique numeric id assigned to each property: 10711725 - 169601949  | int64 |
| bathroomcnt | Number of bathrooms a property has: 0 - 32 | float64 | 
| bedroomcnt | Number of bedrooms a property has: 0 - 25  | float64 |
| calculatedfinishedsquarefeet | Number of square feet of the property: 1 - 952576 | float64 |
| fips | [(FIPS)] Five digit number of which the first two are the FIPS code of the state to which the county belongs. Leading 0 is removed from the data: 6037=Los Angeles County, 6059=Orange County, 6111=Ventura County | float64 |
| lotsizesquarefeet |The land the property occupies in squared feet : 100 - 371000512 | float64 |
| propertylandusetypeid | Unique numeric id that identifies what the land is used for: the 261=Single Family Residential, 262=Rural Residence, 273=Bungalow | float64 |
| roomcnt | Total number of rooms in the principal residence | float64 |
| yearbuilt | Year the property was built | float64 |
| transactiondate| The most recent date the property was sold: yyyy-mm-dd | object |
 
| Target | Definition | Data Type |
| --- | --- | --- |
| taxamount | The total property tax assessed for that assessment year | float64 |
| taxvaluedollarcnt |The total tax assessed value of the parcel | float64 |


Instructions or an explanation of how someone else can reproduce your project and findings (What would someone need to be able to recreate your project on their own?)

USE CORRECT IMPORTS
HAVE BASIC KNOWLEDGE OF CODING
AQUIRE DATA
LOOK DATA OVER
EXPLORE DATA MAKE VIZUALS AND RUN STATS TESTS AS NEEDED
DECIDE WHICH FEATURES YOU WANT TO USE IN MODELING
GET A BASELINE
CREATE MODELS RUNNING THEM ON THE TRAIN AND VALIDATE DATASETS
CHOOSE THE BEST MODEL AND RUN IT ON THE TEST DATASET
COME UP WITH A SUMMARY AND RECOMMENDATIONS

Key findings, recommendations, and takeaways from your project.

Summary and Recommendations

You can see that using fips(county), yearbuilt, and calculatedfinishedsquarefeet created a model slightly better than baseline, but not by much.

We may be able to use this model to predict property value slightly but I wouldn't use it if it's purpose is to make revenue in anyway.

I would recommend trying to locate the house more accuratly/precise than using fips because even within counties you have very high cost areas, vs very low cost areas from block to block. 

There are also many other attributes that contribute to property value, like flood/fire areas, or even what view the property may or may not have.

I think in the future this can be tailored in a MLM from block to block somehow to give a better prediction of property value. 