import pandas as pd
from env import host, username, password
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import numpy as np
def get_zillow_data():
    """
    This function connects to the zillow database and retrieves data from the properties_2017 table for
    all 'Single Family Residential' properties. The resulting DataFrame contains the bedroomcnt, bathroomcnt,
    calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, and fips columns and is returned by
    the function.
    """
   
    # create the connection url
    url = f'mysql+pymysql://{username}:{password}@{host}/zillow'

    # read the SQL query into a DataFrame
    query = '''
            SELECT 
  properties_2017.id, 
  properties_2017.parcelid, 
  properties_2017.bathroomcnt, 
  properties_2017.bedroomcnt, 
  properties_2017.calculatedfinishedsquarefeet, 
  properties_2017.fips, 
  properties_2017.lotsizesquarefeet, 
  properties_2017.propertylandusetypeid, 
  properties_2017.roomcnt, 
  properties_2017.yearbuilt, 
  predictions_2017.transactiondate, 
  properties_2017.taxamount, 
  properties_2017.taxvaluedollarcnt
FROM properties_2017
INNER JOIN predictions_2017 ON properties_2017.parcelid = predictions_2017.parcelid
INNER JOIN propertylandusetype ON properties_2017.propertylandusetypeid = propertylandusetype.propertylandusetypeid
WHERE YEAR(predictions_2017.transactiondate) = 2017
  AND propertylandusetype.propertylandusedesc = 'Single Family Residential';
            '''
    df = pd.read_sql(query, url)

    return df

# function to read from csv file
def read_csv_file():
    df = pd.read_csv('zillow_data.csv')
    return df




def prep_zillow(df):
    """
    This function takes in the Zillow DataFrame and does the following:
    - Replaces any missing values in all columns with the median value of that column
    - Filters out rows where taxvaluedollarcnt is greater than or equal to 5 million
    - Returns the cleaned DataFrame with only selected numerical columns
    """

    # Calculate the percentage of rows with missing values in all columns
    null_perc = df.isnull().mean().mean() * 100
    print(f"Percentage of rows with missing values in all columns: {null_perc:.2f}%")

    # Replace any missing values in all columns with the median value of that column
    df.fillna(df.median(), inplace=True)

    # Filter out rows where taxvaluedollarcnt is greater than or equal to 5 million
    df = df[df.taxvaluedollarcnt < 5000000]

    # Select only the numerical columns (excluding transactiondate, id, parcelid, propertylandusetypeid, and taxamount)
    num_cols = ['bathroomcnt', 'bedroomcnt', 'calculatedfinishedsquarefeet', 'fips', 'lotsizesquarefeet', 'roomcnt', 'yearbuilt', 'taxvaluedollarcnt']
    num_df = df[num_cols]

    # Calculate the percentage of data not used
    not_used_perc = (1 - len(num_df) / len(df)) * 100
    print(f"Percentage of data not used: {not_used_perc:.2f}%")

    # Return the cleaned DataFrame with only selected numerical columns
    return num_df



import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(df, target_variable):
    # Separate the target variable from the features
    X = df.drop(columns=target_variable)
    y = df[target_variable]

    # Ensure that X and y have the same number of samples
    assert len(X) == len(y), "X and y must have the same number of samples"

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Split the train set into train and validate sets
    X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.375, random_state=42)

    return X_train, y_train, X_validate, y_validate, X_test, y_test



import matplotlib.pyplot as plt
from scipy import stats

def county_sqft_test(X_train):
    """
    Performs a hypothesis test to determine if there is a significant difference
    in the mean finished square feet between different counties, using data from
    the X_train dataframe.

    Parameters:
    X_train (pandas.DataFrame): The training data containing the relevant columns.

    Returns:
    None

    """
    # Define null and alternative hypotheses
    null_hypothesis = "There is no significant difference in the mean finished square feet between different counties."
    alt_hypothesis = "There is a significant difference in the mean finished square feet between different counties."

    # Group data by county and calculate mean finished square feet for each group
    grouped_data = X_train.groupby('fips')['calculatedfinishedsquarefeet'].mean()

    # Plot a bar chart to visualize the mean finished square feet for each county
    county_names = {6037: 'Los Angeles', 6111: 'Ventura', 6059: 'Orange'}
    plt.bar([county_names[fips] for fips in grouped_data.index], grouped_data.values)
    plt.xlabel('County')
    plt.ylabel('Mean finished square feet')
    plt.show()

    # Perform a t-test to determine if the mean finished square feet for different counties are significantly different
    t_stat, p_value = stats.f_oneway(X_train.loc[X_train['fips'] == 6037, 'calculatedfinishedsquarefeet'],
                                     X_train.loc[X_train['fips'] == 6111, 'calculatedfinishedsquarefeet'],
                                     X_train.loc[X_train['fips'] == 6059, 'calculatedfinishedsquarefeet'])

    # Interpret the results of the t-test
    alpha = 0.05 # significance level
    print(f"F-statistic: {t_stat:.2f}")
    print(f"p-value: {p_value:.2f}")
    if p_value < alpha:
        print(f"Reject null hypothesis. {alt_hypothesis}")
    else:
        print(f"Fail to reject null hypothesis. {null_hypothesis}")


import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def sqft_year_test(X_train):
    """
    Performs a hypothesis test to determine if there is a significant correlation between
    calculatedfinishedsquarefeet and yearbuilt, using data from the X_train dataframe.

    Parameters:
    X_train (pandas.DataFrame): The training data containing the relevant columns.

    Returns:
    None

    """
    # Define null and alternative hypotheses
    null_hypothesis = "calculatedfinishedsquarefeet and yearbuilt have no correlation."
    alt_hypothesis = "calculatedfinishedsquarefeet and yearbuilt have significant correlation."

    # Calculate Pearson correlation coefficient and p-value
    corr, p_value = pearsonr(X_train['calculatedfinishedsquarefeet'], X_train['yearbuilt'])

    # Interpret the results of the test
    alpha = 0.05 # significance level
    print(f"Pearson correlation coefficient: {corr:.2f}")
    print(f"p-value: {p_value:.2f}")
    if p_value < alpha:
        print(f"Reject null hypothesis. {alt_hypothesis}")
    else:
        print(f"Fail to reject null hypothesis. {null_hypothesis}")

    # Create a scatter plot to visualize the relationship between the two variables
    plt.scatter(X_train['yearbuilt'], X_train['calculatedfinishedsquarefeet'])
    plt.xlabel('Year Built')
    plt.ylabel('Calculated Finished Square Feet')
    plt.show()



from sklearn.feature_selection import SelectKBest, f_regression

def select_top_k_features(X_train, y_train, k=5):
    """
    This function takes in the X_train and y_train DataFrames and selects the top K features
    using the f_regression score function and SelectKBest.
    It returns a list of the top K features.
    """
    
    # Select top K features
    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(X_train, y_train)

    # Get the indices of the top K features
    top_features = selector.get_support(indices=True)

    # Return the top K features
    return list(X_train.columns[top_features])



def plot_histogram(df, col_name):
    # Plot a histogram of the column
    plt.hist(df[col_name], bins=30, color='skyblue', edgecolor='black')
    
    # Add labels and title
    plt.xlabel(col_name)
    plt.ylabel('Count')
    plt.title(f'Distribution of {col_name}')
    
    # Show the plot
    plt.show()



def plot_scatter(df, x_col, y_col):
    plt.scatter(df[x_col], df[y_col], color='green', alpha=0.5)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'{x_col} vs. {y_col}')
    plt.show()


from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


import pandas as pd
from sklearn.metrics import mean_squared_error

def evaluate_model(y_train, y_validate):
    """
    This function evaluates two different models (mean and median) for predicting the taxvaluedollarcnt column
    and returns the root mean squared error for both the train and validate dataframes.
    
    Parameters:
    y_train (dataframe): dataframe containing the taxvaluedollarcnt column for the training data
    y_validate (dataframe): dataframe containing the taxvaluedollarcnt column for the validation data
    
    Returns:
    None
    """
    # We need y_train and y_validate to be dataframes to append the new columns with predicted values. 
    y_train = pd.DataFrame(y_train)
    y_validate = pd.DataFrame(y_validate)
    
    # 1. Predict taxvaluedollarcnt_pred_mean
    taxvaluedollarcnt_pred_mean = y_train['taxvaluedollarcnt'].mean()
    y_train['taxvaluedollarcnt_pred_mean'] = taxvaluedollarcnt_pred_mean
    y_validate['taxvaluedollarcnt_pred_mean'] = taxvaluedollarcnt_pred_mean

    # 2. compute taxvaluedollarcnt_pred_median
    taxvaluedollarcnt_pred_median = y_train['taxvaluedollarcnt'].median()
    y_train['taxvaluedollarcnt_pred_median'] = taxvaluedollarcnt_pred_median
    y_validate['taxvaluedollarcnt_pred_median'] = taxvaluedollarcnt_pred_median

    # 3. RMSE of taxvaluedollarcnt_pred_mean
    rmse_train = mean_squared_error(y_train.taxvaluedollarcnt, y_train.taxvaluedollarcnt_pred_mean)**(1/2)
    rmse_validate = mean_squared_error(y_validate.taxvaluedollarcnt, y_validate.taxvaluedollarcnt_pred_mean)**(1/2)

    print("RMSE using Mean\nTrain/In-Sample: ", round(rmse_train, 2), 
          "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))

    # 4. RMSE of taxvaluedollarcnt_pred_median
    rmse_train = mean_squared_error(y_train.taxvaluedollarcnt, y_train.taxvaluedollarcnt_pred_median)**(1/2)
    rmse_validate = mean_squared_error(y_validate.taxvaluedollarcnt, y_validate.taxvaluedollarcnt_pred_median)**(1/2)

    print("RMSE using Median\nTrain/In-Sample: ", round(rmse_train, 2), 
          "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))


def linear_regression(X_train, y_train, X_validate, y_validate):
    """
    This function takes in the features (X_train), target (y_train), 
    features (X_validate), and target (y_validate) and fits a linear 
    regression model, predicts on train and validate, and returns 
    the root mean squared error (RMSE) for both.
    """
    # create the model object
    lm = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm.fit(X_train, y_train)

    # predict train
    y_train['prediction'] = lm.predict(X_train)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.taxvaluedollarcnt, y_train.prediction)**(1/2)

    # predict validate
    y_validate['prediction'] = lm.predict(X_validate)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.taxvaluedollarcnt, y_validate.prediction)**(1/2)

    print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", rmse_train, 
          "\nValidation/Out-of-Sample: ", rmse_validate)
    
    return lm, rmse_train, rmse_validate

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures

def lasso_lars(X_train, y_train, X_validate, y_validate):
    """
    This function takes in the features (X_train), target (y_train), 
    features (X_validate), and target (y_validate), fits a LassoLars 
    model, predicts on train and validate, and returns the root mean 
    squared error (RMSE) for both.
    """
    # create the model object
    lars = LassoLars(alpha=1.0)

    # fit the model to our training data
    lars.fit(X_train, y_train)

    # predict train
    y_train['prediction'] = lars.predict(X_train)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.taxvaluedollarcnt, y_train.prediction)**(1/2)

    # predict validate
    y_validate['prediction'] = lars.predict(X_validate)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.taxvaluedollarcnt, y_validate.prediction)**(1/2)

    print("RMSE for Lasso + Lars\nTraining/In-Sample: ", rmse_train, 
          "\nValidation/Out-of-Sample: ", rmse_validate)
    
    return lars, rmse_train, rmse_validate

def glm_tweedie(X_train, y_train, X_validate, y_validate):
    """
    This function takes in the features (X_train), target (y_train), 
    features (X_validate), and target (y_validate), fits a Tweedie 
    GLM model with power=0 and alpha=0, predicts on train and validate, 
    and returns the root mean squared error (RMSE) for both.
    """
    # create the model object
    glm = TweedieRegressor(power=0, alpha=0)

    # fit the model to our training data
    glm.fit(X_train, y_train.taxvaluedollarcnt)

    # predict train
    y_train['prediction'] = glm.predict(X_train)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.taxvaluedollarcnt, y_train.prediction)**(1/2)

    # predict validate
    y_validate['prediction'] = glm.predict(X_validate)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.taxvaluedollarcnt, y_validate.prediction)**(1/2)

    print("RMSE for GLM using Tweedie, power=0 & alpha=0\nTraining/In-Sample: ", rmse_train, 
          "\nValidation/Out-of-Sample: ", rmse_validate)
    
    return glm, rmse_train, rmse_validate

def polynomial_regression(X_train, y_train, X_validate, y_validate, degree=1):
    """
    This function takes in the features (X_train), target (y_train), 
    features (X_validate), and target (y_validate), fits a polynomial 
    regression model of specified degree (default=2), predicts on train 
    and validate, and returns the root mean squared error (RMSE) for both.
    """
    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=degree)

    # fit and transform X_train
    X_train_degree = pf.fit_transform(X_train)

    # transform X_validate & X_test
    X_validate_degree = pf.transform(X_validate)

    # create the model object
    lm = LinearRegression(normalize=True)

    # fit the model to our training data
    lm.fit(X_train_degree, y_train.taxvaluedollarcnt)

    # predict train
    y_train['prediction'] = lm.predict(X_train_degree)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.taxvaluedollarcnt, y_train.prediction)**(1/2)

    # predict validate
    y_validate['prediction'] = lm.predict(X_validate_degree)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.taxvaluedollarcnt, y_validate.prediction)**(1/2)

    print(f"RMSE for Polynomial Regression, degrees={degree}\nTraining/In-Sample: {rmse_train:.2f}, \
          \nValidation/Out-of-Sample: {rmse_validate:.2f}")
    
    return lm, pf, rmse_train, rmse_validate


def polynomial_regression(X_train, y_train, X_test, y_test, degree=1):
    """
    This function takes in the features (X_train), target (y_train), 
    features (X_test), and target (y_test), fits a polynomial 
    regression model of specified degree (default=2), predicts on train 
    and test, and returns the root mean squared error (RMSE) for both.
    """
    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=degree)

    # fit and transform X_train
    X_train_degree = pf.fit_transform(X_train)

    # transform X_test
    X_test_degree = pf.transform(X_test)

    # create the model object
    lm = LinearRegression(normalize=True)

    # fit the model to our training data
    lm.fit(X_train_degree, y_train.taxvaluedollarcnt)

    # predict train
    y_train_pred = lm.predict(X_train_degree)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.taxvaluedollarcnt, y_train_pred)**(1/2)

    # predict test
    y_test_pred = lm.predict(X_test_degree)

    # evaluate: rmse
    rmse_test = mean_squared_error(y_test.taxvaluedollarcnt, y_test_pred)**(1/2)

    print(f"RMSE for Polynomial Regression, degrees={degree}\nTraining/In-Sample: {rmse_train:.2f}, \
          \nTesting/Out-of-Sample: {rmse_test:.2f}")
    
    return lm, pf, rmse_train, rmse_test, y_test_pred