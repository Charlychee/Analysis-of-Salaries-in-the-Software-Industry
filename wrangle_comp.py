import pandas as pd
import numpy as np


def removeOutliers(df, col_name):
    """
    This method removes outliers which based on q1 - (1.5 * IQR) and
    q3 - (1.5 * IQR)
    :param df: [pd.DataFrame] Input Dataframe
    :param col_name: [str] Column name whose outliers are to be removed
    :return: [pd.DataFrame] Dataframe with outliers removed
    """
    assert isinstance(df, pd.DataFrame), "Input has to be a pandas dataframe"
    assert isinstance(col_name, str), f"{col_name} has to be a string"
    assert col_name in list(df.columns.values), f"{col_name} is not a column in the dataframe"
    sorted_vals = df[col_name].sort_values()
    q1 = np.percentile(sorted_vals, 25)
    q3 = np.percentile(sorted_vals, 75)
    IQR = q3 - q1
    lwr_bound = q1 - (1.5 * IQR)
    upr_bound = q3 + (1.5 * IQR)
    df = df[(((df[col_name] < lwr_bound) | (df[col_name] > upr_bound)) == False)]
    return df


def wrangleComp(df):
    """
    This method wrangles the 'CompTotal' column present in df. It removes the missing values,
    filters for currency in 'USD' and converts it into annual compensation in thousands
    :param df:[pd.DataFrame] Input Dataframe
    :return: [pd.DataFrame] Output Dataframe with wrangled columns
    """
    assert isinstance(df, pd.DataFrame), "Input has to be a pandas dataframe"
    assert all(x in df.columns.values for x in ['CompTotal', 'Currency', 'CompFreq'])
    df_comp = df.dropna(subset=['CompTotal'])
    # extracting currency symbol from 'Currency' column
    df_comp['curr_symbol'] = df_comp['Currency'].str[:3]

    # keeping only USD
    df_comp = df_comp[df_comp['curr_symbol'].isin(['USD'])]
    df_comp['abs_comp'] = df_comp['CompTotal'].copy()

    # adjusting CompTotal according to frequency
    df_comp['abs_comp'] = np.where(df_comp['CompFreq'] == 'Weekly',
                                   df_comp['CompTotal'] * 52,
                                   df_comp['abs_comp'])
    df_comp['abs_comp'] = np.where(df_comp['CompFreq'] == 'Monthly',
                                   df_comp['CompTotal'] * 12,
                                   df_comp['abs_comp'])
    # Converting compensation into thousands
    df_comp['abs_comp_k'] = df_comp['abs_comp'] / 1000
    df_comp = removeOutliers(df_comp, 'abs_comp_k')
    return df_comp
