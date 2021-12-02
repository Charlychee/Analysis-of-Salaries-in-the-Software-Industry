import numpy as np
import pandas as pd


def wrangleYearsExperience(df):
    '''

    :param df: [pd.DataFrame] Input Dataframe
    :return: A dataframe with year of experience
    '''

    assert isinstance(df, pd.DataFrame), "Input has to be a pandas dataframe"
    assert all(x in df.columns.values for x in ["ResponseId","YearsCodePro","ConvertedCompYearly","Currency",'CompTotal', 'CompFreq'])
    
    yoe = df[["ResponseId","YearsCodePro","ConvertedCompYearly","Currency",'CompTotal', 'CompFreq']]
    yoe = yoe.replace("Prefer not to say",np.NaN)
    yoe=yoe.replace("Less than 1 year",0.5)
    yoe = yoe.replace("More than 50 years",np.NaN)
    yoe = yoe.dropna()
    yoe['YearsCodePro'] = pd.to_numeric(yoe['YearsCodePro'])

    return yoe


def wrangleLocation(df):
    '''

    :param df: [pd.DataFrame] Input Dataframe
    :return: A dataframe with US states
    '''

    assert isinstance(df, pd.DataFrame), "Input has to be a pandas dataframe"
    assert all(x in df.columns.values for x in ["ResponseId", "US_State", "ConvertedCompYearly", "Currency", 'CompTotal', 'CompFreq'])
    
    us_states = df[["ResponseId", "US_State", "ConvertedCompYearly", "Currency", 'CompTotal', 'CompFreq']]
    us_states = us_states.replace("Prefer not to say", np.NaN)
    us_states = us_states.dropna()
    us_states = us_states[us_states['US_State'].isin(
        ['Washington', 'California', 'New Jersey', 'New York', 'Massachusetts', 'Texas', 'Colorado', 'Hawaii'])]
    return us_states
