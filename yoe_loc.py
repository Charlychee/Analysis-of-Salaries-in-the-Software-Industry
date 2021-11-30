import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt


# Read file
file = pd.read_csv('stack-overflow-developer-survey-2021/survey_results_public.csv')

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
    This method
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

class yoe():
    # years of experience
    def convertComp2USD(self):
        yoe = file[["ResponseId","YearsCodePro","ConvertedCompYearly","Currency",'CompTotal', 'CompFreq']]

        # Set "prefer not to say" to NaN and drop
        yoe = yoe.replace("Prefer not to say", np.NaN)
        yoe = yoe.replace("Less than 1 year", 0.5)
        yoe = yoe.replace("More than 50 years", 55)
        yoe = yoe.dropna()

        yoe['YearsCodePro'] = pd.to_numeric(yoe['YearsCodePro'])
        self.yoe = wrangleComp(yoe)

    def regression(self):
        self.convertComp2USD()
        self.yoe['YearsCodePro'] = pd.to_numeric(self.yoe['YearsCodePro'])
        # show regression model and scatter plot
        fig = px.scatter(yoe, y='abs_comp_k', x='YearsCodePro', trendline="ols")
        fig.update_xaxes(
            title_text="Years of Experience",
            tickangle=-45,
            title_font={"size": 20},
            title_standoff=25)
        fig.update_yaxes(
            title_text="Annual Salary (in K)",
            title_font={"size": 20},
            title_standoff=25)
        fig.show()

class loc():

    def euro(self):
        location= file[["ResponseId", "Country","ConvertedCompYearly","Currency",'CompTotal', 'CompFreq']]

        # Set "prefer not to say" to NaN and drop
        location = location.replace("Prefer not to say", np.NaN)
        location = location.dropna()


        location = wrangleComp(location)

        # euro countries
        euro_country = location[location['Country'].isin(
            ['Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czechia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary',
             'Ireland',
             'Italy', 'Luxembourg', 'Malta', 'Netherlands',
             'Poland', 'Portugal', 'Romania', 'Slovakia', 'Slovenia', 'Spain', 'Sweden'])]

        # show plot
        order = euro_country.groupby(by=["Country"])['abs_comp_k'].median().sort_values(ascending=False).index
        order = {'Country': list(order)}

        fig = px.box(euro_country, x="Country", y="abs_comp_k", category_orders=order)
        fig.update_xaxes(
            tickangle=-45,
            title_font={"size": 20},
            title_standoff=25)
        fig.update_yaxes(
            title_text="Annual Salary (in K)",
            title_font={"size": 20},
            title_standoff=25)

        fig.show()


    def us_states(self):

        us_states = file[["ResponseId", "US_State","ConvertedCompYearly","Currency",'CompTotal', 'CompFreq']]

        us_states = us_states.replace("Prefer not to say", np.NaN)
        us_states = us_states.dropna()

        us_states = wrangleComp(us_states)

        us_states = us_states[us_states['US_State'].isin(
            ['Washington', 'California', 'New Jersey', 'New York', 'Massachusetts', 'Texas', 'Colorado', 'Hawaii'])]

        order = us_states.groupby(by=["US_State"])['abs_comp_k'].median().sort_values(ascending=False).index
        order = {'US_State': list(order)}

        fig = px.box(us_states, x="US_State", y="abs_comp_k", category_orders=order)
        fig.update_xaxes(
            title_text="US State",
            tickangle=-45,
            title_font={"size": 20},
            title_standoff=25)
        fig.update_yaxes(
            title_text="Annual Salary (in K)",
            title_font={"size": 20},
            title_standoff=25)

        fig.show()

if __name__ == '__main__':
    yoe_model = yoe()
    yoe_model.regression()
    location_model = loc()
    location_model.euro()
    location_model.us_states()









