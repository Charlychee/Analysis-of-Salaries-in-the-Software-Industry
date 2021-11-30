import pandas as pd
import numpy as np

class Wrangling:
    """
    This class performs data wrangling for different columns in the dataset
    """

    def removeOutliers(self, df, col_name):
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

    def wrangleComp(self, df):
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
        df_comp = self.removeOutliers(df_comp, 'abs_comp_k')
        return df_comp

    def wrangleEdLevel(self, df):
        """
        This method wrangles the column 'EdLevel'
        :param df: [pd.DataFrame] Input Dataframe
        :return: [pd.DataFrame] Output Dataframe with wrangled columns
        """
        assert isinstance(df, pd.DataFrame), "Input has to be a pandas dataframe"
        assert 'EdLevel' in df.columns.values, "Dataframe is missing the column 'EdLevel'"
        df_edu = df.dropna(subset=['EdLevel'])
        df_edu['edu_level'] = df_edu.EdLevel.str.partition(sep='(', expand=True)[0]
        df_edu = df_edu.replace(to_replace={'Some college/university study without earning a degree': \
                                                "College/University without degree"})
        df_edu = df_edu[df_edu.edu_level.isin(["Master’s degree ",
                                "Bachelor’s degree ", "Other doctoral degree ",
       "Associate degree ", "College/University without degree",
       "Professional degree "])]
        return df_edu

    def wrangleOrgSize(self, df):
        """
        This method wrangles the column 'OrgSize'
        :param df: [pd.DataFrame] Input Dataframe
        :return: [pd.DataFrame] Output Dataframe with wrangled columns
        """
        assert isinstance(df, pd.DataFrame), "Input has to be a pandas dataframe"
        assert 'OrgSize' in df.columns.values, "Dataframe is missing the column 'OrgSize'"
        df_org = df.dropna(subset=['OrgSize'])
        df_org = df_org[df_org.OrgSize != "I don’t know"]
        df_org = df_org.replace(to_replace={'Just me - I am a freelancer, sole proprietor, etc.': \
                                                "Freelancer / Sole Proprietor"})
        return df_org

    def load(self):
        """

        :return: [pd.DataFrame] Returns the csv dataset as a dataframe
        """
        file_path = "./data/survey_results_public.csv"
        try:
            df = pd.read_csv(file_path)
            return df
        except FileNotFoundError:
            print('data not found')
        except Exception as e:
            print(e)


if __name__ == '__main__':
    Wrangling.load()
