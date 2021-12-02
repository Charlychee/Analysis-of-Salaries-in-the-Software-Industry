import pandas as pd


class Wrangling:
    """
    This class performs data wrangling for different columns in the dataset
    """

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
