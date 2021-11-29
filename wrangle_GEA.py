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
        Convert salary to anual
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

    def wrangleGEA(self, df):
        """
        Pull out Gende, Ethincity and Age data
        :param df:[pd.DataFrame] Input Dataframe
        :return: [pd.DataFrame] Output Dataframe with wrangled columns
        """
        GEA = df[["ResponseId","Age","Gender","Ethnicity",'abs_comp_k']]
        GEA = GEA.replace("Prefer not to say",np.NaN)
        GEA = GEA.replace("Or, in your own words:",np.NaN)
        GEA = GEA.dropna()
        return GEA
    
    def wrangleGender(self, df):
        """
        Wrangle gender data
        :param df:[pd.DataFrame] Input Dataframe
        :return: [pd.DataFrame] Output Dataframe with wrangled columns
        """
        gender = df[["ResponseId", "Gender","abs_comp_k"]]
        gender["Gender"]=np.where(gender['Gender'].isin(['Man','Woman']), gender['Gender'], 'Non-binary, genderqueer, or gender non-conforming')
        return gender
    
    def wrangleAge(self, df):
        """
        Wrangle age data
        :param df:[pd.DataFrame] Input Dataframe
        :return: [pd.DataFrame] Output Dataframe with wrangled columns
        """
        age = df[["ResponseId","Age","abs_comp_k"]]
        return age
    
    def wrangleEthnicity(self, df):
        """
        Wrangle ethnicity data
        :param df:[pd.DataFrame] Input Dataframe
        :return: [pd.DataFrame] Output Dataframe with wrangled columns
        """
        ethnic= df[["ResponseId","Ethnicity","abs_comp_k"]]
        ethnic['Ethnicity']=np.where(ethnic['Ethnicity'].isin(['White or of European descent','Hispanic or Latino/a/x','South Asian','Middle Eastern','Black or of African descent','East Asian','Southeast Asian',"I don't know"]), ethnic['Ethnicity'], 'Multiracial')
        ethnic = ethnic.replace("I don't know",np.NaN)
        ethnic=ethnic.dropna()
        return ethnic
    
    def wrangleGE(self, gender, ethnicity):
        """
        Wrangle ethnicity data
        :param df:[pd.DataFrame] Input Dataframe
        :return: [pd.DataFrame] Output Dataframe with wrangled columns
        """
        genderandethnic = pd.merge(gender,ethnicity,on="ResponseId")
        genderandethnic = genderandethnic.replace("Non-binary, genderqueer, or gender non-conforming",np.NaN)
        genderandethnic = genderandethnic.dropna()
        genderandethnic['GenderAndEth'] = tuple(zip(genderandethnic.Gender, genderandethnic.Ethnicity))
        genderandethnic=genderandethnic.drop([ 'abs_comp_k_x'], axis=1)
        genderandethnic = genderandethnic.rename(columns={ 'abs_comp_k_y': 'abs_comp_k'})
        return genderandethnic
    
    def median(self,genderandethnic,race):
        """
        calculate median difference of men and women
        :param genderandethnic:[pd.DataFrame] Input Dataframe
        :return: array of median difference
        """
        median_man = []
        median_woman = []
        median = []
        for r in race:
            median_man.append(genderandethnic.loc[genderandethnic['Ethnicity'] == r].loc[genderandethnic['Gender'] == 'Man']['abs_comp_k'].median())
            median_woman.append(genderandethnic.loc[genderandethnic['Ethnicity'] == r].loc[genderandethnic['Gender'] == 'Woman']['abs_comp_k'].median())
        median = np.array(median_man)-np.array(median_woman)
        return median
    
if __name__ == '__main__':
    Wrangling.load()