import pandas as pd
import numpy as np


class Wrangle_GEA:
    """
    Wrangle age, gender and ethnicity
    """

    def wrangleGEA(self, df):
        """
        Pull out Gende, Ethincity and Age data
        :param df: The dataframe after salary conversion
        :return: Gender, Age and Ethinicity dataframe
        """
        assert isinstance(df, pd.DataFrame)
        GEA = df[["ResponseId","Age","Gender","Ethnicity",'abs_comp_k']]
        GEA = GEA.replace("Prefer not to say",np.NaN)
        GEA = GEA.replace("Or, in your own words:",np.NaN)
        GEA = GEA.dropna()
        return GEA
    
    def wrangleGender(self, df):
        """
        Wrangle gender data
        :param df: Gender, Age and Ethinicity dataframe
        :return: Dataframe contains Gender, ResponseId and salary 
        """
        assert isinstance(df, pd.DataFrame)
        gender = df[["ResponseId", "Gender","abs_comp_k"]]
        gender["Gender"]=np.where(gender['Gender'].isin(['Man','Woman']), gender['Gender'], 'Non-binary, genderqueer, or gender non-conforming')
        gender = gender.replace("Non-binary, genderqueer, or gender non-conforming","Non-binary")
        return gender
    
    def wrangleAge(self, df):
        """
        Wrangle age data
        :param df: Gender, Age and Ethinicity dataframe
        :return: Dataframe contains Age, ResponseId and salary 
        """
        assert isinstance(df, pd.DataFrame)
        age = df[["ResponseId","Age","abs_comp_k"]]
        return age
    
    def wrangleEthnicity(self, df):
        """
        Wrangle ethnicity data
        :param df: Gender, Age and Ethinicity dataframe
        :return: Dataframe contains Ethinicity, ResponseId and salary 
        """
        assert isinstance(df, pd.DataFrame)
        ethnic= df[["ResponseId","Ethnicity","abs_comp_k"]]
        ethnic['Ethnicity']=np.where(ethnic['Ethnicity'].isin(['White or of European descent','Hispanic or Latino/a/x','South Asian','Middle Eastern','Black or of African descent','East Asian','Southeast Asian',"I don't know"]), ethnic['Ethnicity'], 'Multiracial')
        ethnic = ethnic.replace("I don't know",np.NaN)
        ethnic=ethnic.dropna()
        return ethnic
    
    def wrangleGE(self, gender, ethnicity):
        """
        Merge and wrangle ethnicity and gender data
        :param gender: Gender dataframe
        :param ethnicity: Ethnicity dataframe
        :return: Merged gender and ethnicity dataframe based on responseID
        """
        assert isinstance(gender, pd.DataFrame)
        assert isinstance(ethnicity, pd.DataFrame)
        genderandethnic = pd.merge(gender,ethnicity,on="ResponseId")
        genderandethnic = genderandethnic.replace("Non-binary",np.NaN)
        genderandethnic = genderandethnic.dropna()
        genderandethnic['GenderAndEth'] = tuple(zip(genderandethnic.Gender, genderandethnic.Ethnicity))
        genderandethnic=genderandethnic.drop([ 'abs_comp_k_x'], axis=1)
        genderandethnic = genderandethnic.rename(columns={ 'abs_comp_k_y': 'abs_comp_k'})
        return genderandethnic
    
    def median(self,genderandethnic,race):
        """
        calculate median difference of men and women
        :param genderandethnic:
        :return: array of median difference
        """
        assert isinstance(genderandethnic, pd.DataFrame)
        assert isinstance(race, list)
        median_man = []
        median_woman = []
        median = []
        for r in race:
            median_man.append(genderandethnic.loc[genderandethnic['Ethnicity'] == r].loc[genderandethnic['Gender'] == 'Man']['abs_comp_k'].median())
            median_woman.append(genderandethnic.loc[genderandethnic['Ethnicity'] == r].loc[genderandethnic['Gender'] == 'Woman']['abs_comp_k'].median())
        median = np.array(median_man)-np.array(median_woman)
        return median
    
if __name__ == '__main__':
    Wrangle_GEA.load()