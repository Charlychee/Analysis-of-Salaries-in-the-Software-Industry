import pandas as pd
import os
from collections import defaultdict
import numpy as np

class Technologies:
    '''
        Class containing helper functions and dataframes, dictionaries for processing different columns of dataframe
    '''
    languageDf = pd.DataFrame()
    languageDict = defaultdict()
    databaseDf = pd.DataFrame()
    databaseDict = defaultdict()
    platformDf = pd.DataFrame()
    platformDict = defaultdict()
    webFrameWorkDf = pd.DataFrame()
    webFrameWorkDict = defaultdict()
    toolsDf = pd.DataFrame()
    toolsDict = defaultdict()

    '''Columns of interest in the dataframe''' 
    relevantColumns = ['LanguageHaveWorkedWith', 'DatabaseHaveWorkedWith', 'PlatformHaveWorkedWith', 'WebframeHaveWorkedWith', 'ToolsTechHaveWorkedWith']
    interested_professions = ['Full-Stack Developers', 'Embedded Developers','Backend Developer', 'Data Engineer', 'Data Scientist']

    def get_df_dict(self, df, final_dict, input_obj, k):
        '''
            function to create dictionary of k most popular technologies for each unique profession
            :param df: input df
            :type df: pandas dataframe
            :param final_dict: the output dictionary
            :type final_dict: defaultdict
            :param input_obj: object of class Technologies
            :type input_obj: Technologies
            :param k: value specifying number of items to be added to dict
            :type k: int
        '''
        assert(isinstance(final_dict, dict))
        assert(isinstance(input_obj, Technologies))
        assert(isinstance(k, int))
        assert(k > 0)
        for prof in self.interested_professions:
            final_dict[prof]['languages'] = input_obj.getkMostPopularTechnologies(input_obj.languageDf, prof, ['ResponseId', 'abs_comp_k', 'DevType', 'curr_symbol'], k)
            final_dict[prof]['database'] = input_obj.getkMostPopularTechnologies(input_obj.databaseDf, prof, ['ResponseId', 'abs_comp_k', 'DevType', 'curr_symbol'], k)
            final_dict[prof]['platform'] = input_obj.getkMostPopularTechnologies(input_obj.platformDf, prof, ['ResponseId', 'abs_comp_k', 'DevType', 'curr_symbol'], k)
            final_dict[prof]['webFramework'] = input_obj.getkMostPopularTechnologies(input_obj.webFrameWorkDf, prof, ['ResponseId', 'abs_comp_k', 'DevType', 'curr_symbol'], k)
            final_dict[prof]['tools'] = input_obj.getkMostPopularTechnologies(input_obj.toolsDf, prof, ['ResponseId', 'abs_comp_k', 'DevType', 'curr_symbol'], k)

    def createUSDDf(self, dataframe, avoidCols, colNames):
        '''
            Function to separate dataframe into dicts containing labels and list of salaries associated with label
            Separates into two dicts, one for Euro and one for US Dollars
            :param dataframe: input dataframe
            :type dataframe: pandas dataframe
            :param avoidCols: columns to avoid
            :type avoidCols: list of string
            :param colNames: 2 value array with column names for df
            :type colNames: list of string
            :return: two dicts for Euro and USD
        '''
        assert(isinstance(dataframe, pd.DataFrame))
        assert(isinstance(avoidCols, list))
        for item in avoidCols:
            assert(isinstance(item, str))
        resultDf = pd.DataFrame(columns=[colNames[0], colNames[1]])

        for col in dataframe.columns:
            if(col not in avoidCols):
                df = dataframe[dataframe[col] == 1][[col, 'abs_comp_k']]
                df.replace({col: 1}, str(col), inplace=True)
                df.rename(columns={col:colNames[0], 'abs_comp_k': colNames[1]}, inplace=True)
                resultDf = resultDf.append(df)
        return resultDf

    def splitString(self, inputStr):
        '''
            Helper function to split a string by ;
            :param inputStr: input string to be split
            :type inputStr: str
            :return: list of strings
        '''
        assert(isinstance(inputStr, (str, float)))
        inputStr = str(inputStr)
        return inputStr.split(';')

    def kPopularTechnologies(self, inputDf, k):
        '''
            Function that returns k most popular frameworks in dataframe df
            :param inputDf: input dataframe
            :type inputDf: pandas dataframe
            :param k: number of frameworks
            :type k: int
            :return: dict containing framework name and percentage of users
        '''
        assert(isinstance(inputDf, pd.DataFrame))
        assert(isinstance(k, int))
        assert(k <= len(inputDf.columns))
        inputDict = inputDf.mean(axis=0).to_dict()
        return dict(sorted(inputDict.items(), key=lambda item: item[1], reverse=True)[:k])

    def updateProf(self, df):
        '''
            helper function to update the professions to shorter names
            :param df: input df
            :type df: pd Dataframe
        '''
        assert(isinstance(df, pd.DataFrame))
        df['DevType'] = np.where(df['DevType'] == 'Engineer, data', 'Data Engineer', df['DevType'])
        df['DevType'] = np.where(df['DevType'] == 'Data scientist or machine learning specialist', 'Data Scientist', df['DevType'])
        df['DevType'] = np.where(df['DevType'] == 'Developer, back-end', 'Backend Developer', df['DevType'])
        df['DevType'] = np.where(df['DevType'] == 'Developer, embedded applications or devices', 'Embedded Developers', df['DevType'])
        df['DevType'] = np.where(df['DevType'] == 'Developer, full-stack', 'Full-Stack Developers', df['DevType'])

    def getkMostPopularTechnologies(self, inputDf, prof, columns_to_avoid, k):
        '''
            Function that returns k most popular technologies in dataframe df
            :param inputDf: input dataframe
            :type inputDf: pandas dataframe
            :param prof: profession
            :type prof: str
            :param columns_to_avoid: columns to remove in df
            :type columns_to_avoid: list
            :param k: number of frameworks
            :type k: int
            :return: dict containing framework name and percentage of users
        '''
        assert(isinstance(inputDf, pd.DataFrame))
        assert(isinstance(prof, str))
        assert(isinstance(columns_to_avoid, list))
        assert(isinstance(k, int))
        df = inputDf[inputDf['DevType'] == prof]
        df = df[df.columns.difference(columns_to_avoid, sort = False)]
        return self.kPopularTechnologies(df, k)

    def to_1D(self, series):
        '''
            Helper function to convert a column of lists to a 1D series
            :param series: input series
            :type series: pd.Series
            :return: pandas Series
        '''
        return pd.Series([x for _list in series for x in _list])

    def boolean_df(self, item_lists, unique_items):
        '''
            function returning a boolean pandas dataframe based on unique_items being present in item_lists
            :param item_lists: input dataframe, with each row a list of items
            :type item_lists: pd.Series
            :param unique_items: distinct values in item_lists
            :type: dict.keys()
            :return: pandas dataframe
        '''
        assert(isinstance(item_lists, pd.Series))
        bool_dict = {}
        # Loop through all the tags
        for _, item in enumerate(unique_items):
            # Apply boolean mask
            bool_dict[item] = item_lists.apply(lambda x: item in x).astype(int)
        # Return the results as a dataframe
        return pd.DataFrame(bool_dict)

    def createDfs(self,df, column):
        '''
            Function to create dataframe for different languages, platforms etc.
            :param df: input dataframe
            :type df: pandas dataframe
            :param column: type of column
            :type column: str
        '''
        assert(isinstance(df, pd.DataFrame))
        assert(isinstance(column, str))
        df[column] = df[column].map(self.splitString)
        ef = self.to_1D(df[column])
        value_dict = ef.value_counts().to_dict()
        if('nan' in value_dict):
            value_dict.pop('nan', None)
        bool_df = pd.DataFrame()
        bool_df['ResponseId'] = df['ResponseId']
        bool_df['DevType'] = df['DevType'].map(self.splitString)
        bool_df = pd.concat([bool_df, self.boolean_df(df[column], value_dict.keys())], axis=1)
        bool_df = bool_df.explode('DevType')
        bool_df['curr_symbol'] = df['curr_symbol']
        bool_df['abs_comp_k'] = df['abs_comp_k']
        self.updateProf(bool_df)

        if(column == 'LanguageHaveWorkedWith'):
            self.languageDf = bool_df
            self.languageDict = value_dict
        elif(column == 'DatabaseHaveWorkedWith'):
            self.databaseDf = bool_df
            self.databaseDict = value_dict
        elif(column == 'PlatformHaveWorkedWith'):
            self.platformDf = bool_df
            self.platformDict = value_dict
        elif(column == 'WebframeHaveWorkedWith'):
            self.webFrameWorkDf = bool_df
            self.webFrameWorkDict = value_dict
        elif(column == 'ToolsTechHaveWorkedWith'):
            self.toolsDf = bool_df
            self.toolsDict = value_dict
        else:
            print('Wrong key')


def main():
    try:
        df = pd.read_csv("../data/survey_results_public.csv")
        professionObject = Technologies()
        for column in professionObject.relevantColumns:
            professionObject.createDfs(df, column)
    except FileNotFoundError:
        print('data not found')
    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()