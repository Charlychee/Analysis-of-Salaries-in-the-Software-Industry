import pandas as pd
import os
from collections import defaultdict
import numpy as np
import plotly.express as px

class Technologies:
    '''
        Class containing helper functions and dataframes, dictionaries for processing different columns of dataframe
    '''
    languageDf = pd.DataFrame()
    databaseDf = pd.DataFrame()
    platformDf = pd.DataFrame()
    webFrameWorkDf = pd.DataFrame()
    toolsDf = pd.DataFrame()
    professionDf = pd.DataFrame(columns=['Professions', 'Annual Salary (in K)'])

    '''Columns of interest in the dataframe''' 
    relevantColumns = ['LanguageHaveWorkedWith', 'DatabaseHaveWorkedWith', 'PlatformHaveWorkedWith', 'WebframeHaveWorkedWith', 'ToolsTechHaveWorkedWith']
    interested_professions = ['Full-Stack Developers', 'Embedded Developers','Backend Developer', 'Frontend developers', 'Data Engineer', 'Data Scientist', 'DevOps specialist', 'QA developers']

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
        df['DevType'] = np.where(df['DevType'] == 'Developer, QA or test', 'QA developers', df['DevType'])
        df['DevType'] = np.where(df['DevType'] == 'Developer, front-end', 'Frontend developers', df['DevType'])

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

    def create_profession_df(self):
        '''
            helper function to add data to professionDf
            ProfessionDf contains information on careers with their corresponding salaries
        '''
        for prof in self.interested_professions:
            temp = self.languageDf[self.languageDf['DevType'] == prof][['DevType', 'abs_comp_k']]
            temp.rename(columns={'DevType':'Professions', 'abs_comp_k': 'Annual Salary (in K)'}, inplace=True)
            self.professionDf = self.professionDf.append(temp)

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
            self.create_profession_df()
        elif(column == 'DatabaseHaveWorkedWith'):
            self.databaseDf = bool_df
        elif(column == 'PlatformHaveWorkedWith'):
            self.platformDf = bool_df
        elif(column == 'WebframeHaveWorkedWith'):
            self.webFrameWorkDf = bool_df
        elif(column == 'ToolsTechHaveWorkedWith'):
            self.toolsDf = bool_df
        else:
            print('Wrong key')

def plotly_bar(input_dict: dict,rel_cols:list, dict_key: str, x_axis: str, y_axis: str, title: str, angle=-45):
    '''
        Helper function to return plotly figure for bar graph
        :param input_dict: input dictionary
        :type input_dict: dict
        :param x_axis: label for x_axis
        :type x_axis: str
        :param y_axis: label for y_axis
        :type y_axis: str
        :param title: title of the plot
        :type title: str
        :param angle: angle for label, default = -45
        :type angle: int
        :param k: number of entries to be shown in plot
        :type k: int
        :return: figure object
    '''
    assert(isinstance(input_dict, dict))
    assert(isinstance(x_axis, str))
    assert(isinstance(y_axis, str))
    assert(isinstance(title, str))
    assert(isinstance(angle, int))

    temp_dict = defaultdict(dict)
    for key, value in input_dict.items():
        temp_dict[key] = value[dict_key]
    ef = pd.DataFrame.from_dict(temp_dict).dropna()
    ef.reset_index(level=0, inplace=True)
    ef = ef.loc[ef['index'].isin(rel_cols)]
    ef['Full-Stack Developers'] *= 100
    ef['Backend Developer'] *= 100
    ef['Data Engineer'] *= 100
    ef['Data Scientist'] *= 100
    fig = px.bar(ef, x="index", y=['Full-Stack Developers','Backend Developer', 'Data Engineer', 'Data Scientist'], labels= {"index": x_axis, "variable":"Legend"}, title=title, barmode="group", color_discrete_sequence=['rgba(244, 128, 36, 255)', 'rgba(34, 36, 38, 255)', 'rgba(188, 187, 187, 255)', 'rgba(244, 36, 48, 255)'])
    fig.update_layout({
        'paper_bgcolor':'rgba(0,0,0,0)',
        'plot_bgcolor':'rgba(0,0,0,0)',
        'font_color':'rgba(0,0,0,255)'
    })
    fig.update_xaxes(
        tickangle=angle,
        title_font = {"size": 20},
        tickfont_size = 15,
        tickcolor = 'black',
        title_standoff = 25)
    fig.update_yaxes(
        title_text = y_axis,
        title_font = {"size": 20},
        tickfont_size = 15,
        tickcolor = 'black',
        title_standoff = 25)
    return fig

def plotly_box(input_df, x_axis: str, y_axis: str, title: str, angle=-45, k = 5):
    '''
        Helper function to return plotly figure for box plot
        :param input_dict: input dictionary
        :type input_dict: dict
        :param x_axis: label for x_axis
        :type x_axis: str
        :param y_axis: label for y_axis
        :type y_axis: str
        :param title: title of the plot
        :type title: str
        :param angle: angle for label, default = -45
        :type angle: int
        :param k: number of entries to be shown in plot
        :type k: int
        :return: figure object
    '''
    assert(isinstance(input_df, pd.DataFrame))
    assert(isinstance(x_axis, str))
    assert(isinstance(y_axis, str))
    assert(isinstance(title, str))
    assert(isinstance(angle, int))
    assert(isinstance(k, int))
    assert(k > 0)
    
    a = input_df.groupby(x_axis).median().sort_values(by=y_axis,ascending=False)[:k]
    a = a.to_dict()
    temp_df = pd.DataFrame(columns=[x_axis, y_axis])
    for key, value in a[y_axis].items():
        temp_df = temp_df.append(input_df[input_df[x_axis] == key])
            
    fig = px.box(temp_df, x=x_axis, y=y_axis, title=title,color_discrete_sequence = ['rgba(63,68,73,1)'])
    fig.update_layout({
        'paper_bgcolor':'rgba(0,0,0,0)',
        'plot_bgcolor':'rgba(0,0,0,0)',
        'font_color':'rgba(0,0,0,255)'
    })
    fig.update_xaxes(
        tickangle=angle,
        tickfont_size = 15,
        tickcolor = 'black',
        title_font = {"size": 20},
        title_standoff = 25)
    fig.update_yaxes(
        tickfont_size = 15,
        tickcolor = 'black',
        title_font = {"size": 20},
        title_standoff = 25)
    return fig

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