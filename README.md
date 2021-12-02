# Analysis of Factors Contributing to Software Developer Salary
ECE 143 Group 5

Third-Party Modules Used:
- `numpy 1.21.2`
- `pandas 1.3.4`
- `plotly 5.1.0`

## Installation
Requires Anaconda

### Clone the Repository:
- `git clone https://github.com/Charlychee/Analysis-of-Salaries-in-the-Software-Industry`

### Create a Anaconda Virtual Environment
- `conda create --name AnalyzeSalaries --file=requirements.txt`

### Activate the Environment
- `conda activate AnalyzeSalaries`

### Unzip data.zip

### [Run](#file-structure) AnalysisOfSalariesSoftwareIndustry.ipynb

### Deactivate and Remove the Environment
- `conda deactivate`
- `conda env remove --name AnalyzeSalaries`

## File Structure
The data we used can be found in the [data zip](./data.zip). It should be unzipped for the python files and notebooks to be able to access the data CSVs inside of it. The data folder contains data surveyed and additional information provided by [StackOverflow](https://insights.stackoverflow.com/survey).

The code used to wrangle the survey data can be found in the [wrangling](./wrangling) directory.

The visualizations to our analysis can be found in the [AnalysisOfSalariesSoftwareIndustry.ipynb](./AnalysisOfSalariesSoftwareIndustry.ipynb)
