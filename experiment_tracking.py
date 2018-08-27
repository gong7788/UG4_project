import sqlalchemy
import pandas as pd
from evaluation import ResultsFile

def read_experiments():
    engine = sqlalchemy.create_engine('sqlite:///db/experiments.db')
    df = pd.read_sql('experiments', index_col='index', con=engine)
    return df

def get_results_file(df, index):
    name = df.at[index, 'experiment_file']
    return ResultsFile(name=name)
