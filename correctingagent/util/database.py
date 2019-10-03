from pathlib import Path

import pandas as pd
import sqlalchemy

from correctingagent.util.util import get_config


class Database(object):

    def __init__(self):
        config = get_config()
        db_location = Path(config['db_location'])
        self.db_location = db_location / 'experiments.db'
        self.engine = sqlalchemy.create_engine('sqlite:///' + self.db_location)

    def update_entry(self, experiment_id, **kwargs):
        df = self.get_df()
        for arg, val in kwargs:
            df.at[experiment_id, arg] = val
        self.save_df(df)

    def save_df(self, df):
        raise NotImplementedError("Must implement save df for this sub class")

    def get_df(self):
        raise NotImplementedError("Must implement get df for this sub class")


class BigExperimentDB(Database):

    def __init__(self):
        super(BigExperimentDB, self).__init__()
        self.experiment_db = ExperimentDB()
        self.join_db = JoinDB()

    def get_df(self):
        return pd.read_sql('big', index_col='index', con=self.engine)

    def add_experiment(self, experiment_name, status='running'):
        df = self.get_df()
        df = df.append({'experiment_name': experiment_name, 'status':status}, ignore_index=True)
        big_id = df.index[-1]
        self.save_df(df)
        return big_id

    def save_df(self, df):
        df.to_sql('big', con=self.engine, if_exists='replace')


class ExperimentDB(Database):

    def get_df(self):
        return pd.read_sql('experiments', index_col='index', con=self.engine)

    def add_experiment(self, experiment_name, colour_model_config, status='running'):
        df = self.get_df()
        df = df.append({'config_name': experiment_name, 'neural_config': colour_model_config, 'stauts':status},
                       ignore_index=True)
        experiment_id = df.index[-1]
        self.save_df(df)
        return experiment_id

    def update_entry(self, experiment_id, **kwargs):
        df = self.get_df()
        for arg, val in kwargs:
            df.at[experiment_id, arg] = val
        self.save_df(df)

    def save_df(self, df):
        df.to_sql('experiments', con=self.engine, if_exists='replace')


class JoinDB(Database):

    def get_df(self):
        return pd.read_sql('rels', index_col='index', con=self.engine)

    def add_experiment(self, big_id, experiment_id):
        df = self.get_df()
        df = df.append({'big_id':big_id, 'experiment_id':experiment_id}, ignore_index=True)
        rel_id = df.index[-1]
        self.save_df(df)
        return rel_id

    def save_df(self, df):
        df.to_sql('rels', con=self.engine, if_exists='replace')
