from pathlib import Path

import pandas as pd
import sqlalchemy

from correctingagent.util.util import get_config


class Database(object):

    def __init__(self):
        config = get_config()
        db_location = Path(config['db_location'])
        self.db_location = db_location / 'experiments.db'
        self.engine = sqlalchemy.create_engine('sqlite:///' + str(self.db_location))

    def update_entry(self, experiment_id, **kwargs):
        df = self.get_df()
        for arg, val in kwargs.items():
            df.at[experiment_id, arg] = val
        self.save_df(df)

    def save_df(self, df):
        df.to_sql(self.db_name, con=self.engine, if_exists='replace')

    def get_df(self, show_only_done=False):
        df = pd.read_sql(self.db_name, index_col='index', con=self.engine)
        if show_only_done:
            df = df[df.status == 'done']
        return df

    def get_entry(self, id):
        df = self.get_df()
        return df.loc[id]


class BigExperimentDB(Database):

    def __init__(self):
        super(BigExperimentDB, self).__init__()
        self.experiment_db = ExperimentDB()
        self.join_db = JoinDB()
        self.db_name = 'big'

    def add_experiment(self, experiment_name, status='running'):
        df = self.get_df()
        df = df.append({'experiment_name': experiment_name, 'status':status}, ignore_index=True)
        big_id = df.index[-1]
        self.save_df(df)
        return big_id


class ExperimentDB(Database):

    def __init__(self):
        super(ExperimentDB, self).__init__()
        self.db_name = 'experiments'

    def add_experiment(self, experiment_name, colour_model_config, status='running'):
        df = self.get_df()
        df = df.append({'config_name': experiment_name, 'neural_config': colour_model_config, 'stauts':status},
                       ignore_index=True)
        experiment_id = df.index[-1]
        self.save_df(df)
        return experiment_id


class JoinDB(Database):

    def __init__(self):
        super(JoinDB, self).__init__()
        self.db_name = 'rels'

    def add_experiment(self, big_id, experiment_id):
        df = self.get_df()
        df = df.append({'big_id':big_id, 'experiment_id':experiment_id}, ignore_index=True)
        rel_id = df.index[-1]
        self.save_df(df)
        return rel_id
