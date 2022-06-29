from datetime import datetime
import glob
import os
import logging
import dill
from collections import defaultdict
import pandas as pd

path = os.environ.get('PROJECT_PATH', '.')
# path = 'C:/Users/litovka_ar/airflow_hw'


def open_model():
    list_of_models = glob.glob(f'{path}/data/models/*')
    latest_model = max(list_of_models, key=os.path.getctime).replace('\\', '/')
    with open(latest_model, 'rb') as file:
        model = dill.load(file)
    logging.info(f'Opened model')
    return model


def create_test_df():
    df_list = []
    fnl = sorted(os.listdir(f'{path}/data/test'))
    for fjson in fnl:
        data = pd.read_json(f'{path}/data/test/{fjson}', typ='series')
        df_list.append(data)
    df = pd.concat(df_list, axis=1).T
    logging.info(f'Created test dataframe')
    return df


def prediction(df_test, model):
    pred_dict = defaultdict(list)
    for i in range(len(df_test)):
        drop_list = []
        for j in range(len(df_test)):
            if j != i:
                drop_list.append(j)
        df_ready = df_test.drop(drop_list, axis=0)
        pred = model.predict(df_ready)
        pred_dict['car_id'].append(df_ready.id.iloc[0])
        pred_dict['pred'].append(pred[0])
    logging.info(f'Made predictions')
    save_prediction(pred_dict)


def save_prediction(pred_dict):
    df_save = pd.DataFrame(pred_dict)
    df_save_filename = f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv'
    df_save.to_csv(df_save_filename, index=False)
    logging.info(f'Predictions are saved as preds_{datetime.now().strftime("%Y%m%d%H%M")}')


def predict():
    model = open_model()
    df_test = create_test_df()
    prediction(df_test, model)


if __name__ == '__main__':
    predict()
