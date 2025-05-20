import json
import pickle
import datetime
import pandas as pd
from flask import Flask, jsonify, request
from peewee import (
    SqliteDatabase, Model, DateField, FloatField,
    CharField, IntegrityError  # <-- sku é string, usar CharField
)
from playhouse.shortcuts import model_to_dict
from sklearn.base import BaseEstimator, TransformerMixin
import os
from playhouse.db_url import connect
import joblib    
import logging
from functools import wraps
from threading import Lock
from collections import OrderedDict

app = Flask(__name__)  # <-- Faltava criar a instância do Flask

class MeanTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.sku_means = {}
        self.structure_means = {}

    def fit(self, X, y):
        df = X.copy()
        df['final_price_chain'] = y
        self.sku_means = df.groupby('sku')['final_price_chain'].mean().to_dict()
        self.structure_means = df.groupby('structure_level_3')['final_price_chain'].mean().to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        X['mean_price_by_sku'] = X['sku'].map(self.sku_means)
        X['mean_price_by_structure3'] = X['structure_level_3'].map(self.structure_means)
        return X

DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

class PricePrediction(Model):
    sku = CharField()  # SKU é string
    time_key = DateField()
    pvp_is_competitorA = FloatField()
    pvp_is_competitorB = FloatField()
    pvp_is_competitorA_actual = FloatField(null=True)
    pvp_is_competitorB_actual = FloatField(null=True)

    class Meta:
        database = DB
        indexes = ((('sku', 'time_key'), True),)

DB.connect()
DB.create_tables([PricePrediction], safe=True)

# === Configura Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger('werkzeug').setLevel(logging.INFO)

db_lock = Lock()


forecast_requests_count = 0
actual_requests_count = 0



# === Load Models and Data ===
with open('columns_novas.json') as f:
    columns = json.load(f)

with open('pipeline_model_A.joblib', 'rb') as f:
    pipeline_A = joblib.load(f)

with open('pipeline_model_B.joblib', 'rb') as f:
    pipeline_B = joblib.load(f)

with open('pipeline_historico.joblib', 'rb') as f:
    pipeline_historico = joblib.load(f)

with open('sku_structure_map.pkl', 'rb') as f:
    mapa_sku = pickle.load(f)
    mapa_sku = mapa_sku.set_index('sku')[['structure_level_2', 'structure_level_3']].to_dict(orient='index')

# === Helper Functions ===
def validate_positive_price(price):
    return isinstance(price, (int, float)) and price >= 0

def validate_date_format(date_str):
    try:
        datetime.datetime.strptime(str(date_str), '%d%m%Y')
        return True
    except ValueError:
        return False

def date_not_before_oct_2024(date_str):
    try:
        date_obj = datetime.datetime.strptime(str(date_str), '%d%m%Y')
        cutoff = datetime.datetime(year=2024, month=10, day=1)
        return date_obj >= cutoff
    except Exception:
        return False

def validate_json_forecast(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        data = request.get_json()
        if not data:
            logger.warning("JSON body não enviado na request.")
            return jsonify({'error': 'JSON body required.'}), 422

        items = data if isinstance(data, list) else [data]

        for item in items:
            # Checa campos obrigatórios antes de tipo para evitar KeyError
            if 'sku' not in item or 'time_key' not in item:
                logger.warning(f"Faltando sku ou time_key: {item}")
                return jsonify({'error': 'sku and time_key are mandatory fields.'}), 422

            if not isinstance(item['sku'], str):
                logger.warning(f"Tipo inválido para sku: deve ser string")
                return jsonify({'error': 'Invalid type: sku must be string'}), 422

            if not isinstance(item['time_key'], int):
                logger.warning(f"Tipo inválido para time_key: deve ser integer")
                return jsonify({'error': 'Invalid type: time_key must be integer (ddmmyyyy format without quotes)'}), 422

            time_key_str = str(item['time_key'])
            if len(time_key_str) != 8 or not time_key_str.isdigit():
                logger.warning(f"Formato inválido para time_key: {item['time_key']}")
                return jsonify({'error': 'time_key must be 8 digits (ddmmyyyy format as integer)'}), 422
            
            if not validate_date_format(time_key_str):
                logger.warning(f"Formato inválido de time_key: {time_key_str}")
                return jsonify({'error': 'Invalid time_key format. Expected ddmmyyyy.'}), 422

            if not date_not_before_oct_2024(time_key_str):
                logger.warning(f"time_key antes de 01/10/2024: {item['time_key']}")
                return jsonify({'error': 'time_key must be on or after 01/10/2024'}), 422

        return f(*args, **kwargs)
    return wrapper

def gerar_features(sku, time_key):
    date = datetime.datetime.strptime(str(time_key), '%d%m%Y')
    day_of_week, month, year = date.weekday(), date.month, date.year
    week_of_year = date.isocalendar()[1]
    is_weekend = int(day_of_week >= 5)
    print(f"Received sku: {sku}, type: {type(sku)}")

    try:
        sku_key = int(sku)  # Converte sku para inteiro, pois as chaves no mapa são ints
        print(f"Converted sku_key: {sku_key}, type: {type(sku_key)}")
    except Exception as e:
        raise ValueError(f"Failed to convert SKU '{sku}' to int: {e}")

    print(f"Mapa SKU keys example type: {type(next(iter(mapa_sku.keys())))}")
    print(f"Mapa SKU keys sample: {list(mapa_sku.keys())[:10]}")

    if sku_key not in mapa_sku:
        raise ValueError(f"SKU '{sku_key}' not found in structure map.")

    struct = mapa_sku[sku_key]
    print(f"Found structure: {struct}")

    preco_previsto = pipeline_historico.predict(pd.DataFrame([{
        'sku': sku,
        'structure_level_2': struct['structure_level_2'],
        'structure_level_3': struct['structure_level_3'],
        'day_of_week': day_of_week,
        'month': month,
        'is_weekend': is_weekend,
        'year': year,
        'week_of_year': week_of_year
    }]))[0]

    return pd.DataFrame([{
        'sku': sku,
        'structure_level_2': struct['structure_level_2'],
        'structure_level_3': struct['structure_level_3'],
        'final_price_chain': preco_previsto,
        'day_of_week': day_of_week,
        'month': month,
        'is_weekend': is_weekend,
        'year': year,
        'week_of_year': week_of_year
    }])

forecast_requests_count = 0
actual_requests_count = 0

@app.route('/forecast_prices/', methods=['POST'])
@validate_json_forecast
def forecast_prices():
    global forecast_requests_count
    forecast_requests_count += 1  # LOG: contar requests
    
    payload = request.get_json()
    sku = payload["sku"]
    time_key = payload["time_key"]
    time_key_dt = datetime.datetime.strptime(str(time_key), '%d%m%Y').date()
    
    logger.info(f"[forecast_prices] Request #{forecast_requests_count} - SKU: {sku}, time_key: {time_key}")

    with db_lock:
        try:
            PricePrediction.get(
                (PricePrediction.sku == sku) &
                (PricePrediction.time_key == time_key_dt)
            )
            return jsonify({"error": "Forecast already exists for this sku and time_key"}), 422
        except PricePrediction.DoesNotExist:
            pass

        try:
            obs_df = gerar_features(sku, time_key)
            price_A = float(pipeline_A.predict(obs_df)[0])
            price_B = float(pipeline_B.predict(obs_df)[0])
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

        try:
            PricePrediction.create(
                sku=sku,
                time_key=time_key_dt,
                pvp_is_competitorA=price_A,
                pvp_is_competitorB=price_B,
            )
        except IntegrityError:
            return jsonify({"error": "Forecast already exists for this sku and time_key"}), 422

    logger.info(f"[forecast_prices] Predicted prices - competitorA: {price_A}, competitorB: {price_B}")

    return jsonify({
        "sku": sku,
        "time_key": time_key,
        "pvp_is_competitorA": price_A,
        "pvp_is_competitorB": price_B
    })

@app.route("/actual_prices/", methods=["POST"])
@validate_json_forecast
def actual_prices():
    global actual_requests_count
    actual_requests_count += 1  # LOG: contar requests

    payload = request.get_json()
    sku = payload["sku"]
    time_key = payload["time_key"]
    time_key_dt = datetime.datetime.strptime(str(time_key), '%d%m%Y').date()

    # Log info dos preços reais recebidos
    pvp_compA_actual = payload.get("pvp_is_competitorA_actual")
    pvp_compB_actual = payload.get("pvp_is_competitorB_actual")
    logger.info(f"[actual_prices] Request #{actual_requests_count} - SKU: {sku}, time_key: {time_key}, "
                f"pvp_is_competitorA_actual: {pvp_compA_actual}, pvp_is_competitorB_actual: {pvp_compB_actual}")

    # Validação campos atual prices no payload
    for key in ["pvp_is_competitorA_actual", "pvp_is_competitorB_actual"]:
        if key not in payload:
            logger.warning(f"Missing field: {key}")
            return jsonify({"error": f"{key} is mandatory."}), 422
        if not isinstance(payload[key], (float, int)) or payload[key] < 0:
            logger.warning(f"Invalid value for {key}: {payload[key]}")
            return jsonify({"error": f"{key} must be a non-negative number."}), 422

    with db_lock:
        try:
            record = PricePrediction.get(
                (PricePrediction.sku == sku) & (PricePrediction.time_key == time_key_dt)
            )
        except PricePrediction.DoesNotExist:
            return jsonify({"error": "No forecast exists for this sku and time_key"}), 422

        record.pvp_is_competitorA_actual = float(payload["pvp_is_competitorA_actual"])
        record.pvp_is_competitorB_actual = float(payload["pvp_is_competitorB_actual"])
        record.save()

    return jsonify(OrderedDict([
        ("sku", record.sku),
        ("time_key", time_key),
        ("pvp_is_competitorA", record.pvp_is_competitorA),
        ("pvp_is_competitorB", record.pvp_is_competitorB),
        ("pvp_is_competitorA_actual", record.pvp_is_competitorA_actual),
        ("pvp_is_competitorB_actual", record.pvp_is_competitorB_actual),
    ]))
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
