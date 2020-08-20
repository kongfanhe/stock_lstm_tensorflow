from pymongo import MongoClient
from datetime import datetime
from downloader import request_data
import pandas as pd
from dateutil.relativedelta import relativedelta


def mongo_collection(db_name, collection_name):
    client = MongoClient('mongodb://localhost:27017/')
    db = client[db_name]
    collection = db[collection_name]
    return collection


def mongo_document():
    coll = mongo_collection("predict", "record")
    doc = coll.find_one({})
    return doc, coll


def update(codes, names, dates, close, volume, p_close, sec_names, sec_close, sec_p_close, end_date):
    stock_dict = dict(zip(codes, names))
    predicts = {
        "dates": dates, "codes": codes, "close": close.round(2).tolist(),
        "volume": volume.round().astype(int).tolist()
    }
    df = pd.DataFrame({
        "code": codes, "name": names,
        "close": close[:, 0].round(2).tolist(), "volume": volume[:, 0].round().astype(int)
    })
    df["upr"] = ((close[:, 0] - p_close) / p_close * 100).round(2)
    rankings = {
        "rise": df.sort_values(by="upr", ascending=False).iloc[:20, :].to_dict('records'),
        "fall": df.sort_values(by="upr", ascending=True).iloc[:20, :].to_dict('records'),
        "volume": df.sort_values(by="volume", ascending=False).iloc[:20, :].to_dict('records')
    }
    df = pd.DataFrame({"name": sec_names, "close": sec_close[:, 0].round(2).tolist()})
    df["up"] = (sec_close[:, 0] - sec_p_close).round(2)
    df["upr"] = ((sec_close[:, 0] - sec_p_close) / sec_p_close * 100).round(2)
    sectors = df.to_dict('records')
    record_new = {
        "stock_dict": stock_dict, "sectors": sectors, "predicts": predicts,
        "rankings": rankings, "date": end_date.strftime("%Y-%m-%d")
    }
    record_old, coll = mongo_document()
    if record_old is None:
        coll.insert_one(record_new)
        print("insert done")
    else:
        coll.replace_one(record_old, record_new)
        print("replace done")


def str2date(s):
    return datetime.strptime(s, '%Y-%m-%d')
    

def get_target_date():
    today = datetime.today()
    hq = request_data("zs_000001", today - relativedelta(days=30), today)
    if hq is None:
        return None
    else:
        target_date = str2date(hq.iloc[-1]["Date"])
        record, _ = mongo_document()
        if record is None:
            return target_date
        else:
            if target_date <= str2date(record["date"]):
                return None
            else:
                return target_date
