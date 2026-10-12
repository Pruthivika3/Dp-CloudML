import json, os, time, threading, uuid
from werkzeug.utils import secure_filename
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from train import train_model, load_dataframe
from experiment import run_all

BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR=os.path.join(BASE_DIR,"results"); RESULTS_FILE=os.path.join(RESULTS_DIR,"results.json")
FRONTEND_DIR=os.path.join(BASE_DIR,"frontend"); DATA_DIR=os.path.join(BASE_DIR,"data")
os.makedirs(RESULTS_DIR,exist_ok=True); os.makedirs(DATA_DIR,exist_ok=True)
app=Flask(__name__,static_folder=FRONTEND_DIR); app.config["MAX_CONTENT_LENGTH"]=100*1024*1024; CORS(app)
training_lock=threading.Lock()

def read_results():
    try:
        with open(RESULTS_FILE,encoding="utf-8") as f:return json.load(f)
    except Exception:return []
def write_results(results):
    with open(RESULTS_FILE,"w",encoding="utf-8") as f:json.dump(results,f,indent=2)

def dataset_path(dataset_id):
    if not dataset_id or not dataset_id.replace("-","").isalnum(): raise ValueError("Invalid dataset id")
    matches=[x for x in os.listdir(DATA_DIR) if x.startswith(dataset_id+".")]
    if not matches: raise ValueError("Dataset not found; upload it again")
    return os.path.join(DATA_DIR,matches[0])

@app.route("/")
def index():return send_from_directory(FRONTEND_DIR,"index.html")
@app.route("/api/health")
def health():return jsonify({"status":"healthy","service":"DP-CloudML"})
@app.route("/api/config")
def config():return jsonify({"dataset":"Dashboard CSV/XLSX upload","model":"Adaptive Tabular MLP","dp_method":"DP-SGD","framework":"Opacus","default_delta":1e-5,"default_clip_norm":1.0,"default_epochs":3})
@app.route("/api/results")
def results():return jsonify(read_results())
@app.route("/api/metrics")
def metrics():
    import psutil
    p=psutil.Process(os.getpid()); return jsonify({"cpu_percent":p.cpu_percent(interval=.05),"memory_mb":round(p.memory_info().rss/(1024*1024),2),"timestamp":time.time()})
@app.route("/api/reset",methods=["POST"])
def reset():write_results([]);return jsonify({"status":"reset","results":[]})

@app.route("/api/upload",methods=["POST"])
def upload():
    f=request.files.get("file")
    if not f or not f.filename:return jsonify({"error":"Choose a CSV or Excel file"}),400
    ext=os.path.splitext(f.filename)[1].lower()
    if ext not in {".csv",".xlsx",".xls"}:return jsonify({"error":"Only CSV, XLSX and XLS files are supported"}),400
    dataset_id=uuid.uuid4().hex
    safe=secure_filename(f.filename) or ("dataset"+ext)
    path=os.path.join(DATA_DIR,dataset_id+ext)
    f.save(path)
    try:
        df=load_dataframe(path)
        preview=json.loads(df.head(8).to_json(orient="records",date_format="iso"))
        columns=[{"name":str(c),"dtype":str(df[c].dtype),"unique":int(df[c].nunique(dropna=True))} for c in df.columns]
        return jsonify({"dataset_id":dataset_id,"filename":safe,"rows":len(df),"columns":columns,"preview":preview})
    except Exception as exc:
        try:os.remove(path)
        except OSError:pass
        return jsonify({"error":str(exc)}),400

@app.route("/api/train",methods=["POST"])
def train():
    if not training_lock.acquire(blocking=False):return jsonify({"error":"Another training job is already running"}),409
    try:
        payload=request.get_json(silent=True) or {}; path=dataset_path(payload.get("dataset_id")); label=payload.get("label_column")
        if not label:return jsonify({"error":"Select a label/target column before training"}),400
        result=train_model(path,label,payload.get("filename","Uploaded dataset"),mode=payload.get("mode","baseline"),
                           epochs=int(payload.get("epochs",3)),batch_size=int(payload.get("batch_size",256)),lr=float(payload.get("learning_rate",0.01)),
                           noise_multiplier=float(payload.get("noise_multiplier",1.0)),max_grad_norm=float(payload.get("max_grad_norm",1.0)),delta=float(payload.get("delta",1e-5)))
        current=read_results();current.append(result);write_results(current);return jsonify(result)
    except Exception as exc:return jsonify({"error":str(exc)}),400
    finally:training_lock.release()

@app.route("/api/run-experiment",methods=["POST"])
def experiment():
    if not training_lock.acquire(blocking=False):return jsonify({"error":"Another training job is already running"}),409
    try:
        payload=request.get_json(silent=True) or {}; path=dataset_path(payload.get("dataset_id")); label=payload.get("label_column")
        if not label:return jsonify({"error":"Select a label/target column before running the experiment"}),400
        results=run_all(path,label,payload.get("filename","Uploaded dataset"),epochs=int(payload.get("epochs",3)),batch_size=int(payload.get("batch_size",256)),lr=float(payload.get("learning_rate",0.01)),clip=float(payload.get("max_grad_norm",1.0)),delta=float(payload.get("delta",1e-5)))
        write_results(results);return jsonify({"status":"completed","results":results})
    except Exception as exc:return jsonify({"error":str(exc)}),400
    finally:training_lock.release()

if __name__=="__main__":app.run(host="0.0.0.0",port=int(os.environ.get("PORT",5000)),debug=False)
