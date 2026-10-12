from train import train_model

DP_CONFIGS=[
    {"name":"DP-0.5","noise_multiplier":0.5},
    {"name":"DP-1.0","noise_multiplier":1.0},
    {"name":"DP-1.5","noise_multiplier":1.5},
]

def run_all(path,label_column,dataset_name,epochs=3,batch_size=256,lr=0.01,clip=1.0,delta=1e-5):
    results=[]
    baseline=train_model(path,label_column,dataset_name,mode="baseline",epochs=epochs,batch_size=batch_size,lr=lr)
    baseline["experiment_name"]="Baseline"; results.append(baseline)
    for cfg in DP_CONFIGS:
        r=train_model(path,label_column,dataset_name,mode="dp",epochs=epochs,batch_size=batch_size,lr=lr,
                      noise_multiplier=cfg["noise_multiplier"],max_grad_norm=clip,delta=delta)
        r["experiment_name"]=cfg["name"]; results.append(r)
    return results
