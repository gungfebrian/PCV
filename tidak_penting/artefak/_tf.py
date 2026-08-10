import json, os, sys, numpy as np, cv2, torch, protokol as P
model,_=P.muat_model(); torch.set_num_threads(4)
kat=P.baca_katalog(); gal,qry=P.bangun_split(kat)
qi=np.arange(0,len(qry),6)[:200]; gi=np.arange(0,len(gal),5)[:400]
def lama(rgb):
    rgb=cv2.resize(rgb,(224,224),interpolation=cv2.INTER_AREA)
    return ((rgb.astype(np.float32)/255.0-P.MEAN)/P.STD).transpose(2,0,1)
TF={"kanonik":P.transform_kanonik,"lama":lama}
def run(tf,paths,out):
    E=np.load(out) if os.path.exists(out) else np.zeros((len(paths),768),np.float32)
    pr=out+".p"; d=int(open(pr).read()) if os.path.exists(pr) else 0
    import time; t0=time.time()
    while d<len(paths) and time.time()-t0<30:
        xs=[tf(cv2.cvtColor(cv2.imread(p),cv2.COLOR_BGR2RGB)) for p in paths[d:d+16]]
        with torch.no_grad(): E[d:d+len(xs)]=model(torch.from_numpy(np.stack(xs))).float().numpy()
        d+=len(xs); np.save(out,E); open(pr,"w").write(str(d))
    return d==len(paths)
os.makedirs("hasil/tf",exist_ok=True)
sel=[("q",[qry[i]["path"] for i in qi]),("g",[gal[i]["path"] for i in gi])]
for nm in TF:
    for tag,paths in sel:
        if not run(TF[nm],paths,f"hasil/tf/{nm}_{tag}.npy"): print("lanjut",nm,tag); sys.exit(0)
res={}
id_g=np.array([r["identity"] for r in gal]); id_q=np.array([r["identity"] for r in qry])
s_g=np.array([r["side"] for r in gal]); s_q=np.array([r["side"] for r in qry])
for nm in TF:
    E=lambda t: (lambda A: A/np.maximum(np.linalg.norm(A,axis=1,keepdims=True),1e-9))(np.load(f"hasil/tf/{nm}_{t}.npy"))
    res[nm]=P.ringkas(P.evaluasi_manual(E("q"),E("g"),id_q[qi],id_g[gi],s_q[qi],s_g[gi]))
res["delta_rank1_kanonik_minus_lama"]=res["kanonik"]["rank1"]-res["lama"]["rank1"]
res["catatan"]="subset 200 query x 400 gallery; hanya SELISIH yang bermakna, bukan angka absolut"
print(json.dumps(res,indent=1)); json.dump(res,open("hasil/transform.json","w"),indent=1)
