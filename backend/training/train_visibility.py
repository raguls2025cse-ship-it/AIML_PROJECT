import json,cv2,joblib,numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
LABEL_FILE=Path("data/rwvc-bdd100k/train_labels.json"); IMAGE_DIR=Path("data/rwvc-bdd100k/images"); MODEL_OUT=Path("models/visibility_rf.joblib")
def feat(img):
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return [float(np.mean(g)),float(np.std(g)),float(cv2.Laplacian(g,cv2.CV_64F).var()),float(np.mean(cv2.Canny(g,80,160)>0))]
def main():
    if not LABEL_FILE.exists(): print("Set LABEL_FILE and IMAGE_DIR to RWVC-BDD100K paths."); return
    raw=json.loads(LABEL_FILE.read_text()); X=[]; y=[]
    for item in raw:
        name=item.get("image") or item.get("name") or item.get("file_name"); label=item.get("visibility")
        if not name or label not in {"good","poor"}: continue
        img=cv2.imread(str(IMAGE_DIR/name))
        if img is not None: X.append(feat(img)); y.append(label)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    clf=RandomForestClassifier(n_estimators=250,random_state=42,class_weight="balanced"); clf.fit(Xtr,ytr)
    print(classification_report(yte,clf.predict(Xte))); MODEL_OUT.parent.mkdir(parents=True,exist_ok=True); joblib.dump(clf,MODEL_OUT)
if __name__=="__main__": main()
