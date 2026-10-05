import json,shutil
from pathlib import Path
SOURCE_JSON=Path("data/bdd100k/labels/det_train.json")
IMAGE_DIR=Path("data/bdd100k/images/100k/train")
OUT=Path("data/yolo_bdd")
CLASSES={"person":0,"rider":1,"car":2,"truck":3,"bus":4,"train":5,"motorcycle":6,"bicycle":7,"traffic light":8,"traffic sign":9}
def main():
    if not SOURCE_JSON.exists(): print("Set SOURCE_JSON to downloaded BDD100K det_train.json"); return
    data=json.loads(SOURCE_JSON.read_text())
    (OUT/"labels").mkdir(parents=True,exist_ok=True); (OUT/"images").mkdir(parents=True,exist_ok=True)
    for item in data:
        name=item["name"]; src=IMAGE_DIR/name
        if not src.exists(): continue
        w=item.get("width",1280); h=item.get("height",720); lines=[]
        for lab in item.get("labels",[]):
            cat=lab.get("category")
            if cat not in CLASSES or "box2d" not in lab: continue
            b=lab["box2d"]; x1,y1,x2,y2=b["x1"],b["y1"],b["x2"],b["y2"]
            lines.append(f"{CLASSES[cat]} {((x1+x2)/2/w):.6f} {((y1+y2)/2/h):.6f} {((x2-x1)/w):.6f} {((y2-y1)/h):.6f}")
        shutil.copy2(src,OUT/"images"/name); (OUT/"labels"/(Path(name).stem+".txt")).write_text("\n".join(lines))
if __name__=="__main__": main()
