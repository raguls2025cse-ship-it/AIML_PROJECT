def assess_risk(detections, frame_w, frame_h, visibility):
    high = False
    caution = False
    for d in detections:
        cx = (d["x1"] + d["x2"]) / 2
        cy = (d["y1"] + d["y2"]) / 2
        area = max(0, d["x2"]-d["x1"]) * max(0, d["y2"]-d["y1"])
        area_ratio = area / max(1, frame_w*frame_h)
        central = 0.20*frame_w < cx < 0.80*frame_w
        lower = cy > 0.45*frame_h
        relevant = d["label"] in {"person","car","truck","bus","motorcycle","bicycle"}
        if relevant and central and lower and area_ratio > 0.045:
            high=True; d["risk"]="HIGH"
        elif relevant and central and area_ratio > 0.015:
            caution=True; d["risk"]="CAUTION"
        else:
            d["risk"]="LOW"
    if visibility == "POOR":
        if high: return "HIGH RISK","POOR VISIBILITY + OBJECT IN DANGER ZONE"
        return "CAUTION","POOR VISIBILITY - REDUCE SPEED / INCREASE AWARENESS"
    if high: return "HIGH RISK","OBJECT DETECTED IN VEHICLE DANGER ZONE"
    if caution: return "CAUTION","POTENTIAL HAZARD DETECTED"
    return "SAFE","NO IMMEDIATE HIGH-RISK OBJECT DETECTED"
