import sys
import subprocess
import os

# ─────────────────────────────────────────────────────────────────────────────
#  AUTO-INSTALL
# ─────────────────────────────────────────────────────────────────────────────
_PACKAGES = [
    ("ultralytics",  "ultralytics"),
    ("opencv-python","cv2"),
    ("torch",        "torch"),
    ("torchvision",  "torchvision"),
    ("numpy",        "numpy"),
    ("pillow",       "PIL"),
    ("scipy",        "scipy"),
    ("filterpy",     "filterpy"),
    ("tqdm",         "tqdm"),
]

print("\n" + "=" * 66)
print("  SMART TRAFFIC & VEHICLE ANALYTICS  |  Dependency Check")
print("=" * 66)
for pkg, imp in _PACKAGES:
    try:
        __import__(imp)
        print(f"  [OK]       {pkg}")
    except ImportError:
        print(f"  [INSTALL]  {pkg} ...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pkg, "-q",
             "--break-system-packages"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print(f"  [DONE]     {pkg}")
print("=" * 66 + "\n")

import time
import datetime
import math
import warnings
import collections
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
from filterpy.kalman import KalmanFilter
from tqdm import tqdm

os.makedirs("screenshots", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
FONT        = cv2.FONT_HERSHEY_DUPLEX
FONT_MONO   = cv2.FONT_HERSHEY_SIMPLEX
OUTPUT_PATH = "output_traffic_ai.mp4"
MAX_W       = 1280

VEHICLE_CLS = {"car","truck","bus","motorcycle","bicycle"}
PERSON_CLS  = {"person"}
TRACKED_CLS = VEHICLE_CLS | PERSON_CLS

# Speed estimation: pixels-per-frame × scale factor
SPEED_SCALE = 0.12   # tunable km/h per px/frame

# Congestion thresholds (vehicles in ROI)
CONGESTION_LOW    = 4
CONGESTION_MED    = 8
CONGESTION_HIGH   = 14

# Counting lines (y-position as fraction of frame height)
COUNT_LINE_Y = 0.55

# Lane boundaries (x as fraction of frame width)
LANE_BOUNDS = [0.0, 0.25, 0.50, 0.75, 1.0]

# Colors (BGR)
C_NEON   = (  0, 255, 180)
C_BLUE   = (220, 120,  30)
C_GREEN  = ( 50, 220,  50)
C_RED    = ( 30,  30, 220)
C_ORANGE = ( 20, 140, 255)
C_YELLOW = (  0, 220, 255)
C_CYAN   = (220, 210,  30)
C_WHITE  = (240, 240, 240)
C_GRAY   = (140, 140, 140)
C_DARK   = ( 12,  12,  12)
C_PURPLE = (200,  60, 180)
C_TEAL   = (200, 200,  30)
C_PINK   = (180,  80, 220)

VEH_COLORS = {
    "car":         (  20, 190, 255),
    "truck":       (  10, 150, 230),
    "bus":         (  30, 160, 220),
    "motorcycle":  (  50, 220, 255),
    "bicycle":     ( 100, 220, 255),
    "person":      ( 200,  80, 220),
}

def vcol(name): return VEH_COLORS.get(name, C_GRAY)

# ─────────────────────────────────────────────────────────────────────────────
#  DEVICE
# ─────────────────────────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        return "cuda", torch.cuda.get_device_name(0)
    return "cpu", "CPU"

# ─────────────────────────────────────────────────────────────────────────────
#  KALMAN VEHICLE TRACKER
# ─────────────────────────────────────────────────────────────────────────────
def _iou(a, b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b
    ix1=max(ax1,bx1); iy1=max(ay1,by1)
    ix2=min(ax2,bx2); iy2=min(ay2,by2)
    iw=max(0,ix2-ix1); ih=max(0,iy2-iy1)
    inter=iw*ih
    union=(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return inter/max(union,1e-6)


class VehicleTrack:
    _ctr = 0
    def __init__(self, bbox, cls_name):
        VehicleTrack._ctr += 1
        self.tid       = VehicleTrack._ctr
        self.cls_name  = cls_name
        self.hits      = 1; self.miss=0; self.age=0
        self.pos_hist  = collections.deque(maxlen=40)
        self.speed_kmh = 0.0
        self.counted   = False   # crossed count line
        self.lane      = 0

        kf=KalmanFilter(dim_x=7,dim_z=4)
        kf.F=np.eye(7); kf.F[0,4]=kf.F[1,5]=kf.F[2,6]=1
        kf.H=np.zeros((4,7))
        kf.H[0,0]=kf.H[1,1]=kf.H[2,2]=kf.H[3,3]=1
        kf.R[2:,2:]*=10; kf.P[4:,4:]*=1000
        kf.P*=10; kf.Q[-1,-1]*=0.01; kf.Q[4:,4:]*=0.01
        cx,cy,w,h=self._st(bbox)
        kf.x[:4]=np.array([[cx],[cy],[w],[h]])
        self.kf=kf

    @staticmethod
    def _st(b): x1,y1,x2,y2=b; return (x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1

    def predict(self): self.kf.predict(); self.age+=1

    def update(self, bbox, cls_name, fps):
        cx,cy,w,h=self._st(bbox)
        self.kf.update(np.array([[cx],[cy],[w],[h]]))
        self.cls_name=cls_name; self.hits+=1; self.miss=0
        self.pos_hist.append((cx,cy))
        if len(self.pos_hist)>=4:
            pts=list(self.pos_hist)
            dx=pts[-1][0]-pts[-4][0]; dy=pts[-1][1]-pts[-4][1]
            dist_px=math.sqrt(dx*dx+dy*dy)
            self.speed_kmh=dist_px*SPEED_SCALE*fps/4.0

    def get_box(self):
        cx,cy,w,h=self.kf.x[:4,0]; return [cx-w/2,cy-h/2,cx+w/2,cy+h/2]

    def centre(self):
        s=self.get_box(); return ((s[0]+s[2])/2,(s[1]+s[3])/2)

    def get_lane(self, frame_w):
        cx=self.centre()[0]/frame_w
        for i in range(len(LANE_BOUNDS)-1):
            if LANE_BOUNDS[i]<=cx<LANE_BOUNDS[i+1]:
                return i
        return len(LANE_BOUNDS)-2


class TrafficTracker:
    def __init__(self): self.tracks=[]

    def update(self, dets, fps):
        for t in self.tracks: t.predict()
        if not self.tracks:
            for d in dets: self.tracks.append(VehicleTrack(d[0],d[1]))
        else:
            tr_b=np.array([t.get_box() for t in self.tracks])
            dt_b=np.array([d[0] for d in dets]) if dets else np.empty((0,4))
            if len(dt_b)>0:
                cost=np.zeros((len(self.tracks),len(dt_b)))
                for i,tb in enumerate(tr_b):
                    for j,db in enumerate(dt_b):
                        cost[i,j]=1.0-_iou(tb,db)
                ri,ci=linear_sum_assignment(cost)
                mt=set(); md=set()
                for r,c in zip(ri,ci):
                    if cost[r,c]<0.75:
                        self.tracks[r].update(dets[c][0],dets[c][1],fps)
                        mt.add(r); md.add(c)
                for j,d in enumerate(dets):
                    if j not in md:
                        self.tracks.append(VehicleTrack(d[0],d[1]))
                for i in range(len(self.tracks)):
                    if i not in mt: self.tracks[i].miss+=1
            else:
                for t in self.tracks: t.miss+=1
        self.tracks=[t for t in self.tracks if t.miss<=10]
        return [t for t in self.tracks if t.hits>=2]

# ─────────────────────────────────────────────────────────────────────────────
#  VEHICLE COUNTER
# ─────────────────────────────────────────────────────────────────────────────
class VehicleCounter:
    def __init__(self):
        self.total   = 0
        self.by_cls  = collections.defaultdict(int)
        self.counted = set()

    def update(self, tracks, line_y, frame_h):
        ly_px = int(line_y * frame_h)
        for t in tracks:
            if t.tid in self.counted: continue
            cx,cy=t.centre()
            if abs(cy-ly_px)<12:
                self.total+=1
                self.by_cls[t.cls_name]+=1
                self.counted.add(t.tid)

# ─────────────────────────────────────────────────────────────────────────────
#  CONGESTION SCORER
# ─────────────────────────────────────────────────────────────────────────────
class CongestionScorer:
    def __init__(self):
        self.history=collections.deque(maxlen=60)

    def update(self, n_vehicles):
        self.history.append(n_vehicles)
        avg=np.mean(self.history)
        score=min(100, int(avg/CONGESTION_HIGH*100))
        if avg>=CONGESTION_HIGH:  level="HEAVY"
        elif avg>=CONGESTION_MED: level="MODERATE"
        elif avg>=CONGESTION_LOW: level="LIGHT"
        else:                     level="FREE FLOW"
        return score, level

# ─────────────────────────────────────────────────────────────────────────────
#  TRAFFIC HEATMAP
# ─────────────────────────────────────────────────────────────────────────────
class TrafficHeatmap:
    def __init__(self,H,W):
        self.map=np.zeros((H,W),np.float32)

    def update(self,tracks,H,W,decay=0.97):
        self.map*=decay
        for t in tracks:
            cx,cy=t.centre()
            cx=int(np.clip(cx,0,W-1)); cy=int(np.clip(cy,0,H-1))
            rr=22
            y1=max(0,cy-rr);y2=min(H,cy+rr)
            x1=max(0,cx-rr);x2=min(W,cx+rr)
            self.map[y1:y2,x1:x2]+=0.3
        self.map=np.clip(gaussian_filter(self.map,sigma=14),0,1)

    def render(self,frame,alpha=0.30):
        h_u8=(self.map*255).astype(np.uint8)
        h_col=cv2.applyColorMap(h_u8,cv2.COLORMAP_TURBO)
        mask=self.map>0.06
        out=frame.copy().astype(np.float32)
        out[mask]=(out[mask]*(1-alpha)+h_col.astype(np.float32)[mask]*alpha)
        return out.astype(np.uint8)

# ─────────────────────────────────────────────────────────────────────────────
#  LANE ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_lane_stats(tracks, frame_w):
    n_lanes=len(LANE_BOUNDS)-1
    lane_counts=[0]*n_lanes
    lane_speeds=[[] for _ in range(n_lanes)]
    for t in tracks:
        if t.cls_name not in VEHICLE_CLS: continue
        l=t.get_lane(frame_w)
        lane_counts[l]+=1
        lane_speeds[l].append(t.speed_kmh)
    avg_speeds=[np.mean(s) if s else 0.0 for s in lane_speeds]
    return lane_counts, avg_speeds

# ─────────────────────────────────────────────────────────────────────────────
#  BEV MINI-MAP
# ─────────────────────────────────────────────────────────────────────────────
class BEVMap:
    def __init__(self, w=240, h=280):
        self.W=w; self.H=h

    def render(self, tracks, frame_w, frame_h, congestion_score):
        W,H=self.W,self.H
        canvas=np.zeros((H,W,3),np.uint8)
        # Road surface
        cv2.rectangle(canvas,(W//4,0),(W*3//4,H),(20,20,20),-1)
        # Lane lines
        n_lanes=len(LANE_BOUNDS)-1
        for b in LANE_BOUNDS[1:-1]:
            lx=int(b*W)
            for y in range(0,H,16):
                cv2.line(canvas,(lx,y),(lx,min(y+10,H)),(60,60,60),1)
        # Grid
        for d in range(0,H,40):
            cv2.line(canvas,(0,d),(W,d),(25,30,25),1)
        # Vehicles
        for t in tracks:
            if t.cls_name not in VEHICLE_CLS: continue
            cx,cy=t.centre()
            bx=int(cx/frame_w*W); by=int(cy/frame_h*H)
            bx=np.clip(bx,2,W-2); by=np.clip(by,2,H-2)
            col=vcol(t.cls_name)
            cv2.rectangle(canvas,(bx-5,by-8),(bx+5,by+8),col,-1)
            cv2.rectangle(canvas,(bx-5,by-8),(bx+5,by+8),(255,255,255),1)
            _t2(canvas,str(t.tid),(bx+6,by+4),0.26,col)
        # Congestion bar
        bar_w=int(congestion_score/100*(W-20))
        col=(C_RED if congestion_score>70 else
             C_ORANGE if congestion_score>40 else C_GREEN)
        cv2.rectangle(canvas,(10,H-14),(10+bar_w,H-8),col,-1)
        cv2.rectangle(canvas,(10,H-14),(W-10,H-8),(50,50,50),1)
        _t2(canvas,"BEV MAP",(4,12),0.32,(0,200,200))
        cv2.rectangle(canvas,(0,0),(W-1,H-1),(0,160,160),1)
        return canvas


def _t2(img,text,pos,scale,color,font=FONT_MONO):
    x,y=int(pos[0]),int(pos[1])
    cv2.putText(img,text,(x,y),font,scale,color,1,cv2.LINE_AA)

# ─────────────────────────────────────────────────────────────────────────────
#  HUD DRAWING
# ─────────────────────────────────────────────────────────────────────────────
def _t(img,text,pos,scale,color,thickness=1,font=FONT_MONO):
    x,y=int(pos[0]),int(pos[1])
    cv2.putText(img,text,(x+1,y+1),font,scale,(0,0,0),thickness+1,cv2.LINE_AA)
    cv2.putText(img,text,(x,y),    font,scale,color,   thickness,  cv2.LINE_AA)


def draw_vehicle_box(frame, track):
    box=track.get_box(); x1,y1,x2,y2=[int(v) for v in box]
    col=vcol(track.cls_name)
    w=x2-x1; h=y2-y1
    # Corner box
    L=min(16,w//3,h//3)
    for (cx2,cy2),(dx,dy) in zip(
        [(x1,y1),(x2,y1),(x2,y2),(x1,y2)],
        [(1,1),(-1,1),(-1,-1),(1,-1)]
    ):
        cv2.line(frame,(cx2,cy2),(cx2+dx*L,cy2),col,2)
        cv2.line(frame,(cx2,cy2),(cx2,cy2+dy*L),col,2)

    # Info plate
    spd=f"{track.speed_kmh:.0f}km/h"
    lbl=f"#{track.tid} {track.cls_name}"
    pw=max(w,130); ph=38
    px=x1; py=max(y1-ph-4,4)
    ov=frame.copy()
    cv2.rectangle(ov,(px,py),(px+pw,py+ph),(10,10,10),-1)
    cv2.addWeighted(ov,0.72,frame,0.28,0,frame)
    cv2.rectangle(frame,(px,py),(px+pw,py+ph),col,1)
    _t(frame,lbl,(px+4,py+14),0.36,col)
    _t(frame,spd,(px+4,py+28),0.34,
       C_RED if track.speed_kmh>80 else C_WHITE)

    # Speed trail
    if len(track.pos_hist)>2:
        pts=list(track.pos_hist)
        for i in range(1,len(pts)):
            a=i/len(pts)
            tc=tuple(int(c*a) for c in col)
            cv2.line(frame,(int(pts[i-1][0]),int(pts[i-1][1])),
                     (int(pts[i][0]),int(pts[i][1])),tc,1)


def draw_count_line(frame, line_y, total_count):
    H,W=frame.shape[:2]
    ly=int(line_y*H)
    # Dashed counting line
    for x in range(0,W,20):
        cv2.line(frame,(x,ly),(min(x+12,W),ly),(0,220,220),2)
    _t(frame,f"COUNT LINE  [{total_count} vehicles passed]",
       (14,ly-8),0.42,(0,220,220))


def draw_lane_panel(frame, lane_counts, lane_speeds, frame_w):
    H,W=frame.shape[:2]
    n=len(lane_counts)
    pw=220; ph=16+n*34+10
    px=14; py=62
    ov=frame.copy()
    cv2.rectangle(ov,(px,py),(px+pw,py+ph),(10,10,10),-1)
    cv2.addWeighted(ov,0.80,frame,0.20,0,frame)
    cv2.rectangle(frame,(px,py),(px+pw,py+ph),(0,200,200),1)
    _t(frame,"LANE ANALYTICS",(px+6,py+14),0.40,(0,220,220))
    for i in range(n):
        ey=py+22+i*34
        col=(C_RED if lane_counts[i]>=CONGESTION_HIGH//n else
             C_ORANGE if lane_counts[i]>=CONGESTION_MED//n else C_GREEN)
        _t(frame,f"Lane {i+1}: {lane_counts[i]} veh  {lane_speeds[i]:.0f}km/h",
           (px+6,ey+14),0.36,col)
        # Mini bar
        bar_w=max(0,min(int(lane_counts[i]/4*(pw-12)),pw-12))
        cv2.rectangle(frame,(px+6,ey+16),(px+6+bar_w,ey+22),col,-1)
        cv2.rectangle(frame,(px+6,ey+16),(px+pw-6,ey+22),(50,50,50),1)


def draw_congestion_panel(frame, score, level, n_vehicles, avg_speed):
    H,W=frame.shape[:2]
    pw=220; ph=108; px=W-pw-14; py=62
    ov=frame.copy()
    cv2.rectangle(ov,(px,py),(px+pw,py+ph),(10,10,10),-1)
    cv2.addWeighted(ov,0.80,frame,0.20,0,frame)
    col=(C_RED if level=="HEAVY" else
         C_ORANGE if level=="MODERATE" else
         C_YELLOW if level=="LIGHT" else C_GREEN)
    cv2.rectangle(frame,(px,py),(px+pw,py+ph),col,1)
    _t(frame,f"CONGESTION: {level}",(px+6,py+18),0.42,col)
    _t(frame,f"Score    : {score:3d}/100",(px+6,py+34),0.36,C_WHITE)
    _t(frame,f"Vehicles : {n_vehicles:4d}",(px+6,py+50),0.36,C_WHITE)
    _t(frame,f"Avg speed: {avg_speed:.0f} km/h",(px+6,py+66),0.36,C_WHITE)
    # Score bar
    bar_w=int(score/100*(pw-12))
    cv2.rectangle(frame,(px+6,py+74),(px+6+bar_w,py+82),col,-1)
    cv2.rectangle(frame,(px+6,py+74),(px+pw-6,py+82),(50,50,50),1)
    _t(frame,"CONGESTION SCORE",(px+6,py+ph-8),0.30,C_GRAY)


def draw_vehicle_count_panel(frame, counter):
    H,W=frame.shape[:2]
    items=[("car",counter.by_cls["car"]),
           ("truck",counter.by_cls["truck"]),
           ("bus",counter.by_cls["bus"]),
           ("motorcycle",counter.by_cls["motorcycle"]),
           ("bicycle",counter.by_cls["bicycle"]),
           ("person",counter.by_cls["person"])]
    items=[i for i in items if i[1]>0]
    if not items: items=[("--",0)]
    pw=190; ph=16+len(items)*20+28
    px=W-pw-14; py=62+116
    ov=frame.copy()
    cv2.rectangle(ov,(px,py),(px+pw,py+ph),(10,10,10),-1)
    cv2.addWeighted(ov,0.80,frame,0.20,0,frame)
    cv2.rectangle(frame,(px,py),(px+pw,py+ph),(0,200,200),1)
    _t(frame,f"TOTAL COUNTED: {counter.total}",(px+6,py+16),0.40,(0,220,220))
    for i,(cls,cnt) in enumerate(items):
        ey=py+24+i*20
        _t(frame,f"{cls:<12s}: {cnt}",(px+6,ey+14),0.34,vcol(cls))


def draw_top_hud(frame,fps,frame_idx,total,inf_ms,device_lbl,congestion_lbl,ts):
    H,W=frame.shape[:2]; bh=54
    ov=frame.copy()
    cv2.rectangle(ov,(0,0),(W,bh),C_DARK,-1)
    cv2.addWeighted(ov,0.82,frame,0.18,0,frame)
    # Cyberpunk gradient accent
    for px2 in range(W):
        t2=px2/W
        r2=int(0+t2*30); g2=int(200+t2*55); b2=int(255-t2*100)
        cv2.line(frame,(px2,bh),(px2,bh),(r2,g2,b2),2)
    _t(frame,"SMART TRAFFIC & VEHICLE ANALYTICS SYSTEM",(14,32),0.62,
       (0,230,255),1)
    r=W-480
    _t(frame,f"FPS {fps:5.1f}",(r,30),0.50,C_GREEN)
    _t(frame,f"|  {inf_ms:.0f}ms",(r+110,30),0.50,C_GRAY)
    _t(frame,f"|  {device_lbl}",(r+200,30),0.50,C_YELLOW)
    _t(frame,f"|  {congestion_lbl}",(r+360,30),0.50,
       C_RED if congestion_lbl=="HEAVY" else C_GREEN)


def draw_bottom_hud(frame,frame_idx,total,n_vehicles,total_counted,ts):
    H,W=frame.shape[:2]; bh=42; y0=H-bh
    ov=frame.copy()
    cv2.rectangle(ov,(0,y0),(W,H),C_DARK,-1)
    cv2.addWeighted(ov,0.82,frame,0.18,0,frame)
    cv2.line(frame,(0,y0),(W,y0),(0,200,200),1)
    if total>0:
        pw2=int(frame_idx/max(total,1)*(W-28))
        cv2.rectangle(frame,(14,y0+7),(14+pw2,y0+12),(0,200,200),-1)
    cv2.rectangle(frame,(14,y0+7),(W-14,y0+12),(50,50,50),1)
    _t(frame,f"FRAME {frame_idx:05d}/{total:05d}",(14,y0+30),0.40,C_GRAY)
    _t(frame,f"ACTIVE: {n_vehicles} veh",(W//2-80,y0+30),0.40,C_CYAN)
    _t(frame,f"COUNTED: {total_counted}",(W//2+60,y0+30),0.40,C_NEON)
    _t(frame,ts,(W-160,y0+30),0.40,C_GRAY)
    tags=["YOLOv8  ","DeepSORT  ","Speed Est.  ",
          "Lane Analytics  ","Congestion AI  ","Smart City"]
    tx=14
    for tag in tags:
        cv2.putText(frame,tag,(tx,y0-6),FONT_MONO,0.28,(0,160,160),1,cv2.LINE_AA)
        tw,_=cv2.getTextSize(tag,FONT_MONO,0.28,1); tx+=tw[0]+6


def embed_bev(frame, bev_img):
    H,W=frame.shape[:2]
    bh,bw=bev_img.shape[:2]
    x0=14; y0=H-44-bh-6
    ov=frame.copy()
    cv2.rectangle(ov,(x0-2,y0-2),(x0+bw+2,y0+bh+2),C_DARK,-1)
    cv2.addWeighted(ov,0.60,frame,0.40,0,frame)
    frame[y0:y0+bh,x0:x0+bw]=cv2.addWeighted(
        frame[y0:y0+bh,x0:x0+bw],0.20,bev_img,0.80,0)
    cv2.rectangle(frame,(x0-2,y0-2),(x0+bw+2,y0+bh+2),(0,160,160),1)


def vignette(frame, strength=0.28):
    H,W=frame.shape[:2]; cx,cy=W/2,H/2
    Y,X=np.ogrid[:H,:W]
    d=np.sqrt(((X-cx)/cx)**2+((Y-cy)/cy)**2)
    v=np.clip(1.0-d*strength,0.58,1.0).astype(np.float32)
    return np.clip(frame.astype(np.float32)*v[:,:,np.newaxis],0,255).astype(np.uint8)

# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("\n"+"="*66)
    print("  SMART TRAFFIC & VEHICLE ANALYTICS  |  Production v1.0")
    print("  YOLOv8 + DeepSORT + Speed + Lane Analytics + Congestion AI")
    print("="*66)

    if len(sys.argv)<2:
        print("\n  Usage:")
        print("    python main.py input.mp4")
        print("    python main.py 0   (webcam)")
        sys.exit(0)

    source=sys.argv[1]
    webcam=str(source) in ("0","1","2")
    cap=cv2.VideoCapture(int(source) if webcam else source)
    if not cap.isOpened():
        print(f"  [ERROR]  Cannot open: {source}"); sys.exit(1)

    W_raw=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H_raw=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in=cap.get(cv2.CAP_PROP_FPS) or 30.0
    total=0 if webcam else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scale=min(1.0,MAX_W/max(W_raw,1))
    W=int(W_raw*scale); H=int(H_raw*scale)

    device,device_lbl=get_device()
    print(f"  [DEVICE]    {device_lbl}")
    print(f"  [MODEL]     Loading YOLOv8n ...")
    model=YOLO("yolov8n.pt")
    if device=="cuda": model.to("cuda")
    print(f"  [MODEL]     Ready")

    writer=cv2.VideoWriter(OUTPUT_PATH,cv2.VideoWriter_fourcc(*"mp4v"),fps_in,(W,H))
    tracker    = TrafficTracker()
    counter    = VehicleCounter()
    cong_score = CongestionScorer()
    heatmap    = TrafficHeatmap(H,W)
    bev        = BEVMap()

    win="Smart Traffic & Vehicle Analytics System"
    cv2.namedWindow(win,cv2.WINDOW_NORMAL); cv2.resizeWindow(win,W,H)

    print(f"\n  [INFO]  {W}x{H}  |  {fps_in:.0f}fps  |  {total or 'live'} frames")
    print(f"  [INFO]  Output: {OUTPUT_PATH}")
    print(f"  Controls: Q=quit  P=pause  S=screenshot\n")

    bar=tqdm(total=total or None,desc="  Processing",unit="frame",dynamic_ncols=True)
    fps_smooth=0.0; t_prev=time.time()
    frame_idx=0; inf_ms=0.0; paused=False

    while True:
        if paused:
            key=cv2.waitKey(30)&0xFF
            if key in (ord("q"),27): break
            if key==ord("p"):        paused=False
            continue

        ret,raw=cap.read()
        if not ret: break
        frame=cv2.resize(raw,(W,H)) if scale<1.0 else raw.copy()
        t0=time.perf_counter()

        # YOLO
        results=model(frame,verbose=False,imgsz=640,conf=0.30,
                      **({"half":True} if device=="cuda" else {}))[0]
        inf_ms=(time.perf_counter()-t0)*1000

        dets=[]
        if results.boxes is not None:
            for box in results.boxes:
                cn=results.names.get(int(box.cls[0]),"")
                if cn not in TRACKED_CLS: continue
                b=[float(v) for v in box.xyxy[0].tolist()]
                dets.append((b,cn,float(box.conf[0])))

        # Track & count
        tracks=tracker.update([(d[0],d[1]) for d in dets],fps_in)
        counter.update(tracks,COUNT_LINE_Y,H)

        # Analytics
        n_veh=sum(1 for t in tracks if t.cls_name in VEHICLE_CLS)
        avg_spd=(np.mean([t.speed_kmh for t in tracks if t.cls_name in VEHICLE_CLS])
                 if any(t.cls_name in VEHICLE_CLS for t in tracks) else 0.0)
        cong_s, cong_lv=cong_score.update(n_veh)
        lane_counts,lane_speeds=compute_lane_stats(tracks,W)

        # Heatmap
        heatmap.update(tracks,H,W)
        frame=heatmap.render(frame,alpha=0.25)

        # BEV
        bev_img=bev.render(tracks,W,H,cong_s)

        # Draw
        draw_count_line(frame,COUNT_LINE_Y,counter.total)
        for t in tracks: draw_vehicle_box(frame,t)
        embed_bev(frame,bev_img)

        # HUD
        ts=datetime.datetime.now().strftime("%H:%M:%S")
        t_now=time.time(); dt=max(t_now-t_prev,1e-6); t_prev=t_now
        fps_smooth=0.88*fps_smooth+0.12/dt

        draw_top_hud(frame,fps_smooth,frame_idx,total,inf_ms,device_lbl,cong_lv,ts)
        draw_bottom_hud(frame,frame_idx,total,n_veh,counter.total,ts)
        draw_lane_panel(frame,lane_counts,lane_speeds,W)
        draw_congestion_panel(frame,cong_s,cong_lv,n_veh,avg_spd)
        draw_vehicle_count_panel(frame,counter)
        frame=vignette(frame)

        writer.write(frame); cv2.imshow(win,frame)
        frame_idx+=1; bar.update(1)

        key=cv2.waitKey(1)&0xFF
        if key in (ord("q"),27):
            print("\n  [QUIT]"); break
        elif key==ord("p"):
            paused=True; print("  [PAUSED]")
        elif key==ord("s"):
            sp=os.path.join("screenshots",
               f"traffic_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg")
            cv2.imwrite(sp,frame); print(f"\n  [SCREENSHOT]  {sp}")

    bar.close(); cap.release(); writer.release()
    cv2.destroyAllWindows()
    print(f"\n  [DONE]  {frame_idx} frames  ->  {OUTPUT_PATH}")
    print("="*66+"\n")

if __name__=="__main__":
    main()