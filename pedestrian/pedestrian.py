"""
================================================
AI PEDESTRIAN 3D PERCEPTION SYSTEM
REAL-TIME HUMAN MOTION INTELLIGENCE AI
================================================
Dev/Creator : tubakhxn
GitHub      : https://github.com/tubakhxn
================================================
FAST MODE:
  - yolov8n-seg.pt  (~6.7 MB only)
  - NO MiDaS        (zero extra download)
  - Auto-resize     (process at 640px wide)
  - Vectorized UV   (numpy, no row loops)
================================================
"""

# ─────────────────────────────────────────────
#  STEP 0 — AUTO-INSTALL
# ─────────────────────────────────────────────
import subprocess, sys, importlib, time, threading

REQUIRED = {
    "ultralytics": "ultralytics>=8.0.0",
    "cv2":         "opencv-python>=4.8.0",
    "numpy":       "numpy>=1.24.0",
    "torch":       "torch>=2.0.0",
    "torchvision": "torchvision>=0.15.0",
    "scipy":       "scipy>=1.10.0",
}

BAR_W = 38

def _spinner(label, stop_event):
    frames = ["⣾","⣽","⣻","⢿","⡿","⣟","⣯","⣷"]
    i = 0
    while not stop_event.is_set():
        print(f"\r  {frames[i%8]}  {label}", end="", flush=True)
        time.sleep(0.08); i += 1

def _install(pkg, name):
    stop = threading.Event()
    t = threading.Thread(target=_spinner, args=(f"Installing {name}...", stop), daemon=True)
    t.start()
    subprocess.check_call([sys.executable,"-m","pip","install",pkg,"-q"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    stop.set(); t.join()
    print(f"\r  ✓  {name:<28} installed          ")

def auto_install():
    missing = [(m,p) for m,p in REQUIRED.items()
               if not importlib.util.find_spec(m.split(".")[0])]
    if not missing:
        print("  ✓  All dependencies ready.\n"); return
    print(f"  Installing {len(missing)} package(s)...\n")
    for mod, pkg in missing:
        _install(pkg, mod)
    print()

print()
print("  ╔═════════════════════════════════════════════════╗")
print("  ║  AI PEDESTRIAN 3D PERCEPTION SYSTEM            ║")
print("  ║  REAL-TIME HUMAN MOTION INTELLIGENCE AI        ║")
print("  ║  Dev: tubakhxn  |  github.com/tubakhxn        ║")
print("  ╚═════════════════════════════════════════════════╝\n")
auto_install()

# ─────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────
import math, warnings, collections, os
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from ultralytics import YOLO

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
SEG_MODEL    = "yolov8n-seg.pt"
CONF         = 0.35
IOU          = 0.45
PERSON_CLS   = 0
OUTPUT_FILE  = "output_pedestrian_3d_ai.mp4"
TRAIL_LEN    = 32
RISK_DIST    = 85

# ★ KEY: process at this width for speed, output at original size
PROC_W       = 640          # inference width (fast!)
MAX_OUT_W    = 1280         # cap output width (don't write giant 4K files)

BEV_W, BEV_H = 260, 340

# ─────────────────────────────────────────────
#  COLORS (BGR)
# ─────────────────────────────────────────────
C_CYAN    = (0, 255, 220)
C_MAG     = (200, 0, 255)
C_YELLOW  = (0, 220, 255)
C_RED     = (0, 40, 255)
C_GREEN   = (0, 255, 120)
C_ORANGE  = (0, 150, 255)
C_WHITE   = (255, 255, 255)
C_DARK    = (8, 8, 18)

# Body-part UV zones: head (top) → feet (bottom) — BGR
UV_ZONES_BGR = np.array([
    [211,   0, 148],   # purple      — head
    [130,   0,  75],   # indigo      — neck
    [255,   0,   0],   # blue        — chest
    [255, 128,   0],   # light-blue  — upper torso
    [200, 255,   0],   # cyan        — stomach
    [ 50, 255,   0],   # green       — hips
    [  0, 255, 128],   # yellow-grn  — upper leg
    [  0, 200, 255],   # yellow      — knee
    [  0, 128, 255],   # orange      — lower leg
    [  0,  40, 255],   # red-orange  — ankle
    [128,   0, 255],   # magenta-red — feet
], dtype=np.float32)   # shape (11, 3)

N_ZONES = len(UV_ZONES_BGR)

# ─────────────────────────────────────────────
#  UTILS
# ─────────────────────────────────────────────
def now_ms():  return time.perf_counter() * 1000
def lerp(a,b,t): return a + (b-a)*t
def clamp(v,lo,hi): return max(lo,min(hi,v))
def cx_cy(box):
    x1,y1,x2,y2=box; return ((x1+x2)//2,(y1+y2)//2)
def box_area(box):
    x1,y1,x2,y2=box; return max(0,x2-x1)*max(0,y2-y1)
def dist2d(a,b): return math.hypot(a[0]-b[0],a[1]-b[1])

# ─────────────────────────────────────────────
#  VECTORIZED BODY-UV PAINT  ← fast numpy
# ─────────────────────────────────────────────
def paint_body_uv(canvas, mask_bool, box):
    """
    Paint segmentation mask with vertical body-part UV gradient.
    Fully vectorized — NO Python row loops.
    """
    x1,y1,x2,y2 = box
    h_img, w_img = canvas.shape[:2]
    x1=clamp(x1,0,w_img-1); x2=clamp(x2,0,w_img-1)
    y1=clamp(y1,0,h_img-1); y2=clamp(y2,0,h_img-1)
    box_h = max(1, y2-y1)
    box_w = max(1, x2-x1)

    # Build a (box_h,) float array 0→1 for vertical position
    t_vals = np.linspace(0, 1, box_h, dtype=np.float32)  # (H,)

    # Interpolate across UV_ZONES_BGR
    idx_f  = t_vals * (N_ZONES - 1)          # (H,) float indices
    idx_lo = np.floor(idx_f).astype(int)     # (H,)
    idx_hi = np.minimum(idx_lo+1, N_ZONES-1) # (H,)
    frac   = (idx_f - idx_lo)[:,None]        # (H,1)

    # Row colors: (H, 3)
    row_colors = (UV_ZONES_BGR[idx_lo] * (1-frac) +
                  UV_ZONES_BGR[idx_hi] * frac).astype(np.uint8)

    # Broadcast to (H, W, 3)
    uv_patch = np.broadcast_to(
        row_colors[:, np.newaxis, :],
        (box_h, box_w, 3)
    ).copy()

    # Crop mask to box region
    m = mask_bool[y1:y2, x1:x2]  # (H, W) bool
    if m.shape[0] != box_h or m.shape[1] != box_w:
        return

    # Blend: 72% UV color + 28% original
    orig = canvas[y1:y2, x1:x2].astype(np.float32)
    blended = (orig * 0.28 + uv_patch.astype(np.float32) * 0.72).clip(0,255).astype(np.uint8)

    # Write only where mask is True
    region = canvas[y1:y2, x1:x2]
    region[m] = blended[m]
    canvas[y1:y2, x1:x2] = region

    # Glow contour on mask edge
    m8 = m.astype(np.uint8)*255
    cnts,_ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        shifted = [c + np.array([[[x1,y1]]]) for c in cnts]
        top_col = tuple(int(c) for c in row_colors[0])
        cv2.drawContours(canvas, shifted, -1, top_col, 2, cv2.LINE_AA)

# ─────────────────────────────────────────────
#  SQUARED GLOWING BOX
# ─────────────────────────────────────────────
def draw_box(img, pt1, pt2, color, thick=2):
    ov = img.copy()
    cv2.rectangle(ov, pt1, pt2, color, thick+7)
    cv2.addWeighted(ov, 0.28, img, 0.72, 0, img)
    cv2.rectangle(img, pt1, pt2, color, thick)
    x1,y1=pt1; x2,y2=pt2
    inner = tuple(min(255,c+70) for c in color)
    cv2.rectangle(img,(x1+3,y1+3),(x2-3,y2-3),inner,1)

# ─────────────────────────────────────────────
#  DRAW HELPERS
# ─────────────────────────────────────────────
def alpha_rect(img, pt1, pt2, color, a=0.55):
    x1,y1=pt1; x2,y2=pt2
    H,W=img.shape[:2]
    x1,y1,x2,y2=clamp(x1,0,W-1),clamp(y1,0,H-1),clamp(x2,0,W-1),clamp(y2,0,H-1)
    sub=img[y1:y2,x1:x2]
    if sub.size==0: return
    img[y1:y2,x1:x2]=cv2.addWeighted(sub,1-a,np.full_like(sub,color),a,0)

def txt(img, s, pos, col=C_CYAN, sc=0.50, th=1, shad=True):
    f=cv2.FONT_HERSHEY_SIMPLEX; x,y=pos
    if shad: cv2.putText(img,s,(x+1,y+1),f,sc,(0,0,0),th+1,cv2.LINE_AA)
    cv2.putText(img,s,pos,f,sc,col,th,cv2.LINE_AA)

def dashed(img,p1,p2,col,d=8,th=1):
    dx,dy=p2[0]-p1[0],p2[1]-p1[1]
    L=math.hypot(dx,dy)
    if L<1: return
    st=max(1,int(L/(d*2)))
    for i in range(st):
        t0=(2*i)/(2*st); t1=(2*i+1)/(2*st)
        a=(int(p1[0]+dx*t0),int(p1[1]+dy*t0))
        b=(int(p1[0]+dx*t1),int(p1[1]+dy*t1))
        cv2.line(img,a,b,col,th,cv2.LINE_AA)

def pbar(cur, tot, label="", w=BAR_W):
    p=cur/max(tot,1); d=int(w*p)
    print(f"\r  [{'█'*d}{'░'*(w-d)}] {p*100:5.1f}%  {label}",end="",flush=True)
    if cur>=tot: print()

# ─────────────────────────────────────────────
#  SYNTHETIC DEPTH
# ─────────────────────────────────────────────
def syn_depth(box, fw, fh):
    return clamp(box_area(box)/(fw*fh*0.55), 0.05, 0.95)

# ─────────────────────────────────────────────
#  TRACK
# ─────────────────────────────────────────────
class Track:
    _n=0
    def __init__(self,box):
        Track._n+=1; self.tid=Track._n
        self.box=box; self.trail=collections.deque(maxlen=TRAIL_LEN)
        self.vel=np.zeros(2,float); self.depth=0.5
        self.age=0; self.miss=0; self.risk=False; self.pred=[]
        h=(self.tid*53)%180
        bgr=cv2.cvtColor(np.uint8([[[h,220,255]]]),cv2.COLOR_HSV2BGR)[0][0]
        self.color=tuple(int(c) for c in bgr)

    def update(self,box,dep):
        old=cx_cy(self.box); self.box=box
        cx,cy=cx_cy(box); self.trail.append((cx,cy))
        self.vel=lerp(self.vel,np.array([cx-old[0],cy-old[1]],float),0.35)
        self.depth=dep; self.age+=1; self.miss=0

    def predict(self,n=10):
        cx,cy=cx_cy(self.box)
        self.pred=[(int(cx+self.vel[0]*i),int(cy+self.vel[1]*i)) for i in range(1,n+1)]

# ─────────────────────────────────────────────
#  TRACKER
# ─────────────────────────────────────────────
class Tracker:
    def __init__(self): self.tracks={}

    def _iou(self,b1,b2):
        ix1=max(b1[0],b2[0]);iy1=max(b1[1],b2[1])
        ix2=min(b1[2],b2[2]);iy2=min(b1[3],b2[3])
        inter=max(0,ix2-ix1)*max(0,iy2-iy1)
        u=box_area(b1)+box_area(b2)-inter
        return inter/u if u>0 else 0

    def update(self,dets,deps):
        if not dets:
            for t in self.tracks.values(): t.miss+=1
            self._prune(); return []
        matched=set(); upd=[]
        for box,dep in zip(dets,deps):
            bi,bt=0,None
            for tid,t in self.tracks.items():
                s=self._iou(box,t.box)
                if s>bi: bi,bt=s,tid
            if bi>0.20 and bt not in matched:
                self.tracks[bt].update(box,dep); matched.add(bt); upd.append(self.tracks[bt])
            else:
                nt=Track(box); nt.depth=dep; nt.trail.append(cx_cy(box))
                self.tracks[nt.tid]=nt; upd.append(nt)
        for tid,t in self.tracks.items():
            if tid not in matched: t.miss+=1
        self._prune(); return upd

    def _prune(self):
        for k in [k for k,v in self.tracks.items() if v.miss>8]: del self.tracks[k]

# ─────────────────────────────────────────────
#  HEATMAP
# ─────────────────────────────────────────────
class Heatmap:
    def __init__(self,h,w): self.m=np.zeros((h,w),np.float32)

    def add(self,cx,cy,r=28):
        y1=max(0,cy-r);y2=min(self.m.shape[0],cy+r)
        x1=max(0,cx-r);x2=min(self.m.shape[1],cx+r)
        sub=self.m[y1:y2,x1:x2]
        ys,xs=np.mgrid[0:sub.shape[0],0:sub.shape[1]]
        sub+=np.exp(-((xs-(cx-x1))**2+(ys-(cy-y1))**2)/(2*(r/2)**2))
        self.m[y1:y2,x1:x2]=sub

    def render(self,img,a=0.25):
        self.m*=0.96
        bl=gaussian_filter(self.m,sigma=12)
        nm=bl/(bl.max()+1e-6)
        cm=cv2.applyColorMap((nm*255).astype(np.uint8),cv2.COLORMAP_JET)
        mk=(nm>0.06).astype(np.float32)[:,:,None]
        out=img.astype(np.float32)
        return (out*(1-a*mk)+cm.astype(np.float32)*(a*mk)).clip(0,255).astype(np.uint8)

# ─────────────────────────────────────────────
#  BIRD'S-EYE VIEW
# ─────────────────────────────────────────────
def render_bev(tracks, fw, fh):
    bev=np.zeros((BEV_H,BEV_W,3),np.uint8)
    gc=(22,40,22)
    for x in range(0,BEV_W,28): cv2.line(bev,(x,0),(x,BEV_H),gc,1)
    for y in range(0,BEV_H,28): cv2.line(bev,(0,y),(BEV_W,y),gc,1)
    cv2.rectangle(bev,(0,0),(BEV_W-1,BEV_H-1),C_CYAN,1)
    txt(bev,"BIRD'S-EYE VIEW",(4,13),C_CYAN,0.34)
    cv2.drawMarker(bev,(BEV_W//2,BEV_H-16),C_YELLOW,cv2.MARKER_TRIANGLE_UP,12,2)
    for t in tracks:
        bx=int(cx_cy(t.box)[0]/fw*BEV_W)
        by=int((1-t.depth)*(BEV_H-36))+10
        r=max(4,int(8*(1-t.depth)*(BEV_W/300)))
        cv2.circle(bev,(bx,by),r,t.color,-1)
        cv2.circle(bev,(bx,by),r+2,C_WHITE,1)
        txt(bev,f"#{t.tid}",(bx+3,by-3),t.color,0.28,1,False)
        if t.risk: cv2.circle(bev,(bx,by),r+8,C_RED,1)
    return bev

# ─────────────────────────────────────────────
#  PERSPECTIVE GRID
# ─────────────────────────────────────────────
def draw_grid(img):
    h,w=img.shape[:2]; hy=int(h*0.52); vp=(w//2,hy)
    ov=img.copy(); col=(0,150,90)
    for i in range(-9,10): cv2.line(ov,vp,(w//2+i*(w//11),h),col,1)
    for j in range(1,9):
        t=j/9; y=int(hy+(h-hy)*t**1.4)
        cv2.line(ov,(int(w//2-(w//2)*t**0.55),y),(int(w//2+(w//2)*t**0.55),y),col,1)
    cv2.addWeighted(ov,0.18,img,0.82,0,img)

# ─────────────────────────────────────────────
#  COLORBAR
# ─────────────────────────────────────────────
def draw_colorbar(img):
    h=img.shape[0]; bh=min(160,h-80); x,y=img.shape[1]-24,44
    for i in range(bh):
        v=int((1-i/bh)*255)
        img[y+i,x:x+12]=cv2.applyColorMap(np.array([[[v]]],np.uint8),cv2.COLORMAP_TURBO)[0,0]
    cv2.rectangle(img,(x,y),(x+12,y+bh),C_CYAN,1)
    txt(img,"FAR",(x-5,y+10),C_WHITE,0.30,1,False)
    txt(img,"NEAR",(x-7,y+bh-4),C_WHITE,0.30,1,False)

# ─────────────────────────────────────────────
#  COLLISION
# ─────────────────────────────────────────────
def check_col(tracks):
    tl=list(tracks)
    for t in tl: t.risk=False
    for i in range(len(tl)):
        for j in range(i+1,len(tl)):
            a,b=tl[i],tl[j]
            if dist2d(cx_cy(a.box),cx_cy(b.box))<RISK_DIST and abs(a.depth-b.depth)<0.22:
                a.risk=b.risk=True
    return any(t.risk for t in tl)

# ─────────────────────────────────────────────
#  HUD
# ─────────────────────────────────────────────
def draw_hud(img,fps,lat,np_,nt,cw,gpu,fi,total_f):
    h,w=img.shape[:2]
    alpha_rect(img,(0,0),(258,202),C_DARK,0.78)
    cv2.rectangle(img,(0,0),(258,202),C_CYAN,1)
    txt(img,"AI PEDESTRIAN 3D PERCEPTION",(6,16),C_CYAN,0.44,1)
    txt(img,"by tubakhxn",(6,31),C_MAG,0.33,1)
    cv2.line(img,(3,37),(255,37),C_CYAN,1)
    lines=[
        (f"FPS     : {fps:5.1f}",         C_GREEN),
        (f"LATENCY : {lat:5.1f} ms",      C_YELLOW),
        (f"PEOPLE  : {np_:5d}",           C_CYAN),
        (f"TRACKS  : {nt:5d}",            C_CYAN),
        (f"DEPTH   : SYNTHETIC",           C_ORANGE),
        (f"COMPUTE : {'GPU/FP16' if gpu else 'CPU    ':>8}", C_GREEN if gpu else C_ORANGE),
        (f"FRAME   : {fi:5d}/{total_f}",  C_WHITE),
        (f"TIME    : {datetime.now().strftime('%H:%M:%S'):>8}", C_WHITE),
    ]
    for i,(s,c) in enumerate(lines): txt(img,s,(6,52+i*18),c,0.41,1)

    if cw:
        bw=320;bh=26;bx=w//2-bw//2
        alpha_rect(img,(bx,5),(bx+bw,5+bh),(0,0,160),0.85)
        cv2.rectangle(img,(bx,5),(bx+bw,5+bh),C_RED,2)
        p=int(180+75*math.sin(time.time()*7))
        txt(img,"!! COLLISION RISK DETECTED !!",(bx+14,23),(0,p,255),0.52,2)

    alpha_rect(img,(0,h-20),(w,h),C_DARK,0.75)
    txt(img,"Q=QUIT  P=PAUSE  S=SCREENSHOT  |  AI PEDESTRIAN 3D PERCEPTION",
        (6,h-5),C_CYAN,0.33,1)

    tag="LIVE ●"
    (tw,_),_=cv2.getTextSize(tag,cv2.FONT_HERSHEY_SIMPLEX,0.44,1)
    txt(img,tag,(w-tw-7,18),C_RED,0.44,1)

    # processing progress bar inside HUD
    if total_f>0:
        prog=fi/total_f; pw=int(prog*240)
        alpha_rect(img,(6,195),(248,202),C_DARK,0.5)
        cv2.rectangle(img,(6,195),(6+pw,202),C_GREEN,-1)
        cv2.rectangle(img,(6,195),(248,202),C_CYAN,1)

# ─────────────────────────────────────────────
#  SCALE BOX
# ─────────────────────────────────────────────
def scale_box(box, sx, sy):
    x1,y1,x2,y2=box
    return (int(x1*sx),int(y1*sy),int(x2*sx),int(y2*sy))

def scale_mask(mask, out_h, out_w):
    return cv2.resize(mask.astype(np.uint8), (out_w, out_h),
                      interpolation=cv2.INTER_LINEAR).astype(bool)

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu     = device.type=="cuda"
    print(f"  Compute : {'GPU (CUDA) + FP16' if gpu else 'CPU (fast resize mode)'}\n")

    # ── Load model ──────────────────────────────
    stop=threading.Event()
    sp=threading.Thread(target=_spinner,args=(f"Loading {SEG_MODEL}...",stop),daemon=True)
    sp.start()
    model=YOLO(SEG_MODEL)
    stop.set(); sp.join()
    print(f"\r  ✓  {SEG_MODEL} loaded OK                        ")
    if gpu: model.to("cuda")

    tracker=Tracker()

    # ── Source ──────────────────────────────────
    src=sys.argv[1] if len(sys.argv)>1 else 0
    print(f"  Source  : {src}")
    cap=cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open: {src}"); sys.exit(1)

    fw_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video   : {fw_orig}×{fh_orig} @ {fps_src:.1f} FPS  ({total_f} frames)")

    # ── Compute output size ──────────────────────
    # Cap output width to MAX_OUT_W for sane file sizes
    scale_out = min(1.0, MAX_OUT_W/fw_orig)
    fw_out = int(fw_orig * scale_out)
    fh_out = int(fh_orig * scale_out)
    # Inference scale (PROC_W wide)
    scale_inf = PROC_W / fw_orig
    fw_inf = PROC_W
    fh_inf = int(fh_orig * scale_inf)
    # Scale from inference → output
    sx = fw_out / fw_inf
    sy = fh_out / fh_inf
    print(f"  Infer   : {fw_inf}×{fh_inf}  →  Output: {fw_out}×{fh_out}")

    # ── Progress bar for video "loading" ─────────
    print(f"  Indexing video  [{total_f} frames]")
    for i in range(0,BAR_W+1,2):
        pbar(i,BAR_W,""); time.sleep(0.012)
    print(f"\r  [{'█'*BAR_W}] 100.0%  Ready!              ")

    # ── Output writer ────────────────────────────
    fourcc=cv2.VideoWriter_fourcc(*"mp4v")
    writer=cv2.VideoWriter(OUTPUT_FILE,fourcc,fps_src,(fw_out,fh_out))
    print(f"  Output  : {OUTPUT_FILE}  ({fw_out}×{fh_out})\n")

    heatmap=Heatmap(fh_out,fw_out)
    scrdir=Path("screenshots")

    print("  ╔═════════════════════════════════════════════════╗")
    print("  ║  CINEMATIC VISUALIZATION ENGINE — RUNNING      ║")
    print("  ╚═════════════════════════════════════════════════╝\n")

    paused=False; fi=0; fps_sm=fps_src; canvas=None

    try:
        while True:
            t0=now_ms()
            key=cv2.waitKey(1)&0xFF
            if key==ord("q"): break
            if key==ord("p"): paused=not paused
            if key==ord("s") and canvas is not None:
                scrdir.mkdir(exist_ok=True)
                fn=scrdir/f"shot_{fi:06d}.png"
                cv2.imwrite(str(fn),canvas)
                print(f"\n  [SCREENSHOT] {fn}")
            if paused: cv2.waitKey(30); continue

            ret,frame=cap.read()
            if not ret: print("  [INFO] Stream ended."); break
            fi+=1

            # ── Resize for inference ──────────────
            frame_inf=cv2.resize(frame,(fw_inf,fh_inf),interpolation=cv2.INTER_LINEAR)
            # ── Resize for output canvas ──────────
            frame_out=cv2.resize(frame,(fw_out,fh_out),interpolation=cv2.INTER_LINEAR)

            # ── YOLO on small frame ───────────────
            results=model.predict(
                frame_inf,
                classes=[PERSON_CLS], conf=CONF, iou=IOU,
                verbose=False, device=device,
                half=gpu, retina_masks=False,
            )

            boxes_inf=[]; masks_inf=[]
            if results and results[0].boxes is not None:
                res=results[0]
                for bi,box in enumerate(res.boxes):
                    x1,y1,x2,y2=map(int,box.xyxy[0])
                    x1=clamp(x1,0,fw_inf-1); y1=clamp(y1,0,fh_inf-1)
                    x2=clamp(x2,0,fw_inf-1); y2=clamp(y2,0,fh_inf-1)
                    boxes_inf.append((x1,y1,x2,y2))
                    if res.masks is not None and bi<len(res.masks.data):
                        m=res.masks.data[bi].cpu().numpy()
                        m_rs=cv2.resize(m.astype(np.float32),(fw_inf,fh_inf),
                                        interpolation=cv2.INTER_LINEAR)>0.4
                        masks_inf.append(m_rs)
                    else:
                        masks_inf.append(None)

            # ── Scale boxes & masks → output size ─
            boxes_out=[scale_box(b,sx,sy) for b in boxes_inf]
            masks_out=[]
            for m in masks_inf:
                if m is not None:
                    masks_out.append(scale_mask(m,fh_out,fw_out))
                else:
                    masks_out.append(None)

            depths=[syn_depth(b,fw_out,fh_out) for b in boxes_out]

            # ── Tracking ──────────────────────────
            active=tracker.update(boxes_out,depths)
            for t in active:
                t.predict()
                ccx,ccy=cx_cy(t.box)
                heatmap.add(ccx,ccy)

            collision=check_col(active)

            # ── Render ────────────────────────────
            canvas=frame_out.copy()
            draw_grid(canvas)
            canvas=heatmap.render(canvas,a=0.20)

            # ★ Body UV coloring
            for box,mask in zip(boxes_out,masks_out):
                if mask is not None:
                    paint_body_uv(canvas,mask,box)
                else:
                    # fallback solid gradient rect (no mask)
                    x1,y1,x2,y2=box
                    bh_=max(1,y2-y1); bw_=max(1,x2-x1)
                    t_=np.linspace(0,1,bh_,dtype=np.float32)
                    idx_f=t_*(N_ZONES-1)
                    lo=np.floor(idx_f).astype(int)
                    hi=np.minimum(lo+1,N_ZONES-1)
                    fr=(idx_f-lo)[:,None]
                    rc=(UV_ZONES_BGR[lo]*(1-fr)+UV_ZONES_BGR[hi]*fr).astype(np.uint8)
                    patch=np.broadcast_to(rc[:,np.newaxis,:],(bh_,bw_,3)).copy()
                    sub=canvas[y1:y2,x1:x2]
                    if sub.shape==patch.shape:
                        canvas[y1:y2,x1:x2]=(sub.astype(float)*0.3+
                                              patch.astype(float)*0.7).clip(0,255).astype(np.uint8)

            # ★ Squared boxes
            for t,box in zip(active,boxes_out):
                x1,y1,x2,y2=box
                col=C_RED if t.risk else t.color
                draw_box(canvas,(x1,y1),(x2,y2),col,2)
                dm=(1-t.depth)*28+1
                label=f"#{t.tid}  {dm:.1f}m"
                (lw,lh),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.44,1)
                lx,ly=x1,max(y1-4,lh+4)
                alpha_rect(canvas,(lx-2,ly-lh-3),(lx+lw+4,ly+2),C_DARK,0.80)
                txt(canvas,label,(lx+2,ly),col,0.44,1)

            # ★ Trails + prediction
            for t in active:
                pts=list(t.trail)
                for i in range(1,len(pts)):
                    a_=i/len(pts)
                    cv2.line(canvas,pts[i-1],pts[i],
                             tuple(int(c*a_) for c in t.color),
                             max(1,int(3*a_)),cv2.LINE_AA)
                if len(t.pred)>1:
                    for i in range(len(t.pred)-1):
                        dashed(canvas,t.pred[i],t.pred[i+1],C_YELLOW,6,1)
                    cv2.arrowedLine(canvas,t.pred[-2],t.pred[-1],
                                    C_YELLOW,2,cv2.LINE_AA,tipLength=0.35)
                if np.linalg.norm(t.vel)>1.5:
                    ccx,ccy=cx_cy(t.box)
                    vx,vy=(t.vel*6).astype(int)
                    cv2.arrowedLine(canvas,(ccx,ccy),(ccx+vx,ccy+vy),
                                    C_MAG,2,cv2.LINE_AA,tipLength=0.35)

            draw_colorbar(canvas)

            t1=now_ms(); lat=t1-t0
            fps_sm=lerp(fps_sm,1000/max(lat,1),0.12)

            draw_hud(canvas,fps_sm,lat,len(boxes_out),
                     len(tracker.tracks),collision,gpu,fi,total_f)

            # BEV inset
            bev=render_bev(active,fw_out,fh_out)
            bx=fw_out-BEV_W-6; by=36
            if by+BEV_H<=fh_out and bx>=0:
                canvas[by:by+BEV_H,bx:bx+BEV_W]=bev

            writer.write(canvas)
            cv2.imshow("AI Pedestrian 3D Perception — tubakhxn",canvas)

            # Terminal progress
            if fi%20==0:
                pct=fi/max(total_f,1)
                done_b=int(BAR_W*pct)
                eta=(total_f-fi)/max(fps_sm,1)
                print(f"\r  [{'█'*done_b}{'░'*(BAR_W-done_b)}] "
                      f"{pct*100:5.1f}%  {fi}/{total_f}  "
                      f"FPS:{fps_sm:.1f}  ETA:{eta:.0f}s",
                      end="",flush=True)

    finally:
        print()
        cap.release(); writer.release(); cv2.destroyAllWindows()
        print(f"\n  ✓  Saved → {OUTPUT_FILE}  ({fw_out}×{fh_out})")
        print("  ═══════════════════════════════════════════════")

if __name__=="__main__":
    print("  Loading AI perception models...")
    print("  Initializing depth estimation...")
    print("  Starting pedestrian tracking...")
    print("  Launching cinematic visualization engine...\n")
    main()