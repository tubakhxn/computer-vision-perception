// pedestrian.cpp
// Basic AI Pedestrian 3D Perception System (C++ version)
// Minimal comments, logic follows Python version

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <deque>
#include <cmath>
#include <ctime>
#include <string>

using namespace cv;
using namespace std;

// Config
const float CONF = 0.35f;
const float IOU = 0.45f;
const int PERSON_CLS = 0;
const string OUTPUT_FILE = "output_pedestrian_3d_ai.mp4";
const int TRAIL_LEN = 32;
const int RISK_DIST = 85;
const int PROC_W = 640;
const int MAX_OUT_W = 1280;
const int BEV_W = 260, BEV_H = 340;

// Colors (BGR)
Scalar C_CYAN(220, 255, 0);
Scalar C_MAG(255, 0, 200);
Scalar C_YELLOW(255, 220, 0);
Scalar C_RED(255, 40, 0);
Scalar C_GREEN(120, 255, 0);
Scalar C_ORANGE(255, 150, 0);
Scalar C_WHITE(255, 255, 255);
Scalar C_DARK(18, 8, 8);

// Utility
float clamp(float v, float lo, float hi) { return max(lo, min(hi, v)); }
Point cx_cy(Rect box) { return Point((box.x + box.x + box.width) / 2, (box.y + box.y + box.height) / 2); }
float box_area(Rect box) { return max(0, box.width) * max(0, box.height); }
float dist2d(Point a, Point b) { return hypot(a.x - b.x, a.y - b.y); }

// Synthetic depth
float syn_depth(Rect box, int fw, int fh) {
    return clamp(box_area(box) / (fw * fh * 0.55f), 0.05f, 0.95f);
}

// Track
struct Track {
    static int next_id;
    int tid;
    Rect box;
    deque<Point> trail;
    Point2f vel;
    float depth;
    int age, miss;
    bool risk;
    vector<Point> pred;
    Scalar color;
    Track(Rect b) : box(b), vel(0, 0), depth(0.5f), age(0), miss(0), risk(false) {
        tid = ++next_id;
        trail.push_back(cx_cy(box));
        color = Scalar(rand() % 256, rand() % 256, rand() % 256);
    }
    void update(Rect b, float dep) {
        Point old = cx_cy(box);
        box = b;
        Point c = cx_cy(b);
        trail.push_back(c);
        if (trail.size() > TRAIL_LEN) trail.pop_front();
        vel = 0.65f * vel + 0.35f * Point2f(c.x - old.x, c.y - old.y);
        depth = dep;
        age++;
        miss = 0;
    }
    void predict(int n = 10) {
        Point c = cx_cy(box);
        pred.clear();
        for (int i = 1; i <= n; ++i) {
            pred.push_back(Point(int(c.x + vel.x * i), int(c.y + vel.y * i)));
        }
    }
};
int Track::next_id = 0;

// Tracker
struct Tracker {
    vector<Track> tracks;
    void update(const vector<Rect>& dets, const vector<float>& deps) {
        for (auto& t : tracks) t.miss++;
        for (size_t i = 0; i < dets.size(); ++i) {
            bool matched = false;
            for (auto& t : tracks) {
                float iou = (float)(dets[i] & t.box).area() / ((float)(dets[i] | t.box).area() + 1e-6f);
                if (iou > 0.20f && t.miss < 8) {
                    t.update(dets[i], deps[i]);
                    matched = true;
                    break;
                }
            }
            if (!matched) tracks.emplace_back(dets[i]);
        }
        tracks.erase(remove_if(tracks.begin(), tracks.end(), [](Track& t) { return t.miss > 8; }), tracks.end());
    }
};

// Main
int main(int argc, char** argv) {
    srand((unsigned)time(0));
    string src = argc > 1 ? argv[1] : "0";
    VideoCapture cap(src == "0" ? 0 : src);
    if (!cap.isOpened()) {
        cout << "Cannot open: " << src << endl;
        return 1;
    }
    int fw_orig = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int fh_orig = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
    double fps_src = cap.get(CAP_PROP_FPS);
    int total_f = (int)cap.get(CAP_PROP_FRAME_COUNT);
    float scale_out = min(1.0f, (float)MAX_OUT_W / fw_orig);
    int fw_out = int(fw_orig * scale_out);
    int fh_out = int(fh_orig * scale_out);
    float scale_inf = (float)PROC_W / fw_orig;
    int fw_inf = PROC_W;
    int fh_inf = int(fh_orig * scale_inf);
    float sx = (float)fw_out / fw_inf;
    float sy = (float)fh_out / fh_inf;
    VideoWriter writer(OUTPUT_FILE, VideoWriter::fourcc('m', 'p', '4', 'v'), fps_src, Size(fw_out, fh_out));
    Tracker tracker;
    int fi = 0;
    Mat frame, frame_inf, frame_out, canvas;
    while (true) {
        if (!cap.read(frame)) break;
        resize(frame, frame_inf, Size(fw_inf, fh_inf));
        resize(frame, frame_out, Size(fw_out, fh_out));
        // Placeholder for YOLO detection
        vector<Rect> boxes_inf; // Fill with detected boxes
        vector<float> depths;   // Fill with synthetic depths
        // Example: simulate one box in center
        if (fi % 30 < 15) {
            int bx = fw_inf / 4, by = fh_inf / 4, bw = fw_inf / 2, bh = fh_inf / 2;
            boxes_inf.push_back(Rect(bx, by, bw, bh));
        }
        vector<Rect> boxes_out;
        for (auto& b : boxes_inf) {
            boxes_out.push_back(Rect(int(b.x * sx), int(b.y * sy), int(b.width * sx), int(b.height * sy)));
            depths.push_back(syn_depth(boxes_out.back(), fw_out, fh_out));
        }
        tracker.update(boxes_out, depths);
        for (auto& t : tracker.tracks) t.predict();
        canvas = frame_out.clone();
        for (auto& t : tracker.tracks) {
            rectangle(canvas, t.box, t.risk ? C_RED : t.color, 2);
            putText(canvas, to_string(t.tid), Point(t.box.x, t.box.y - 5), FONT_HERSHEY_SIMPLEX, 0.5, t.color, 1);
            for (size_t i = 1; i < t.trail.size(); ++i) {
                line(canvas, t.trail[i - 1], t.trail[i], t.color, 2);
            }
            if (!t.pred.empty()) {
                for (size_t i = 1; i < t.pred.size(); ++i) {
                    line(canvas, t.pred[i - 1], t.pred[i], C_YELLOW, 1);
                }
            }
        }
        writer.write(canvas);
        imshow("Pedestrian 3D Perception", canvas);
        if (waitKey(1) == 'q') break;
        fi++;
    }
    cap.release();
    writer.release();
    destroyAllWindows();
    cout << "Saved to " << OUTPUT_FILE << endl;
    return 0;
}
