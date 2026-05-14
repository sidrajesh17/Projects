import os
import csv
import cv2
import time
import threading
import numpy as np
from collections import deque
from flask import Flask, render_template, Response, jsonify, request
import scipy.signal as sps
import logging

logging.getLogger('werkzeug').setLevel(logging.ERROR)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Dependencies
try:
    import mediapipe as mp
    import mediapipe.python.solutions.face_mesh as mp_face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.3, min_tracking_confidence=0.3)
except Exception as e:
    mp_face_mesh = None
    face_mesh = None
    print("MediaPipe not available:", e)

try:
    from deepface import DeepFace
except Exception as e:
    DeepFace = None
    print("DeepFace not available:", e)

# --- AI SIGNAL QUALITY BOUNCER ---
try:
    from tf_keras.models import load_model
    if os.path.exists("mindmirror_sqi_bouncer.h5"):
        sqi_model = load_model("mindmirror_sqi_bouncer.h5", compile=False)
        sqi_active = True
        print("🧠 [SUCCESS] AI Signal Quality Bouncer Armed!")
    else:
        sqi_model = None
        sqi_active = False
except Exception as e:
    sqi_model = None
    sqi_active = False

# ---------- CONFIG ----------
CAMERA_FPS = 30
AI_FPS = 5
COLOR_BUFFER_LEN = 300
HR_MIN_WINDOW = 150
DETECT_SPIKE_ABS = 15.0
DETECT_SPIKE_PCT = 0.20
BASELINE_CAPTURE_SECONDS = 10
FACE_LOST_CLEAR_SECONDS = 3.0

RESULTS_CSV = "results.csv"
TRAIN_CSV = "training_data.csv"
LOG_INTERVAL_SECONDS = 1.0
NEGATIVE_EMOTIONS = {"angry", "fear", "sad", "disgust"}
TRUE_STRESS_HR_OFFSET = 10.0

# ---------- GLOBALS ----------
output_frame = None
status_lock = threading.Lock()
csv_lock = threading.Lock()

current_hr = None
current_math_hr = None
current_sqi_score = 1.0 # 1.0 = Perfect Signal
current_emotion = "unknown"
processing_enabled = True
baseline_value = None
baseline_buffer = deque(maxlen=300)
capturing_baseline = False
hr_buffer = deque(maxlen=45) 
color_buffer = deque(maxlen=COLOR_BUFFER_LEN)

physical_ground_truth_hr = None
training_buffer = []

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

app = Flask(__name__)

CHEEK_LEFT = [118, 119, 100, 120, 114, 205, 50, 101, 117]
CHEEK_RIGHT = [347, 348, 329, 349, 343, 425, 280, 330, 346]
FOREHEAD = [103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356, 71, 68, 104, 69, 108] 

def ensure_results_csv():
    if not os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp", "emotion", "hr", "baseline", "predicted_stress", "true_stress", "delta", "delta_pct"])

def compute_true_stress(hr, baseline, emotion):
    if hr is None: return 0
    hr_rule = hr > (baseline + TRUE_STRESS_HR_OFFSET) if baseline is not None else False
    return 1 if (hr_rule or emotion in NEGATIVE_EMOTIONS) else 0

def log_results_row(hr, emotion, baseline, predicted_stress, true_stress):
    delta = pct = None
    if hr is not None and baseline is not None:
        delta = round(hr - baseline, 1)
        pct = round((delta / baseline) if baseline else 0.0, 3)

    with csv_lock:
        with open(RESULTS_CSV, "a", newline="") as f:
            csv.writer(f).writerow([time.strftime("%Y-%m-%d %H:%M:%S"), emotion, hr, baseline, int(predicted_stress), int(true_stress), delta, pct])


def processing_thread():
    global output_frame, current_hr, current_math_hr, current_emotion, color_buffer, training_buffer, current_sqi_score

    ensure_results_csv()
    last_ai_time = 0; ai_interval = 1.0 / AI_FPS
    last_hr_calc_time = 0; hr_calc_interval = 1.0 / 3.0 
    face_lost_time = None; last_log_time = 0
    
    is_hr_calculating = [False]

    def analyze_emotion(roi):
        global current_emotion
        if DeepFace is None: return
        try:
            result = DeepFace.analyze(roi, actions=['emotion'], enforce_detection=False, silent=True)
            emo = (result[0] if isinstance(result, list) else result).get('dominant_emotion', 'unknown')
            with status_lock: current_emotion = emo
        except Exception: pass

    def calculate_hr_worker(seq_arr, b_val, use_sqi, capturing_base):
        global current_hr, current_math_hr, current_sqi_score
        math_hr = None
        sqi_score = 1.0
        print(f"[HR-WORKER] Started. buf={len(seq_arr)}, b_val={b_val}, use_sqi={use_sqi}")
        
        try:
            # 1. PURE CHROM CALCULATION - always runs
            b_arr, g_arr, r_arr = seq_arr[:, 2], seq_arr[:, 1], seq_arr[:, 0]
            Xs = (3.0*r_arr - 2.0*g_arr); Ys = (1.5*r_arr + g_arr - 1.5*b_arr)
            Xs, Ys = (Xs - Xs.mean()) / (Xs.std()+1e-8), (Ys - Ys.mean()) / (Ys.std()+1e-8)
            Xs, Ys = sps.detrend(Xs), sps.detrend(Ys)
            wl = 9 if len(Xs) >= 9 else (len(Xs)//2)*2+1
            if wl >= 5: Xs, Ys = sps.savgol_filter(Xs, wl, 3), sps.savgol_filter(Ys, wl, 3)

            min_bpm = max(40, b_val - 30) if b_val else 45
            max_bpm = min(180, b_val + 40) if b_val else 150
            nyq = 0.5 * 30  # Nyquist = 15 Hz
            low, high = min_bpm / 60.0 / nyq, max_bpm / 60.0 / nyq
            low, high = max(low, 0.01), min(high, 0.99)

            if low < high:
                sos = sps.butter(3, [low, high], btype='band', output='sos')
                Xf, Yf = sps.sosfiltfilt(sos, Xs), sps.sosfiltfilt(sos, Ys)
                S = sps.detrend(Xf - (np.std(Xf)/(np.std(Yf)+1e-8))*Yf)
                f, Pxx = sps.welch(S, fs=30, nperseg=min(256, max(16, len(S))))
                idx = np.where((f >= min_bpm/60.0) & (f <= max_bpm/60.0))[0]
                if idx.size > 0:
                    math_hr = float(round(f[idx[np.argmax(Pxx[idx])]]*60.0, 1))
            print(f"[HR-WORKER] math_hr={math_hr}, sqi={sqi_score:.2f}")

            # 2. RUN AI SIGNAL QUALITY BOUNCER (if loaded)
            if use_sqi and sqi_model:
                seq_scaled = (seq_arr - seq_arr.mean(axis=0)) / (seq_arr.std(axis=0) + 1e-8)
                sqi_pred = sqi_model(np.expand_dims(seq_scaled, axis=0), training=False).numpy()[0][0]
                sqi_score = float(sqi_pred)

            # 3. SMOOTH CONFIDENCE-WEIGHTED BLEND
            # Instead of a hard gate that blocks updates, the SQI score controls
            # HOW MUCH we trust the new CHROM reading vs the last stable value.
            # SQI=0.95 (clean) → 95% new, 5% old → fast response to real changes
            # SQI=0.3  (noisy) → 30% new, 70% old → dampened, resists artifacts
            final_hr = None
            with status_lock:
                last_stable = current_hr

            if math_hr is not None:
                if use_sqi and last_stable is not None:
                    # Clamp SQI to a minimum of 0.15 so HR is never fully frozen
                    weight = max(sqi_score, 0.15)
                    final_hr = round((math_hr * weight) + (last_stable * (1.0 - weight)), 1)
                else:
                    # No SQI or first reading ever — accept math directly
                    final_hr = math_hr

            if final_hr is not None:
                hr_buffer.append(final_hr)
                hr_avg = float(np.median(list(hr_buffer))) if len(hr_buffer) >= 10 else sum(hr_buffer)/len(hr_buffer)
                with status_lock:
                    current_hr = round(hr_avg, 1)
                    current_math_hr = math_hr
                    current_sqi_score = sqi_score
                    if capturing_base: baseline_buffer.append(current_hr)

        except Exception as e:
            print(f"[HR-WORKER ERROR] {e}")
        finally: is_hr_calculating[0] = False


    while True:
        ret, frame = cap.read()
        if not ret: time.sleep(0.01); continue

        frame = cv2.flip(frame, 1)
        target_img = frame.copy() 
        h_img, w_img = target_img.shape[:2]
        now = time.time()
        display_frame = target_img.copy()

        if processing_enabled and face_mesh is not None:
            rgb = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                face_lost_time = None
                pts = np.array([(int(l.x * w_img), int(l.y * h_img)) for l in landmarks])
                raw_min_x, raw_max_x = np.min(pts[:, 0]), np.max(pts[:, 0])
                raw_min_y, raw_max_y = np.min(pts[:, 1]), np.max(pts[:, 1])
                pad_x, pad_y = int((raw_max_x - raw_min_x)*0.15), int((raw_max_y - raw_min_y)*0.15)
                min_x, max_x = max(0, raw_min_x - pad_x), min(w_img, raw_max_x + pad_x)
                min_y, max_y = max(0, raw_min_y - pad_y), min(h_img, raw_max_y + pad_y)
                
                if now - last_ai_time >= ai_interval:
                    last_ai_time = now
                    face_crop = rgb[min_y:max_y, min_x:max_x].copy()
                    if face_crop.size > 0: threading.Thread(target=analyze_emotion, args=(face_crop,), daemon=True).start()

                mask = np.zeros((h_img, w_img), dtype=np.uint8)
                def get_hull(indices):
                    valid = [i for i in indices if i < len(pts)]
                    return cv2.convexHull(pts[valid])
                cv2.fillConvexPoly(mask, get_hull(CHEEK_LEFT), 255)
                cv2.fillConvexPoly(mask, get_hull(CHEEK_RIGHT), 255)
                cv2.fillConvexPoly(mask, get_hull(FOREHEAD), 255)
                
                overlay = display_frame.copy()
                overlay[mask == 255] = [0, 255, 0]
                cv2.addWeighted(overlay, 0.15, display_frame, 0.85, 0, display_frame)
                
                ycrcb = cv2.cvtColor(target_img, cv2.COLOR_BGR2YCrCb)
                skin_mask = cv2.inRange(ycrcb, np.array([0, 135, 85]), np.array([255, 180, 135]))
                final_mask = cv2.bitwise_and(mask, skin_mask)
                
                mean_val = cv2.mean(target_img, mask=final_mask if np.count_nonzero(final_mask) > 100 else mask)
                r, g, b = mean_val[2], mean_val[1], mean_val[0]
                color_buffer.append((r, g, b))

                if physical_ground_truth_hr is not None:
                    training_buffer.append([now, r, g, b, physical_ground_truth_hr])
                    if len(training_buffer) >= 60:
                        with open(TRAIN_CSV, "a", newline="") as f: csv.writer(f).writerows(training_buffer)
                        training_buffer.clear()

                if not is_hr_calculating[0] and (now - last_hr_calc_time >= hr_calc_interval):
                    if len(color_buffer) >= 150:
                        is_hr_calculating[0] = True; last_hr_calc_time = now
                        threading.Thread(target=calculate_hr_worker, args=(np.array(color_buffer)[-150:], baseline_value, sqi_active, capturing_baseline), daemon=True).start()
            else:
                if face_lost_time is None: face_lost_time = now
                elif now - face_lost_time > FACE_LOST_CLEAR_SECONDS:
                    color_buffer.clear(); hr_buffer.clear()
                    with status_lock: current_hr = current_math_hr = None; current_sqi_score = 1.0

        with status_lock:
            disp_hr = current_hr; disp_emo = current_emotion; disp_base = baseline_value

        if processing_enabled and (now - last_log_time >= LOG_INTERVAL_SECONDS):
            if disp_hr is not None:
                predicted_stress = 1 if (disp_base and (abs(disp_hr - disp_base) >= DETECT_SPIKE_ABS or abs((disp_hr - disp_base) / disp_base) >= DETECT_SPIKE_PCT)) else 0
                log_results_row(disp_hr, disp_emo, disp_base, predicted_stress, compute_true_stress(disp_hr, disp_base, disp_emo))
            last_log_time = now
        with status_lock: output_frame = display_frame
        time.sleep(0.005)

def generate_mjpeg():
    global output_frame
    while True:
        with status_lock:
            flag, encoded = cv2.imencode(".jpg", output_frame if output_frame is not None else np.zeros((320, 420, 3), dtype=np.uint8))
        if flag: yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encoded) + b'\r\n')
        time.sleep(0.03)

def baseline_capture_worker(dur=BASELINE_CAPTURE_SECONDS):
    global capturing_baseline, baseline_value, hr_buffer, color_buffer
    capturing_baseline = True; baseline_buffer.clear()
    start = time.time()
    while time.time() - start < dur: time.sleep(0.5)
    samples = [v for v in list(baseline_buffer) if v is not None]
    baseline_value = round(sum(samples) / len(samples), 1) if len(samples) >= 3 else None
    capturing_baseline = False; hr_buffer.clear(); color_buffer.clear()

@app.route("/")
def index(): return render_template("index.html")

@app.route("/video_feed")
def video_feed(): return Response(generate_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/status")
def status():
    with status_lock:
        hr = current_hr; emo = current_emotion; base = baseline_value
        sqi = current_sqi_score
        
    delta = pct = None; spike = False
    if hr is not None and base is not None:
        delta = round(hr - base, 1)
        pct = round((delta / base) if base else 0.0, 3)
        if abs(delta) >= DETECT_SPIKE_ABS or abs(pct) >= DETECT_SPIKE_PCT: spike = True

    return jsonify({
        "hr": hr, "emotion": emo, "baseline": base,
        "sqi": sqi,
        "spike": spike, "delta": delta, "delta_pct": pct,
        "processing": processing_enabled,
        "capturing_baseline": capturing_baseline,
        "sqi_active": sqi_active,
        "training_active": physical_ground_truth_hr is not None
    })

@app.route("/start", methods=["POST"])
def start_processing(): global processing_enabled; processing_enabled = True; return jsonify({"ok": True})

@app.route("/stop", methods=["POST"])
def stop_processing(): global processing_enabled; processing_enabled = False; return jsonify({"ok": True})

@app.route("/capture_baseline", methods=["POST"])
def capture_baseline():
    global capturing_baseline
    if capturing_baseline: return jsonify({"ok": False})
    threading.Thread(target=baseline_capture_worker, daemon=True).start()
    return jsonify({"ok": True, "message": f"Capturing base hr for {BASELINE_CAPTURE_SECONDS}s."})

@app.route("/set_baseline", methods=["POST"])
def set_baseline():
    global baseline_value, hr_buffer, color_buffer
    try:
        val = float((request.get_json(silent=True) or {}).get("baseline", 0))
        if not (30 <= val <= 200): raise ValueError()
        baseline_value = round(val, 1); hr_buffer.clear(); color_buffer.clear() 
        return jsonify({"ok": True, "baseline": baseline_value})
    except: return jsonify({"ok": False})

@app.route("/reset_baseline", methods=["POST"])
def reset_baseline():
    global baseline_value, current_hr, hr_buffer
    baseline_value = None; current_hr = None; hr_buffer.clear(); return jsonify({"ok": True})

@app.route("/set_physical_hr", methods=["POST"])
def set_physical_hr():
    global physical_ground_truth_hr, baseline_value, hr_buffer, color_buffer
    try:
        payload = request.get_json(silent=True) or {}
        if "hr" in payload and payload["hr"] is not None:
            val = float(payload["hr"])
            physical_ground_truth_hr = val
            # Auto-seed baseline so CHROM search band is immediately calibrated
            if baseline_value is None:
                baseline_value = val
                hr_buffer.clear(); color_buffer.clear()
            return jsonify({"ok": True, "message": f"Ground truth @ {val} BPM injected. Baseline auto-seeded."})
        else:
            physical_ground_truth_hr = None
            return jsonify({"ok": True, "message": "Training mode deactivated."})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

if __name__ == "__main__":
    ensure_results_csv()
    threading.Thread(target=processing_thread, daemon=True).start()
    app.run(host="0.0.0.0", port=5555, debug=False, threaded=True)