import cv2
import time
import math
import os
import csv
import torch
import joblib
import winsound
import pandas as pd
import torch.nn as nn
import numpy as np
import subprocess
import threading
import json
from datetime import datetime
from collections import defaultdict, deque
from ultralytics import YOLO
from database import insert_event

  #Autoencoder
'''class Autoencoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Linear(5, 4),
            nn.ReLU(),

            nn.Linear(4, 2)
        )

        self.decoder = nn.Sequential(

            nn.Linear(2, 4),
            nn.ReLU(),

            nn.Linear(4, 5),
            nn.Sigmoid()
        )

    def forward(self, x):

        latent = self.encoder(x)

        reconstructed = self.decoder(latent)

        return reconstructed

# Load trained model
autoencoder = Autoencoder()

autoencoder.load_state_dict(
    torch.load("drone_autoencoder.pth")
)

autoencoder.eval()

scaler = joblib.load("scaler.pkl")
with open("threshold.txt", "r") as f:
    THRESHOLD = float(f.read())

print("Loaded Threshold:", THRESHOLD)
ai_alerted_ids = set()
'''
#LSTM Autoencoder
class LSTMAutoencoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.LSTM(
            input_size=4,
            hidden_size=16,
            batch_first=True
        )

        self.decoder = nn.LSTM(
            input_size=16,
            hidden_size=4,
            batch_first=True
        )

    def forward(self, x):

        encoded, (hidden, cell) = self.encoder(x)

        reconstructed, _ = self.decoder(encoded)

        return reconstructed
lstm_autoencoder = LSTMAutoencoder()

lstm_autoencoder.load_state_dict(
    torch.load("lstm_autoencoder.pth")
)

lstm_autoencoder.eval()
lstm_scaler = joblib.load(
    "lstm_scaler.pkl"
)
with open("threshold.txt", "r") as f:

    LSTM_THRESHOLD = float(f.read())

print("LSTM Threshold:", LSTM_THRESHOLD)
temporal_alerted_ids = set()
trajectory_buffers = {}

# ==============================
# POLYGON RESTRICTED ZONE FILE
# ==============================

ZONE_FILE = "zones.json"


def load_polygon_zones():

    if not os.path.exists(ZONE_FILE):
        return {}

    try:

        with open(ZONE_FILE, "r") as f:

            return json.load(f)

    except:

        return {}
    

# CAMERA SOURCES


rtsp_sources = [
    0,  # webcam
    1   # second webcam 
    #r"C:\ML_DL_Project\videos\drone2.mp4",
    ]


# Shared frames for Flask
frames      = [None] * len(rtsp_sources)
frame_locks = [threading.Lock() for _ in rtsp_sources]


# ALERT STORAGE

os.makedirs("alerts", exist_ok=True)

'''log_file = "alerts/event_log.csv"

if not os.path.exists(log_file):
    with open(log_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Drone_ID", "Event_Type"])
'''


# SAVE VIDEO USING FFMPEG
'''def save_video_opencv(frames_list, output_path, fps=10):

    if not frames_list:
        print("No frames to save")
        return

    h, w = frames_list[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame in frames_list:
        out.write(frame)

    out.release()

    print(f"Video saved using OpenCV: {output_path}")'''

def save_video_ffmpeg(frames_list, output_path, fps=10):

    if not frames_list:
        print(f"No frames to save for {output_path}", flush=True)
        return

    h, w = frames_list[0].shape[:2]
    print(f"Saving {len(frames_list)} frames at {fps:.1f} FPS -> {output_path}", flush=True)

    cmd = [
        'ffmpeg', '-y',
        '-f',       'rawvideo',
        '-vcodec',  'rawvideo',
        '-s',       f'{w}x{h}',
        '-pix_fmt', 'bgr24',
        '-r',       str(fps),
        '-i',       'pipe:0',
        '-vcodec',  'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset',  'fast',
        '-crf',     '23',
        output_path
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        for f in frames_list:
            proc.stdin.write(f.tobytes())
        proc.stdin.close()
        out, err = proc.communicate()
        if err:
            print(f"FFmpeg stderr: {err.decode()}", flush=True)
        print(f"Video saved: {output_path}", flush=True)

    except FileNotFoundError:
        print("FFmpeg not found! Install: winget install --id Gyan.FFmpeg", flush=True)
    except Exception as e:
        print(f"FFmpeg error: {e}", flush=True)




# ADD CAMERA DYNAMICALLY

def add_camera(url):
    cam_id = len(rtsp_sources)
    rtsp_sources.append(url)
    frames.append(None)
    frame_locks.append(threading.Lock())
    t = threading.Thread(
        target=process_camera,
        args=(cam_id, url),
        daemon=True
    )
    t.start()
    print(f"Camera {cam_id+1} added: {url}", flush=True)
    return cam_id



#Remove Camera
def remove_camera(cam_id):
    try:
        if cam_id >= len(rtsp_sources):
            return False

        print(f"Stopping Camera {cam_id+1}", flush=True)

        # Mark camera inactive
        rtsp_sources[cam_id] = None

        # Clear frame
        frames[cam_id] = None

        return True

    except Exception as e:
        print(f"Remove error: {e}", flush=True)
        return False

#autoecoder start

CSV_FILE = "drone_behavior_dataset.csv"

def save_features(row):

    file_exists = os.path.isfile(CSV_FILE)

    with open(CSV_FILE, mode='a', newline='') as f:

        writer = csv.writer(f)

        # Write header once
        if not file_exists:
            writer.writerow([
                "timestamp",
                "cam_id",
                "drone_id",
                "x",
                "y",
                "speed",
                "inside_zone",
                "hover_time"
            ])

        writer.writerow(row)

# CAMERA PROCESSING FUNCTION


def process_camera(cam_id, source):
    
    model = YOLO(r"C:\ML_DL_Project\models\best.pt")

    #cap = cv2.VideoCapture(source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Camera {cam_id+1} cannot open")
        return

    print(f"Camera {cam_id+1} started")

    track_history = defaultdict(lambda: deque(maxlen=60))
    hover_timer = defaultdict(float)
    alert_cooldown = defaultdict(float)

    recording = defaultdict(bool)
    frame_buffers = defaultdict(list)
    fps_during_recording = defaultdict(list)
    video_filenames = {}
    reported_drones      = set()
    
    last_seen_time = defaultdict(float)
    GRACE_TIME = 2  # seconds

    #restricted_zone = (300, 100, 600, 400)
    zones = load_polygon_zones()

    prev_time = time.time()
    
    

    while True:
        if rtsp_sources[cam_id] is None:
            print(f"Camera {cam_id+1} stopped", flush=True)
            break

        ret, frame = cap.read()
        #frame = cv2.flip(frame, 1)

        if not ret:
            
            print(
                f"Camera {cam_id+1}: frame lost"
            )

            time.sleep(0.1)

            continue

        frame = cv2.resize(frame, (960, 540))

        # ===================================
        # YOLO DETECTION + TRACKING
        # ===================================

        results = model.track(
            frame,
            conf=0.6,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )
        #print("Results is:- ", results)
        annotated_frame = results[0].plot()
        #print("annotated frame is :-", annotated_frame)

       # FPS
       

        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
        prev_time = current_time

        cv2.putText(
            annotated_frame,
            f"Cam {cam_id +1} | FPS: {fps:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        dt = 1 / fps if fps > 0 else 0.03

        # DRAW RESTRICTED ZONE

        '''zone_x1, zone_y1, zone_x2, zone_y2 = restricted_zone

        cv2.rectangle(
            annotated_frame,
            (zone_x1, zone_y1),
            (zone_x2, zone_y2),
            (0, 0, 255),
            2
        )
        cv2.putText(annotated_frame,"RESTRICTED ZONE",(zone_x1,zone_y1-8),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,255),2)
	    '''
        #DRAW CAMERA POLYGON
        zones = load_polygon_zones()
        polygon_points=zones.get(str(cam_id),[])
        polygon=None
        if len(polygon_points)>=3:
            polygon=np.array(polygon_points,np.int32)
            cv2.polylines(annotated_frame,[polygon],True,(0,0,255),3)
            top_y = min(point[1] for point in polygon)

            top_points = [point for point in polygon if point[1] == top_y]

            center_x = int(sum(point[0] for point in top_points) / len(top_points))

            text_x = center_x - 70
            text_y = max(top_y - 10, 20)

            cv2.putText(
                annotated_frame,
                "RESTRICTED ZONE",
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,0,255),
                2
            )
       # DRONE LOGIC
        

        if results[0].boxes.id is not None:

            for box, track_id in zip(results[0].boxes.xyxy,
                                     results[0].boxes.id):

                x1, y1, x2, y2 = map(int, box)
                
                track_id = int(track_id)
                # ── Log first detection of this drone ──
                if track_id not in reported_drones:
                    reported_drones.add(track_id)
                    print(f"Camera {cam_id+1}: Drone {track_id} first detected", flush=True)
                    
                    first_time_detection_video = f"alerts/first_time_detected_{track_id}_{int(time.time())}.mp4"

                    threading.Thread(
                        target=save_video_ffmpeg,
                        args=([annotated_frame.copy()], first_time_detection_video, 10),
                            daemon=True
                    ).start()
                    insert_event(
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        cam_id + 1,
                        track_id,
                        "Drone Detected",
                        first_time_detection_video
                    )
                current_time = time.time()
                last_seen_time[track_id] = current_time

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # trajectory
                track_history[track_id].append((center_x, center_y))

                for i in range(1, len(track_history[track_id])):
                    cv2.line(
                        annotated_frame,
                        track_history[track_id][i - 1],
                        track_history[track_id][i],
                        (255, 0, 0),
                        2
                    )

                # speed
                speed = 0

                if len(track_history[track_id]) >= 2:

                    x_prev, y_prev = track_history[track_id][-2]

                    distance = math.hypot(
                        center_x - x_prev,
                        center_y - y_prev
                    )

                    speed = distance / dt if dt > 0 else 0

                cv2.putText(
                    annotated_frame,
                    f"Speed: {speed:.1f}",
                    (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2
                )

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

               # HOVER DETECTION
                

                if speed < 5:
                    hover_timer[track_id] += dt
                else:
                    hover_timer[track_id] = 0

                if hover_timer[track_id] > 3 and current_time - alert_cooldown[track_id] > 5:

                    cv2.putText(
                        annotated_frame,
                        "HOVER ALERT!",
                        (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

                    #snapshot = f"alerts/hover_{track_id}_{int(time.time())}.jpg"
                    #cv2.imwrite(snapshot, annotated_frame)
                    hover_video = f"alerts/hover_{track_id}_{int(time.time())}.mp4"

                    threading.Thread(
                        target=save_video_ffmpeg,
                        args=([annotated_frame.copy()], hover_video, 10),
                            daemon=True
                    ).start()

                    insert_event(timestamp, cam_id+1, track_id,"Hovering Alert", hover_video)

                    

                    alert_cooldown[track_id] = current_time


                # RESTRICTED ZONE
                

                '''inside_zone = not (
                    x2 < zone_x1 or
                    x1 > zone_x2 or
                    y2 < zone_y1 or
                    y1 > zone_y2
                )'''
                inside_zone = False

                if polygon is not None:

                    bbox_polygon = np.array([
                        [x1, y1],
                        [x2, y1],
                        [x2, y2],
                        [x1, y2]
                    ], dtype=np.float32)

                    zone_polygon = cv2.convexHull(
                        polygon.astype(np.float32)
                    )

                    try:
                    

                        intersection_area, _ = cv2.intersectConvexConvex(
                            bbox_polygon,
                            zone_polygon
                        )

                        inside_zone = intersection_area > 0

                    except Exception as e:
                        print( f"Polygon Intersection Error: {e}" )

                        inside_zone = False
                row = [
                        timestamp,
                        cam_id+1,
                        track_id,
                        center_x,
                        center_y,
                        speed,
                        int(inside_zone),
                        hover_timer[track_id]
                    ]
                
                # =========================
                # AI ANOMALY DETECTION
                # =========================

                '''features = pd.DataFrame([{
                        "x": center_x,
                        "y": center_y,
                        "speed": speed,
                        "inside_zone": int(inside_zone),
                        "hover_time": hover_timer[track_id]
                    }])

                # Normalize
                features_scaled = scaler.transform(features)

                # Convert to tensor
                input_tensor = torch.FloatTensor(features_scaled)

                # Autoencoder prediction
                with torch.no_grad():

                    reconstructed = autoencoder(input_tensor)

                    reconstruction_error = torch.mean(
                        (input_tensor - reconstructed) ** 2
                    ).item()

                # Draw AI score
                cv2.putText(
                    annotated_frame,
                    f"AI Error: {reconstruction_error:.4f}",
                    (x1, y1 - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 255),
                    2
                )
                

                # Detect anomaly
                if reconstruction_error > THRESHOLD:

                    cv2.putText(
                        annotated_frame,
                        "AI ANOMALY DETECTED",
                        (x1, y1 - 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        3
                    )

                    # Prevent repeated alerts
                    if track_id not in ai_alerted_ids:

                        ai_alerted_ids.add(track_id)

                        print(
                            f"AI Anomaly Detected "
                            f"Drone {track_id} "
                            f"Error={reconstruction_error:.4f}"
                        )

                        insert_event(
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            cam_id + 1,
                            track_id,
                            "AI Behavioral Anomaly",
                            "none"
                        )
                    '''
                trajectory_feature = [
                    center_x,
                    center_y,
                    speed,
                    hover_timer[track_id]
                            ]
                # UNIQUE MULTI-CAMERA KEY
                buffer_key = (cam_id, track_id)
                #Create Buffer
                if buffer_key not in trajectory_buffers:
                    trajectory_buffers[buffer_key] = deque(maxlen=10)
                #Append Current Buffer    
                trajectory_buffers[buffer_key].append(trajectory_feature)
                    
                if len(trajectory_buffers[buffer_key]) == 10:
                    sequence = np.array(trajectory_buffers[buffer_key])
                    #Normalize
                    sequence_df = pd.DataFrame(
                                    sequence,
                                    columns=[
                                        "x",
                                        "y",
                                        "speed",
                                        "hover_time"
                                        ]
                                            )

                    sequence_scaled = lstm_scaler.transform(
                                            sequence_df
                                            )
                    #Add Batch Dimension
                    sequence_scaled = np.expand_dims( sequence_scaled, axis=0 )
                    #To Tensor
                    sequence_tensor = torch.FloatTensor( sequence_scaled )
                    #LSTM Inference
                    with torch.no_grad(): 
                        reconstructed = lstm_autoencoder( sequence_tensor ) 
                    reconstruction_error = torch.mean( (sequence_tensor - reconstructed) ** 2 ).item()
                    
                    # Draw AI score
                    cv2.putText(
                    annotated_frame,
                    f"LSTM Error: {reconstruction_error:.4f}",
                    (x1, y1 - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 255),
                    2
                    )
                    # Detect anomaly
                    if reconstruction_error > LSTM_THRESHOLD:

                        cv2.putText(
                        annotated_frame,
                        "TEMPORAL ANOMALY",
                        (x1, y1 - 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        3
                        )
                        temporal_trajectory_detection_video = f"alerts/temporal_anomaly_video_{track_id}_{int(time.time())}.mp4"
                        alert_key=(cam_id,track_id)
                        # Prevent repeated alerts
                        if alert_key not in temporal_alerted_ids:

                            temporal_alerted_ids.add(alert_key)

                            print(
                                f"Temporal Anomaly Detected "
                                f"Drone {track_id} "
                                f"Error={reconstruction_error:.4f}"
                            )
                            threading.Thread(
                            target=save_video_ffmpeg,
                            args=([annotated_frame.copy()], temporal_trajectory_detection_video, 10),
                            daemon=True
                            ).start()

                            insert_event(
                                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                cam_id + 1,
                                track_id,
                                "Temporal Trajectory Anomaly",
                                temporal_trajectory_detection_video
                            )
                        
 
                

    

                
                if not inside_zone and hover_timer[track_id] < 3:
                    save_features(row)
                    
                    print("Saving features...", flush=True)
                    print(row, flush=True)
                #print(int(inside_zone))

                # ZONE ENTRY
                if inside_zone and not recording[track_id]:
                    print(f"Camera {cam_id+1}: Drone {track_id} entered zone", flush=True)
                    recording[track_id]            = True
                    frame_buffers[track_id]        = []
                    fps_during_recording[track_id] = []
                    mp4_filename = f"alerts/cam{cam_id+1}_drone_{track_id}_{int(time.time())}.mp4"
                    video_filenames[track_id]      = mp4_filename
                    threading.Thread(
                        target=winsound.Beep,
                        args=(1500, 300),
                        daemon=True
                    ).start()
                    insert_event(timestamp, cam_id+1, track_id,
                                 "Restricted Zone Entry", mp4_filename)

                # BUFFER FRAMES
                if recording[track_id]:
                    frame_buffers[track_id].append(annotated_frame.copy())
                    fps_during_recording[track_id].append(fps)

                # ZONE EXIT
                if not inside_zone and recording[track_id]:
                    print(f"Camera {cam_id+1}: Drone {track_id} exited zone", flush=True)
                    recording[track_id] = False
                    mp4_path        = video_filenames.get(track_id, "")
                    buffered_frames = frame_buffers.pop(track_id, [])
                    fps_values      = fps_during_recording.pop(track_id, [10])
                    avg_fps         = sum(fps_values) / len(fps_values) if fps_values else 10
                    print(f"Avg FPS during recording: {avg_fps:.1f}", flush=True)
                    print(f"Total frames buffered: {len(buffered_frames)}", flush=True)
                    threading.Thread(
                        target=save_video_ffmpeg,
                        args=(buffered_frames, mp4_path, avg_fps),
                        daemon=True
                    ).start()
                    insert_event(timestamp, cam_id+1, track_id,
                                 "Restricted Zone Exit", mp4_path)
                '''if recording[track_id]:

                    time_since_seen = current_time - last_seen_time[track_id]

                    if not inside_zone and time_since_seen > GRACE_TIME:

                        print(f"Camera {cam_id+1}: Drone {track_id} exited zone", flush=True)

                        recording[track_id] = False

                        mp4_path        = video_filenames.get(track_id, "")
                        buffered_frames = frame_buffers.pop(track_id, [])
                        fps_values      = fps_during_recording.pop(track_id, [10])
                        avg_fps         = sum(fps_values) / len(fps_values) if fps_values else 10

                        if len(buffered_frames) > 5:
                            threading.Thread(
                            target=save_video_ffmpeg,
                            args=(buffered_frames, mp4_path, avg_fps),
                            daemon=True
                            ).start()

                            insert_event(timestamp, cam_id+1, track_id,
                                 "Restricted Zone Exit", mp4_path)'''                 

                    # HANDLE LOST TRACKING
                current_time = time.time()

                for tid in list(recording.keys()):

                    if recording[tid]:

                        time_since_seen = current_time - last_seen_time[tid]

                        if time_since_seen > GRACE_TIME:

                            print(f"Camera {cam_id+1}: Drone {tid} lost/exited", flush=True)

                            recording[tid] = False
                            
                            '''if tid in ai_alerted_ids:
                                ai_alerted_ids.remove(tid)'''
                            buffer_key=(cam_id,track_id)      
                            if buffer_key in trajectory_buffers:
                                 del trajectory_buffers[buffer_key]
                            alert_key=(cam_id,track_id)     
                            if alert_key in temporal_alerted_ids:
                                temporal_alerted_ids.remove(alert_key)
                            mp4_path        = video_filenames.get(tid, "")
                            buffered_frames = frame_buffers.pop(tid, [])
                            fps_values      = fps_during_recording.pop(tid, [10])
                            avg_fps         = sum(fps_values) / len(fps_values) if fps_values else 10

                            if len(buffered_frames) > 5:

                                threading.Thread(
                                target=save_video_ffmpeg,
                                args=(buffered_frames, mp4_path, avg_fps),
                                daemon=True
                                ).start()

                                insert_event(timestamp, cam_id+1, tid, "Restricted Zone Exit", mp4_path)
        # SHARE FRAME WITH FLASK
        

        with frame_locks[cam_id]:
            frames[cam_id] = annotated_frame

    '''cv2.imshow(f"Camera {cam_id}", annotated_frame)

        if cv2.waitKey(1) == 27:
            break'''


# START ALL CAMERAS


def process_cameras():

    threads = []

    for cam_id, source in enumerate(rtsp_sources):

        t = threading.Thread(
            target=process_camera,
            args=(cam_id, source),
            daemon=True
        )

        t.start()
        threads.append(t)

    for t in threads:
        t.join()


# RUN DIRECTLY


if __name__ == "__main__":

    process_cameras()