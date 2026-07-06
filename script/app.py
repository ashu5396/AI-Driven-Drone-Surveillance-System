import time
import base64
from flask import Flask, render_template, send_file, request
from flask_socketio import SocketIO
import cv2
import drone_detection
import threading
import os

from database import init_db, get_events
import json

ZONE_FILE = "zones.json"

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")


# ==============================
# ROUTES
# ==============================


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/alerts')
def alerts():
    data = get_events()
    return {"events": data}


@app.route('/video_play/<path:filename>')
def video_play(filename):
    full_path = os.path.abspath(filename)

    if not os.path.exists(full_path):
        return "File not Found", 404

    print("Playing video:", full_path)
    return send_file(full_path, mimetype='video/mp4')


@app.route('/get_cameras')
def get_cameras():
    cameras = []

    for i, src in enumerate(drone_detection.rtsp_sources):
        if src is not None:
            cameras.append({"id": i})

    return {"cameras": cameras}


@app.route('/add_camera', methods=['POST'])
def add_camera():
    data = request.get_json()
    url = data.get('url', '').strip()

    if not url:
        return {"success": False, "message": "Empty URL"}

    cam_id = drone_detection.add_camera(url)

    return {"success": True, "cam_id": cam_id}


@app.route('/remove_camera', methods=['POST'])
def remove_camera():
    data = request.get_json()
    cam_id = data.get('cam_id')

    success = drone_detection.remove_camera(cam_id)

    return {"success": success}


# ==============================
# POLYGON RESTRICTED ZONE SAVE API
# ==============================

@app.route('/save_zone', methods=['POST'])
def save_zone():

    data = request.get_json()

    cam_id = str(data['cam_id'])

    points = data['points']

    zones = {}

    if os.path.exists(ZONE_FILE):

        with open(ZONE_FILE, 'r') as f:
            zones = json.load(f)

    zones[cam_id] = points

    with open(ZONE_FILE, 'w') as f:
        json.dump(zones, f, indent=4)

    return {"success": True}


# ==============================
# LOAD SAVED POLYGON
# ==============================

@app.route('/get_zone/<int:cam_id>')
def get_zone(cam_id):

    if not os.path.exists(ZONE_FILE):

        return {"points": []}

    with open(ZONE_FILE, 'r') as f:
        zones = json.load(f)

    return {
        "points": zones.get(str(cam_id), [])
    }

# ==============================
# WEBSOCKET FRAME STREAM
# ==============================

'''def send_frames():
    while True:

        for cam_id in range(len(drone_detection.frames)):

            with drone_detection.frame_locks[cam_id]:
                frame = drone_detection.frames[cam_id]

            if frame is None:
                continue

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            encoded = base64.b64encode(buffer).decode('utf-8')

            socketio.emit('video_frame', {
                'cam_id': cam_id,
                'frame': encoded
            })

        socketio.sleep(0.05)   # control FPS
'''
def send_frames():
    while True:
        for cam_id in range(len(drone_detection.frames)):
            try:
                if cam_id >= len(drone_detection.frame_locks):
                    continue

                with drone_detection.frame_locks[cam_id]:
                    frame = drone_detection.frames[cam_id]

                if frame is None:
                    continue

                ret, buffer = cv2.imencode('.jpg', frame,
                                          [cv2.IMWRITE_JPEG_QUALITY, 50])
                if not ret:
                    continue

                encoded = base64.b64encode(buffer).decode('utf-8')

                socketio.emit('video_frame', {
                    'cam_id': cam_id,
                    'frame': encoded
                })

            except Exception as e:
                print(f"Frame send error cam {cam_id}: {e}", flush=True)

        socketio.sleep(0.1)   # 10 FPS per camera — reduced from 20

# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    init_db()

    # Start detection threads
    t = threading.Thread(target=drone_detection.process_cameras)
    t.daemon = True
    t.start()

    # Start websocket streaming
    socketio.start_background_task(send_frames)

    socketio.run(app, host="0.0.0.0", port=5000)