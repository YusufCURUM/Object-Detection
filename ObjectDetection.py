import asyncio
import websockets
import base64
import cv2
import torch
from Tracker import *
import numpy as np
from threading import Thread
import dependencies.pafy as pafy

model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

# green area
area_1 = [(30, 40), (30, 460), (620, 460), (620, 40)]


def points(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)


async def send_data(websocket, video1_path):
    counter1 = set()
    tracker1 = Tracker()

    cap1 = cv2.VideoCapture(video1_path)
    objects_to_count = ["person", "car"]

    async def send_video(cap, counter, tracker):
        while True:
            # Read an image from the camera
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 480))
            cv2.polylines(frame, [np.array(area_1, np.int32)], True, (0, 255, 255), 3)
            results = model(frame)
            list = []

            for index, row in results.pandas().xyxy[0].iterrows():
                x1 = int(row["xmin"])
                y1 = int(row["ymin"])
                x2 = int(row["xmax"])
                y2 = int(row["ymax"])
                obj_name = row["name"]
                if obj_name in objects_to_count:
                    list.append([x1, y1, x2, y2])
            boxes_ids = tracker.update(list)
            for box_id in boxes_ids:
                x, y, w, h, id = box_id
                cv2.rectangle(frame, (x, y), (w, h), (255, 0, 255), 2)
                cv2.putText(
                    frame, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2
                )
                result = cv2.pointPolygonTest(
                    np.array(area_1, np.int32), (int(w), int(h)), False
                )
                if result > 0:
                    counter.add(id)
            p = len(counter)
            cv2.putText(
                frame, str(p), (20, 30), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 2
            )

            # Convert the image to a JPEG byte string
            ret, buffer = cv2.imencode(".jpg", frame)
            jpg_as_text = base64.b64encode(buffer).decode("utf-8")
            # Send the image and the p value over the websocket
            message = f"{p},{jpg_as_text}"
            await websocket.send(message)
            # Wait for some time before sending the next image
            await asyncio.sleep(0.01)

        cap.release()
        cv2.destroyAllWindows()

    # Run the two video capture loops asynchronously
    await asyncio.gather(send_video(cap1, counter1, tracker1))


async def start_server(video_path, port):
    # Create a new VideoCapture object
    if "youtube" in video_path:
        video = pafy.new(video_path)
        best = video.getbest()
        video_path = best.url
    async with websockets.serve(
        lambda websocket: send_data(websocket, video1_path=video_path),
        "127.0.0.1",
        port,
    ):
        await asyncio.Future()


# Create two threads for running the servers


thread1 = Thread(target=lambda: asyncio.run(start_server("walking.mp4", 8765)))
thread2 = Thread(target=lambda: asyncio.run(start_server("vb.m4v", 8766)))


if __name__ == "__main__":
    # Start the threads
    thread1.start()
    thread2.start()
    # Wait for the threads to complete
    thread1.join()
    thread2.join()
