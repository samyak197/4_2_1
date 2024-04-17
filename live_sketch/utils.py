import cv2
import os
import cv2


import cv2


def download_photo():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        ret, mask = cv2.threshold(edges, 70, 255, cv2.THRESH_BINARY_INV)
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        filename = "livesketch/live_sketch_photo.jpg"
        cv2.imwrite(filename, frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n\r\n"
        )
        yield filename  # Yield the filename after saving the photo
    else:
        yield None  # Yield None if capturing fails


def generate2():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 50, 150)
            ret, mask = cv2.threshold(edges, 70, 255, cv2.THRESH_BINARY_INV)
            _, buffer = cv2.imencode(".jpg", mask)
            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n\r\n"
            )
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
