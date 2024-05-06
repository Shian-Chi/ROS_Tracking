import cv2
import os
import time


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d !"
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def show_camera(record_video=False):
    window_title = "CSI Camera"
    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if video_capture.isOpened():
        try:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = None
            recording = False

            while True:
                ret_val, frame = video_capture.read()
                #cv2.imshow(window_title, frame)

                if record_video:
                    if not recording:
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        output_file = f"{timestamp}.avi"
                        out = cv2.VideoWriter(output_file, fourcc, 30.0, (frame.shape[1], frame.shape[0]))
                        recording = True
                        print(f"Start recording to file {output_file}")
                    out.write(frame)
                else:
                    if recording:
                        out.release()
                        recording = False
                        print("Stop recording")

                if cv2.waitKey(1) == ord("q"):
                    break

            if recording:
                out.release()
                print("Stop recording")

        finally:
            video_capture.release()
            cv2.destroyAllWindows()

    else:
        print("Error: Unable to open camera")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--record", help="record video", action="store_true")
    args = parser.parse_args()

    show_camera(record_video=args.record)