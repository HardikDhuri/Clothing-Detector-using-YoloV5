import cv2
import os
import pandas as pd
import numpy as np
import glob
import shutil
from yolov5.detect import run

# Yolov5 arguments
weights = "yolov5/runs/train/yolov5s_results/weights/last.pt"  # path to yolov5 weights
source = ""
img = 416
conf = 0.6
name = ""
save_txt = True
save_conf = True
save_crop = True


def get_frames(path, skip_frames, save_frames=False):
    """
    Input:
    path -> Path to the video file
    skip_frame -> Number of frames to skip (int)
    --------
    Output:
    dir_name -> Name of the folder where frames are extracted
    """
    cap = cv2.VideoCapture(path)
    i = 0

    # a variable to keep track of the frame to be saved
    frame_count = 0
    dir_name = path.split("/")[-1].split(".")[0]
    # Opens the inbuilt camera of laptop to capture video.
    cap = cv2.VideoCapture(path)

    # a condition to check if folder already exists or not
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i > skip_frames - 1:
            frame_count += 1
            if save_frames:
                frame = cv2.resize(frame, (416, 416))
                cv2.imwrite(f'./{dir_name}/{str(frame_count)}.jpg', frame)
            i = 0
            continue
        i += 1

    cap.release()
    cv2.destroyAllWindows()

    if save_frames:
        return f"./{dir_name}"
    else:
        return frame_count


def calculate_fps_of_video(path, time_in_secs):
    """
    Input:
    path -> Path to the video file
    time_in_secs -> duration of the video in seconds (int) (round the value to the closest integer)
    --------
    Output:
    fps -> frames per second video was extracted (int)
    """
    frames = int(get_frames(path, 0))
    fps = np.round(frames / time_in_secs)
    return int(fps)


# Parses through the labels of predicted frames and adds them to pandas DataFrame
def get_outputs(run, save_csv=False):
    """
    Input:
    run -> Path to the folder inside the yolov5/runs/detect/
    save_csv -> if you want the df to be saved inside the current dir (boolean)
    --------
    Output:
    df -> Data frame with all the detected objects (pd.DataFrame)
    """
    class_names = ["Car", "Clothing", "Food", "Motorcycle"]
    cols = ['frame', 'confidence', 'category', "bbox"]
    df = pd.DataFrame(columns=cols)
    path = f'yolov5/runs/detect/{run}'
    frames = len(glob.glob(f'{path}/*.jpg'))
    for i in range(frames):
        try:
            with open(f"{path}/labels/{i}.txt") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.split(" ")
                    class_name = class_names[int(line[0])]
                    confidence = line[1]
                    bbox = " ".join(line[2:])
                    temp_df = pd.DataFrame([[i, confidence, class_name, bbox]], columns=cols)
                    df = df.append(temp_df)
        except FileNotFoundError:
            continue
    df = df.set_index(["frame"])
    df = df.reset_index()
    if save_csv:
        df.to_csv("./output.csv")
    return df


def get_final(run_folder, name, main_frames):
    """
    Input:
    run -> Path to the folder inside the yolov5/runs/detect/
    name -> name of the folder in the final cropped images of a video (boolean)
    --------
    Output:
    saves cropped images of the 5 frames with high confidence score in a folder named "name"
    Path: ./final/name
    """
    if not os.path.exists("./final"):
        os.mkdir("final")
    if not os.path.exists(f"final/{name}"):
        os.mkdir(f"final/{name}")
    if not os.path.exists(f"final/{name}/frames"):
        os.mkdir(f"final/{name}/frames")
    if not os.path.exists(f"final/{name}/cropped"):
        os.mkdir(f"final/{name}/cropped")

    path = f"./final/{name}"

    df = get_outputs(run_folder)
    final = df.sort_values(["confidence"], ascending=False)[:5]
    frames = list(final.frame)
    for frame in frames:
        print(frame)
        shutil.copy(f"{main_frames}/{frame}.jpg", f"{path}/frames/{frame}.jpg")
        cropped_paths = glob.glob(f"yolov5/runs/detect/{run_folder}/crops/Clothing/{frame}.jpg") + glob.glob(
            f"yolov5/runs/detect/{run_folder}/crops/Clothing/{frame}[0-9].jpg")
        for cropped_path in cropped_paths:
            dest = f"{path}/cropped/image{frame}.jpg"
            shutil.copy(cropped_path, dest)
    final.to_csv(f"final/{name}/output.csv")
    print(f"Saved at final/{name}")


def run_detection(weights=weights, source=source, img=img, conf=conf, name=name, save_txt=save_txt, save_conf=save_conf,
                  save_crop=save_crop):
    run(weights=weights, source=source, imgsz=img, conf_thres=conf, name=name, save_txt=save_txt, save_conf=save_conf,
        save_crop=save_crop)
