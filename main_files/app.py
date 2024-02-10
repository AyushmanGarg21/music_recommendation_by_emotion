import numpy as np
import cv2
import av
import webbrowser
import streamlit as st
import scorecal
import time
from keras.models import model_from_json
from keras.utils import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

emotion_dict = {0:'angry', 1 :'fear', 2: 'happy', 3:'neutral', 4: 'sad', 5: 'surprise'}
json_file = open('..\Emotion_Models\emotion_model_fernet.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)
classifier.load_weights("..\Emotion_Models\emotion_model_fernet.h5")
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

try:
    emotion = np.load("emotion.npy")
    songs = np.load("song_rec.npy")
except Exception:
    emotion = ""
    songs = ""
     
predicted_em = []
all_em_values = []
class VideoTransformer(VideoTransformerBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                all_em_values.append(prediction)
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)
                predicted_em.append(output)
            label_position = (x, y)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            np.save("emotion.npy", np.array(predicted_em))
            np.save("emotion_all.npy", np.array(all_em_values))
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.title("Real Time Face Emotion Detection and Multimedia recommendation")
    activiteis = ["Home", "Webcam","recommend song videos"]
    choice = st.sidebar.selectbox("Select Activity", activiteis)
    if choice == "Home":
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                            <h2 style="color:white;text-align:center;">
                                            HOME.</h2>
                                            <h4 style="color:white;text-align:center;">
                                            Refresh for reset the all emotion</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        btn = st.button("Refresh")
        if btn:
            np.save("emotion.npy", np.array([""]))
            np.save("emotion_all.npy", np.array([""]))
            np.save("song_rec.npy", np.array([""]))

    elif choice == "Webcam":
        st.header("Webcam Live Feed")
        st.write("Click on start")
        webrtc_streamer(key="example", video_processor_factory=VideoTransformer,sendback_audio=False)

    elif choice == "recommend song videos":
         btn = st.button("click")
         if btn:
            if len(emotion)<2:
                st.warning("Capture your emotion first")
            else:
                scorecal.setSongs()
                time.sleep(1)
                songs  = np.load("song_rec.npy")
                ri = np.random.randint(0, len(songs)-1)
                rec_song = songs[ri]
                webbrowser.open(f"https://www.youtube.com/results?search_query={rec_song}")  
    else:
        np.save("emotion.npy", np.array([""]))
        np.save("emotion_all.npy", np.array([""]))
        np.save("song_rec.npy", np.array([""]))
        pass


if __name__ == "__main__":
    main()
