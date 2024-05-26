import cv2
import mediapipe as mp

import tempfile
import streamlit as st

#mediapipe inbuilt solutions 
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def main():

    st.title('Face Detection App')

    st.markdown(' ## Output')
    stframe = st.empty()
    
    tfflie = tempfile.NamedTemporaryFile(delete=False)

    vid = cv2.VideoCapture(0)
    

    # width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = int(vid.get(cv2.CAP_PROP_FPS))
    # codec = cv2.VideoWriter_fourcc('V','P','0','9')


    # st.sidebar.text('Input Video')
    # st.sidebar.video(tfflie.name)


    with mp_face_detection.FaceDetection() as face_detection:
        
        while vid.isOpened():

            ret, image = vid.read()

            if not ret:
                break
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image)

            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(image, detection)
            stframe.image(image,use_column_width=True)

        vid.release()
        cv2.destroyAllWindows()

    st.success('Video is Processed')
    st.stop()

if __name__ == '__main__':
    main()
