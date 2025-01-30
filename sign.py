import speech_recognition as sr
import streamlit.components.v1 as components
from streamlit_option_menu import option_menu
import requests
import numpy as np
import cv2
import mediapipe as mp
import copy
import itertools
import keras
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import string
import streamlit as st
import os
import time
import random
import threading

# List of lowercase alphabets
arr = list(string.ascii_lowercase)


def text_detection():
    # List of available ISL GIFs
    isl_gif = ['any questions', 'are you angry', 'are you busy', 'are you hungry', 'are you sick', 'be careful',
               'can we meet tomorrow', 'did you book tickets', 'did you finish homework', 'do you go to office', 'do you have money',
               'do you want something to drink', 'do you want tea or coffee', 'do you watch TV', 'dont worry', 'flower is beautiful',
               'good afternoon', 'good evening', 'good morning', 'good night', 'good question', 'had your lunch', 'happy journey',
               'hello what is your name', 'how many people are there in your family', 'i am a clerk', 'i am bore doing nothing', 
               'i am fine', 'i am sorry', 'i am thinking', 'i am tired', 'i dont understand anything', 'i go to a theatre', 'i love to shop',
               'i had to say something but i forgot', 'i have headache', 'i like pink colour', 'i live in nagpur', 'lets go for lunch', 'my mother is a homemaker',
               'my name is john', 'nice to meet you', 'no smoking please', 'open the door', 'please call me later',
               'please clean the room', 'please give me your pen', 'please use dustbin dont throw garbage', 'please wait for sometime', 'shall I help you',
               'shall we go together tommorow', 'sign language interpreter', 'sit down', 'stand up', 'take care', 'there was traffic jam', 'wait I am thinking',
               'what are you doing', 'what is the problem', 'what is todays date', 'what is your father do', 'what is your job',
               'what is your mobile number', 'what is your name', 'whats up', 'when is your interview', 'when we will go', 'where do you stay',
               'where is the bathroom', 'where is the police station', 'you are wrong', 'address', 'agra', 'ahemdabad', 'all', 'april', 'assam', 'august', 'australia', 'badoda', 'banana', 'banaras', 'banglore',
               'bihar', 'bridge', 'cat', 'chandigarh', 'chennai', 'christmas', 'church', 'clinic', 'coconut', 'crocodile', 'dasara',
               'deaf', 'december', 'deer', 'delhi', 'dollar', 'duck', 'febuary', 'friday', 'fruits', 'glass', 'grapes', 'gujrat', 'hello',
               'hindu', 'hyderabad', 'india', 'january', 'jesus', 'job', 'july', 'karnataka', 'kerala', 'krishna', 'litre', 'mango',
               'may', 'mile', 'monday', 'mumbai', 'museum', 'muslim', 'nagpur', 'october', 'orange', 'pakistan', 'pass', 'police station',
               'post office', 'pune', 'punjab', 'rajasthan', 'ram', 'restaurant', 'saturday', 'september', 'shop', 'sleep', 'southafrica',
               'story', 'sunday', 'tamil nadu', 'temperature', 'temple', 'thursday', 'toilet', 'tomato', 'town', 'tuesday', 'usa', 'village',
               'voice', 'wednesday', 'weight', 'please wait for sometime', 'what is your mobile number', 'what are you doing', 'are you busy']

    # User input
    text_input = st.text_input("Enter the text command:")
    if text_input:
        a = text_input.lower()
        st.write("You entered: " + a)

        # Remove punctuation
        for c in string.punctuation:
            a = a.replace(c, "")

        # Goodbye command
        if a in ['goodbye', 'good bye', 'bye']:
            st.write("Oops! Time to say goodbye")
            st.stop()
            
        image_list = []
        letters_dir = r"letters"
        for char in a:
            if char in arr:
                img_path = os.path.join(letters_dir, f"{char}.jpg")
                image_list.append(img_path)

        # Display images using PIL and Streamlit
        if image_list:
            if 'index' not in st.session_state:
                st.session_state.index = 0

            # Create a placeholder for displaying images
            placeholder = st.empty()

            while st.session_state.index < len(image_list):
                img_path = image_list[st.session_state.index]
                try:
                    # Load the image using PIL
                    image = Image.open(img_path)
                    placeholder.image(image)
                except Exception as e:
                    st.write(f"Error displaying image: {e}")
                
                st.session_state.index += 1
                time.sleep(1)  # Delay for 1 second before showing the next image

            # Reset the index for the next input
            st.session_state.index = 0

def voice_detection():
    isl_gif = ['any questions', 'are you angry', 'are you busy', 'are you hungry', 'are you sick', 'be careful',
               'can we meet tomorrow', 'did you book tickets', 'did you finish homework', 'do you go to office', 'do you have money',
               'do you want something to drink', 'do you want tea or coffee', 'do you watch TV', 'dont worry', 'flower is beautiful',
               'good afternoon', 'good evening', 'good morning', 'good night', 'good question', 'had your lunch', 'happy journey',
               'hello what is your name', 'how many people are there in your family', 'i am a clerk', 'i am bore doing nothing', 
               'i am fine', 'i am sorry', 'i am thinking', 'i am tired', 'i dont understand anything', 'i go to a theatre', 'i love to shop',
               'i had to say something but i forgot', 'i have headache', 'i like pink colour', 'i live in nagpur', 'lets go for lunch', 'my mother is a homemaker',
               'my name is john', 'nice to meet you', 'no smoking please', 'open the door', 'please call me later',
               'please clean the room', 'please give me your pen', 'please use dustbin dont throw garbage', 'please wait for sometime', 'shall I help you',
               'shall we go together tommorow', 'sign language interpreter', 'sit down', 'stand up', 'take care', 'there was traffic jam', 'wait I am thinking',
               'what are you doing', 'what is the problem', 'what is todays date', 'what is your father do', 'what is your job',
               'what is your mobile number', 'what is your name', 'whats up', 'when is your interview', 'when we will go', 'where do you stay',
               'where is the bathroom', 'where is the police station', 'you are wrong', 'address', 'agra', 'ahemdabad', 'all', 'april', 'assam', 'august', 'australia',
               'badoda', 'banana', 'banaras', 'banglore', 'bihar', 'bridge', 'cat', 'chandigarh', 'chennai', 'christmas', 'church', 'clinic', 'coconut', 'crocodile', 
               'dasara', 'deaf', 'december', 'deer', 'delhi', 'dollar', 'duck', 'febuary', 'friday', 'fruits', 'glass', 'grapes', 'gujrat', 'hello', 'hindu', 
               'hyderabad', 'india', 'january', 'jesus', 'job', 'july', 'karnataka', 'kerala', 'krishna', 'litre', 'mango', 'may', 'mile', 'monday', 'mumbai', 
               'museum', 'muslim', 'nagpur', 'october', 'orange', 'pakistan', 'pass', 'police station', 'post office', 'pune', 'punjab', 'rajasthan', 'ram', 
               'restaurant', 'saturday', 'september', 'shop', 'sleep', 'southafrica', 'story', 'sunday', 'tamil nadu', 'temperature', 'temple', 'thursday', 'toilet', 
               'tomato', 'town', 'tuesday', 'usa', 'village', 'voice', 'wednesday', 'weight']

    arr = list(string.ascii_lowercase)
    letters_dir = r"letters"
    gifs_dir = r"ISL_Gifs"
    # Initialize recognizer
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening... Please speak.")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

        try:
            command = recognizer.recognize_google(audio).lower()
            st.write("You said: " + command)

            # Remove punctuation from the command
            command = ''.join(char for char in command if char not in string.punctuation)

            # Goodbye command
            if command in ['goodbye', 'good bye', 'bye']:
                st.write("Oops! Time to say goodbye")
                return

            # Check if command matches any ISL GIF
            if command in isl_gif:
                gif_path = os.path.join(gifs_dir, f"{command}.gif")
                try:
                    st.image(gif_path)
                except Exception as e:
                    st.write(f"Error loading GIF: {e}")
            else:
                # Convert command to individual letters
                image_list = [os.path.join(letters_dir, f"{char}.jpg") for char in command if char in arr]

                if image_list:
                    if 'index' not in st.session_state:
                        st.session_state.index = 0

                    # Display each image with a delay
                    if st.session_state.index < len(image_list):
                        img_path = image_list[st.session_state.index]
                        try:
                            image = Image.open(img_path)
                            st.image(image)
                            st.session_state.index += 1
                            time.sleep(1)
                            st.experimental_rerun()
                        except Exception as e:
                            st.write(f"Error displaying image: {e}")
                    else:
                        st.session_state.index = 0  # Reset for the next command
        except sr.UnknownValueError:
            st.write("Sorry, I could not understand the audio.")
        except sr.RequestError as e:
            st.write(f"Error with the speech recognition service: {e}")


def WebcamDetection():
    # Load the trained model
    st.title("Sign Language Detection using Webcam")
    model = tf.keras.models.load_model(r"model (1).h5")

    # Initialize Mediapipe utilities
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands

    # Define the alphabet for classification labels
    alphabet = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    alphabet += list(string.ascii_uppercase)

    # Function to calculate the landmark list
    def calc_landmark_list(image, landmarks):
        image_width, image_height = image.shape[1], image.shape[0]
        landmark_point = []

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            landmark_point.append([landmark_x, landmark_y])

        return landmark_point

    # Function to preprocess landmarks
    def pre_process_landmark(landmark_list):
        temp_landmark_list = copy.deepcopy(landmark_list)

        base_x, base_y = 0, 0
        for index, landmark_point in enumerate(temp_landmark_list):
            if index == 0:
                base_x, base_y = landmark_point[0], landmark_point[1]

            temp_landmark_list[index][0] -= base_x
            temp_landmark_list[index][1] -= base_y

        temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
        max_value = max(map(abs, temp_landmark_list))

        temp_landmark_list = [n / max_value for n in temp_landmark_list]

        return temp_landmark_list

    # Initialize webcam capture
    cap = cv2.VideoCapture(0)
    stframe = st.empty()  # Placeholder for Streamlit image display
    stop = False

    # Add start and stop buttons
    start_button = st.button("Start Webcam")
    stop_button = st.button("Stop Webcam")

    if start_button:
        stop = False

    if stop_button:
        stop = True

    # Initialize Mediapipe Hands
    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

        while not stop and cap.isOpened():
            success, image = cap.read()
            image = cv2.flip(image, 1)

            if not success:
                st.warning("Ignoring empty camera frame.")
                continue

            # Preprocess the image for hand detection
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            # Draw hand annotations and make predictions
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            debug_image = copy.deepcopy(image)

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)

                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                    # Prepare data for prediction
                    df = pd.DataFrame(pre_processed_landmark_list).transpose()
                    predictions = model.predict(df, verbose=0)
                    predicted_classes = np.argmax(predictions, axis=1)
                    label = alphabet[predicted_classes[0]]

                    # Display the predicted label on the image
                    cv2.putText(image, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                    st.write(f"Detected Sign: {label}")

            # Stream the output image in Streamlit
            stframe.image(image, channels="BGR", caption="Webcam Stream")

            # Stop the loop if the "Stop Webcam" button is clicked
            if stop:
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# About Section function
def about_section():
    """Function to display the About section content.""" 
    st.markdown("<h2 style='text-align: center; color: #4CAF50;'>About the Hearing Impairment Assistance System</h2>", unsafe_allow_html=True)
    st.write(""" 
        This application is designed to assist individuals with hearing impairments by translating text and voice commands into sign language. 
        The system leverages advanced machine learning and natural language processing techniques to provide an accessible communication platform for users.
    """)

    # Features of the System
    st.markdown("### 🌟 Key Features:")
    st.markdown(""" 
    - *Text-to-Sign Language Translation*: Converts user-inputted text into corresponding sign language gestures.
    - *Voice-to-Sign Language Translation*: Listens to voice commands and translates them into sign language using a speech recognition module.
    - *Real-time Feedback*: Provides immediate feedback through visual representation of sign language gestures using GIFs and images.
    - *User-Friendly Interface*: Built with a clean and accessible design using Streamlit, ensuring ease of use for all users.
    """)

    # Technology Stack
    st.markdown("### 🔧 Technology Stack:")
    st.markdown(""" 
    - *Python*: Core programming language used for development.
    - *Streamlit*: Framework for creating an interactive web application.
    - *SpeechRecognition*: Python library used for converting speech to text.
    - *PIL (Python Imaging Library)*: Used for handling and displaying images.
    - *Machine Learning*: Employed for voice and text recognition tasks.
    """)

    # Use Cases
    st.markdown("### 🤝 Use Cases:")
    st.markdown(""" 
    - *Assisting in Communication*: Helps hearing-impaired individuals to communicate more effectively with non-sign language users.
    - *Educational Tool*: Useful for learning and teaching sign language.
    - *Public Service Accessibility*: Can be used in public service sectors like hospitals and customer service centers to assist individuals with hearing impairments.
    """)

    # Social Impact
    st.markdown("### 🌍 Social Impact:")
    st.write(""" 
        This project aims to bridge the communication gap for individuals with hearing impairments. By providing a reliable translation system,
        we hope to foster inclusivity and make everyday interactions easier and more efficient for those who rely on sign language.
    """)

    # Contact Information
    st.markdown("### 📭 Contact Us:")
    st.write("For more information or feedback, please reach out to the development team at: [YourEmail@example.com](mailto:YourEmail@example.com)")

# Main Streamlit App
st.title("Indian Sign Language (ISL)")
menu = ["Home", "Text Detection", "Voice Detection", "WebCam Detection", "About"]
with st.sidebar:
    choice = option_menu("Menu", menu, icons=['house', 'road', 'binoculars', 'info-circle'], menu_icon="cast", default_index=0, orientation="vertical")

if choice == "Home":
    # Home Section Content
    st.title("Welcome to the Hearing Impairment Assistant")
    st.write("The Indian Sign Language (ISL) application is designed to facilitate communication for the deaf community. It translates live speech or audio recordings into text and displays corresponding Indian Sign Language images or GIFs.")
    st.image("isl.jpg", use_container_width=True)    
    st.write("Choose a functionality from the sidebar to start.")
elif choice == "Text Detection":
    text_detection()
elif choice == "Voice Detection":
    voice_detection()
elif choice == "WebCam Detection":
    WebcamDetection()
elif choice == "About":
    about_section()