import streamlit as st
import openai
import speech_recognition as sr
import os


openai.api_key = os.environ.get('OPENAI_API_KEY')


# Initialize Streamlit
st.title("This is Celeste's amazing Chatbot")

def transcribe_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Please speak something...")
        audio_data = r.record(source, duration=5)
        st.write("Recording time is over, wait for a few seconds...")
    try:
        text = r.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        st.write("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        st.write("Could not request results from Google Speech Recognition service; {0}".format(e))
    return ""

if st.button("Speak"):
    user_input = transcribe_audio()
    if user_input:
        st.write("You said: " + user_input)
    else:
        st.write("Sorry, I could not understand what you said. Please try again.")
else:
    user_input = st.text_input("Or type your mood:")


# Send the user's query to OpenAI GPT-3
if user_input:
    response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=f"Translate the following negative text into positive text: {user_input}",
    max_tokens=50
    )
    result_text = response.choices[0].text.strip()
    st.write(result_text)

