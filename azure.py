# Import necessary libraries
import streamlit as st
import openai
import speech_recognition as sr
import os


# Set up your OpenAI API key
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
    prompt="Translate the following negative text into positive text: user_input",
    max_tokens=50
    )
    result_text = response.choices[0].text.strip()
print(result_text)

from audiocraft.models import MusicGen
from audiocraft.models import MultiBandDiffusion

USE_DIFFUSION_DECODER = False
# Using small model, better results would be obtained with `medium` or `large`.
model = MusicGen.get_pretrained('facebook/musicgen-small')
if USE_DIFFUSION_DECODER:
    mbd = MultiBandDiffusion.get_mbd_musicgen()

model.set_generation_params(
    use_sampling=True,
    top_k=250,
    duration=30
)

import math
import torchaudio
import torch
from audiocraft.utils.notebook import display_audio

def get_bip_bip(bip_duration=0.125, frequency=440,
                duration=0.5, sample_rate=32000, device="cuda"):
    """Generates a series of bip bip at the given frequency."""
    t = torch.arange(
        int(duration * sample_rate), device="cuda", dtype=torch.float) / sample_rate
    wav = torch.cos(2 * math.pi * 440 * t)[None]
    tp = (t % (2 * bip_duration)) / (2 * bip_duration)
    envelope = (tp >= 0.5).float()
    return wav * envelope

# Here we use a synthetic signal to prompt both the tonality and the BPM
# of the generated audio.
res = model.generate_continuation(
    get_bip_bip(0.125).expand(2, -1, -1),
    32000, ['Jazz jazz and only jazz',
            'Heartful EDM with beautiful synths and chords'],
    progress=True)
display_audio(res, 32000)

from audiocraft.utils.notebook import display_audio

output = model.generate(
    descriptions=[
        #'80s pop track with bassy drums and synth',
        #'90s rock song with loud guitars and heavy drums',
        #'Progressive rock drum and bass solo',
        #'Punk Rock song with loud drum and power guitar',
        #'Bluesy guitar instrumental with soulful licks and a driving rhythm section',
        #'Jazz Funk song with slap bass and powerful saxophone',
        'drum and bass beat with intense percussions',
        result_text
    ],
    progress=True, return_tokens=True
)
display_audio(output[0], sample_rate=32000)
if USE_DIFFUSION_DECODER:
    out_diffusion = mbd.tokens_to_wav(output[1])
    display_audio(out_diffusion, sample_rate=32000)

