import os
import streamlit as st
import requests
from google.cloud import speech, texttospeech

# Set Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Workspace\\Curious_PM_Task\\curious-pm-task-506e4459ec02.json"

def transcribe_audio(audio_path):
    """Transcribe the given audio file using Google Speech-to-Text"""
    client = speech.SpeechClient()

    # Load audio file
    with open(audio_path, "rb") as audio_file:
        audio_content = audio_file.read()

    audio = speech.RecognitionAudio(content=audio_content)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )

    response = client.recognize(config=config, audio=audio)
    
    transcription = ""
    for result in response.results:
        transcription += result.alternatives[0].transcript

    return transcription

def correct_transcription(transcription):
    """Correct the transcription using Azure OpenAI GPT-4o"""
    azure_openai_key = "22ec84421ec24230a3638d1b51e3a7dc"
    azure_openai_endpoint = "https://internshala.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview"
    
    headers = {
        "Content-Type": "application/json",
        "api-key": azure_openai_key,
    }

    data = {
        "messages": [{"role": "user", "content": f"Correct the transcription: {transcription}"}],
        "max_tokens": 500
    }

    response = requests.post(azure_openai_endpoint, headers=headers, json=data)
    
    if response.status_code == 200:
        result = response.json()
        corrected_transcription = result['choices'][0]['message']['content']
        return corrected_transcription
    else:
        return transcription

def generate_audio_from_text(transcription_text):
    """Generate audio from text using Google Text-to-Speech with a male voice"""
    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(text=transcription_text)

    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Wavenet-D",  # Valid male voice
        ssml_gender=texttospeech.SsmlVoiceGender.MALE
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    output_audio_path = "output_speech.mp3"
    with open(output_audio_path, "wb") as out:
        out.write(response.audio_content)
        print(f'Generated audio saved to {output_audio_path}')

    return output_audio_path

def main():
    st.title("AI-Generated Voice for Video")  # Updated title

    audio_file = st.file_uploader("Upload a video file", type=["mp4", "wav"])
    
    if audio_file is not None:
        st.write("Processing audio...")

        # Step 1: Transcribe audio
        audio_path = audio_file.name
        with open(audio_path, "wb") as f:
            f.write(audio_file.getbuffer())
        
        transcription = transcribe_audio(audio_path)
        st.write("Original Transcription:", transcription)
        
        # Step 2: Correct transcription using GPT-4o
        corrected_transcription = correct_transcription(transcription)
        st.write("Corrected Transcription:", corrected_transcription)

        # Step 3: Generate new audio with corrected transcription
        output_audio_path = generate_audio_from_text(corrected_transcription)
        st.audio(output_audio_path)

if __name__ == "__main__":
    main()
