import os
import streamlit as st
import requests
from google.cloud import speech, texttospeech
import moviepy.editor as mp
from pydub import AudioSegment
import wave

# Set Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Workspace\\Curious_PM_Task\\curious-pm-task-506e4459ec02.json"

def convert_to_mono(audio_path):
    """Convert audio to mono if it's not already"""
    sound = AudioSegment.from_wav(audio_path)
    if sound.channels > 1:
        sound = sound.set_channels(1)
        sound.export(audio_path, format="wav")
        print("Converted audio to mono.")

def transcribe_audio(audio_path):
    """Transcribe the given audio file using Google Speech-to-Text"""
    client = speech.SpeechClient()

    # Convert the audio to mono before processing
    convert_to_mono(audio_path)

    # Load audio file and get sample rate
    with wave.open(audio_path, "rb") as audio_file:
        sample_rate = audio_file.getframerate()

    with open(audio_path, "rb") as audio_file:
        audio_content = audio_file.read()

    audio = speech.RecognitionAudio(content=audio_content)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate,  # Use the sample rate from the WAV file
        language_code="en-US",
    )

    response = client.recognize(config=config, audio=audio)

    transcription = ""
    for result in response.results:
        transcription += result.alternatives[0].transcript + " "

    return transcription.strip()

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
        name="en-US-Wavenet-D",  # Male voice
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

def replace_audio_in_video(video_path, audio_path, output_path):
    """Replace the audio in the video file with the new audio"""
    video = mp.VideoFileClip(video_path)
    audio = mp.AudioFileClip(audio_path)
    video = video.set_audio(audio)
    video.write_videofile(output_path, codec='libx264', audio_codec='aac')
    print(f'Output video saved to {output_path}')

def main():
    st.title("AI-Generated Voice for Video")

    video_file = st.file_uploader("Upload a video file", type=["mp4", "wav"])
    
    if video_file is not None:
        st.write("Processing audio...")

        # Step 1: Save uploaded video file
        video_path = video_file.name
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())
        
        # Step 2: Extract audio from the video
        audio_path = "temp_audio.wav"
        video_clip = mp.VideoFileClip(video_path)
        video_clip.audio.write_audiofile(audio_path)

        # Step 3: Transcribe audio
        transcription = transcribe_audio(audio_path)
        st.write("Original Transcription:", transcription)
        
        # Step 4: Correct transcription using GPT-4o
        corrected_transcription = correct_transcription(transcription)
        st.write("Corrected Transcription:", corrected_transcription)

        # Step 5: Generate new audio with corrected transcription
        output_audio_path = generate_audio_from_text(corrected_transcription)

        # Step 6: Replace audio in the original video with new audio
        output_video_path = "output_video.mp4"
        replace_audio_in_video(video_path, output_audio_path, output_video_path)

        # Display the final video
        st.video(output_video_path)

if __name__ == "__main__":
    main()