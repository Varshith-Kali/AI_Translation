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

def split_audio(audio_path, chunk_length_ms=59000):
    """Split the audio into smaller chunks (under 1 minute)"""
    sound = AudioSegment.from_wav(audio_path)
    chunks = []
    for i in range(0, len(sound), chunk_length_ms):
        chunk = sound[i:i + chunk_length_ms]
        chunk_path = f"chunk_{i // chunk_length_ms}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    return chunks

def transcribe_audio_chunks(chunk_paths):
    """Transcribe each audio chunk using Google Speech-to-Text and concatenate results"""
    client = speech.SpeechClient()
    transcription = ""

    for chunk_path in chunk_paths:
        with wave.open(chunk_path, "rb") as audio_file:
            sample_rate = audio_file.getframerate()

        with open(chunk_path, "rb") as audio_file:
            audio_content = audio_file.read()

        audio = speech.RecognitionAudio(content=audio_content)

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,  # Use the sample rate from the WAV file
            language_code="en-US",
        )

        # Call synchronous recognition for each chunk
        response = client.recognize(config=config, audio=audio)

        # Concatenate each chunk's transcription
        for result in response.results:
            transcription += result.alternatives[0].transcript + " "

    return transcription.strip()

def filter_filler_words(transcription):
    """Remove filler words and leave gaps for natural pauses"""
    filler_words = ["umm", "uh", "hmm", "ah", "like", "you know"]
    transcription_words = transcription.split()
    filtered_transcription = []
    for word in transcription_words:
        if word.lower() in filler_words:
            filtered_transcription.append("...")  # Leave gap for natural pause
        else:
            filtered_transcription.append(word)
    return " ".join(filtered_transcription)

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

def generate_audio_from_text(transcription_text, original_audio_duration):
    """Generate audio from text using Google Text-to-Speech, adjusting the rate to match original duration."""
    client = texttospeech.TextToSpeechClient()

    # Calculate the average speaking rate based on transcription length and original duration
    word_count = len(transcription_text.split())
    average_speaking_rate = word_count / original_audio_duration  # words per second

    # Normal speaking rate for English is around 150 words per minute (~2.5 words/second)
    default_rate = 2.5  # Adjust this based on your original audio
    speaking_rate = average_speaking_rate / default_rate

    # Generate audio with the adjusted speaking rate
    synthesis_input = texttospeech.SynthesisInput(text=transcription_text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Wavenet-D",  # Male voice
        ssml_gender=texttospeech.SsmlVoiceGender.MALE
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3,
        speaking_rate=speaking_rate  # Adjust speaking rate to match original duration
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

    # Trim or loop audio to match the video's duration
    if audio.duration > video.duration:
        audio = audio.subclip(0, video.duration)  # Trim the audio to match the video duration
    elif audio.duration < video.duration:
        silence_duration = video.duration - audio.duration
        silence = mp.AudioClip(lambda t: 0, duration=silence_duration)  # Add silence if audio is shorter
        audio = mp.concatenate_audioclips([audio, silence])

    video = video.set_audio(audio)

    try:
        # Save the final video with the new audio
        video.write_videofile(output_path, codec='libx264', audio_codec='aac')
        print(f'Output video saved to {output_path}')
    finally:
        # Close video and audio resources properly
        video.close()
        audio.close()
        del video
        del audio

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

        # Convert audio to mono
        convert_to_mono(audio_path)

        # Step 3: Split the audio into chunks (under 1 minute each)
        audio_chunks = split_audio(audio_path)

        # Step 4: Transcribe the audio chunks
        transcription = transcribe_audio_chunks(audio_chunks)
        st.write("Original Transcription:", transcription)
        
        # Step 5: Filter filler words
        filtered_transcription = filter_filler_words(transcription)
        st.write("Filtered Transcription:", filtered_transcription)
        
        # Step 6: Correct transcription using GPT-4o
        corrected_transcription = correct_transcription(filtered_transcription)
        st.write("Corrected Transcription:", corrected_transcription)

        # Step 7: Generate new audio with corrected transcription without stretching words
        output_audio_path = generate_audio_from_text(corrected_transcription, video_clip.duration)

        # Step 8: Replace audio in the original video with new audio, ensuring sync
        output_video_path = "output_video.mp4"
        replace_audio_in_video(video_path, output_audio_path, output_video_path)

        # Display the final video
        st.video(output_video_path)

        # Close the video file clip
        video_clip.close()
        del video_clip

if __name__ == "__main__":
    main()
