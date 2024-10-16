import os
import streamlit as st
import requests
from google.cloud import speech, texttospeech
import moviepy.editor as mp
from pydub import AudioSegment
import wave
import io

# Set Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "C:\\Workspace\\Curious_PM_Task\\curious-pm-task-506e4459ec02.json"

def convert_to_mono(audio_path):
    sound = AudioSegment.from_wav(audio_path)
    if sound.channels > 1:
        sound = sound.set_channels(1)
        sound.export(audio_path, format="wav")


def split_audio(audio_path, chunk_length_ms=59000):
    sound = AudioSegment.from_wav(audio_path)
    chunks = []
    for i in range(0, len(sound), chunk_length_ms):
        chunk = sound[i:i + chunk_length_ms]
        chunk_path = f"chunk_{i // chunk_length_ms}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    return chunks


def get_word_timing_from_audio(chunk_paths):
    client = speech.SpeechClient()
    word_timings = []
    for chunk_path in chunk_paths:
        with wave.open(chunk_path, "rb") as audio_file:
            sample_rate = audio_file.getframerate()
        with open(chunk_path, "rb") as audio_file:
            audio_content = audio_file.read()
        audio = speech.RecognitionAudio(content=audio_content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,
            language_code="en-US",
            enable_word_time_offsets=True
        )
        response = client.recognize(config=config, audio=audio)
        for result in response.results:
            for word_info in result.alternatives[0].words:
                word_timings.append((word_info.word, word_info.start_time.total_seconds(), word_info.end_time.total_seconds()))
    return word_timings


def transcribe_audio_chunks(chunk_paths):
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
            sample_rate_hertz=sample_rate,
            language_code="en-US",
        )
        response = client.recognize(config=config, audio=audio)
        for result in response.results:
            transcription += result.alternatives[0].transcript + " "
    return transcription.strip()


def filter_filler_words(transcription):
    filler_words = ["umm", "uh", "hmm", "ah", "like", "you know"]
    transcription_words = transcription.split()
    filtered_transcription = []
    for word in transcription_words:
        if word.lower() in filler_words:
            filtered_transcription.append("...")
        else:
            filtered_transcription.append(word)
    return " ".join(filtered_transcription)


def correct_transcription(transcription):
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
        corrected_transcription = result['choices'][0]['message']['content'].replace("Corrected Transcription:", "").strip()
        return corrected_transcription
    else:
        return transcription


def generate_audio_with_natural_flow(transcription_text, original_audio_duration, word_timings):
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=transcription_text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Wavenet-D",
        ssml_gender=texttospeech.SsmlVoiceGender.MALE
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
    )
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    output_audio_path = "output_with_natural_flow.wav"
    with open(output_audio_path, "wb") as out:
        out.write(response.audio_content)
    sound = AudioSegment.from_wav(output_audio_path)
    final_sound = AudioSegment.silent(duration=int(original_audio_duration * 1000))
    word_index = 0
    for word, start_time, end_time in word_timings:
        word_duration = int((end_time - start_time) * 1000)
        word_audio = sound[word_index:word_index + word_duration]
        final_sound = final_sound.overlay(word_audio, start_time * 1000)
        word_index += word_duration
    final_sound.export(output_audio_path, format="wav")
    return output_audio_path


def replace_audio_in_video(video_path, audio_path, output_path):
    video = mp.VideoFileClip(video_path)
    audio = mp.AudioFileClip(audio_path)
    video = video.set_audio(audio)
    try:
        video.write_videofile(output_path, codec='libx264', audio_codec='aac')
    finally:
        video.close()
        audio.close()
        del video
        del audio


def main():
    st.title("AI-Generated Voice for Video")
    video_file = st.file_uploader("Upload a video file", type=["mp4", "wav"])
    if video_file is not None:
        st.write("Processing audio...")
        video_path = video_file.name
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())
        audio_path = "temp_audio.wav"
        video_clip = mp.VideoFileClip(video_path)
        video_clip.audio.write_audiofile(audio_path)
        convert_to_mono(audio_path)
        audio_chunks = split_audio(audio_path)
        word_timings = get_word_timing_from_audio(audio_chunks)
        transcription = transcribe_audio_chunks(audio_chunks)
        st.write("Original Transcription:", transcription)
        filtered_transcription = filter_filler_words(transcription)
        st.write("Filtered Transcription:", filtered_transcription)
        corrected_transcription = correct_transcription(filtered_transcription)
        st.write("Corrected Transcription:", corrected_transcription)
        output_audio_path = generate_audio_with_natural_flow(corrected_transcription, video_clip.duration, word_timings)
        output_video_path = "output_video.mp4"
        replace_audio_in_video(video_path, output_audio_path, output_video_path)
        st.video(output_video_path)
        video_clip.close()
        del video_clip


if __name__ == "__main__":
    main()