import os
import streamlit as st
import requests
from google.cloud import speech, texttospeech
import moviepy.editor as mp
from pydub import AudioSegment
import wave
import nltk
from nltk.tokenize import sent_tokenize
from pydub import AudioSegment
import io
from google.cloud import texttospeech


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"credentials.json"
nltk.download('punkt_tab',download_dir=".")
nltk.data.path.append(".")

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


def get_sentence_timing_from_audio(chunk_paths):
    client = speech.SpeechClient()
    sentence_timings = []
    sentence = ""
    sentence_start_time = None
    filler_words = ["umm", "hmm", "uh", "ah", "erm"]  # Add more filler words if needed

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
                word = word_info.word.lower()
                start_time = word_info.start_time.total_seconds()
                end_time = word_info.end_time.total_seconds()

                if sentence_start_time is None:
                    sentence_start_time = start_time

                # Use filler words or punctuation as a stopping point
                if word in filler_words or word.endswith("."):
                    if sentence:
                        # Add the sentence and its timing
                        sentence_timings.append((sentence.strip(), sentence_start_time, end_time))
                    sentence = ""
                    sentence_start_time = None
                else:
                    sentence += " " + word

    # Handle leftover sentence
    if sentence:
        sentence_timings.append((sentence.strip(), sentence_start_time, end_time))
        
    return sentence_timings


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
        corrected_transcription = result['choices'][0]['message']['content'].strip()

        # List of unwanted phrases to remove from the OpenAI response
        unwanted_phrases = [
            "Sure, here is the corrected transcription:",
            "Here is the corrected version:",
            "I simplified and clarified the original transcription:",
            "Certainly! Here's a corrected version of the transcription:"
        ]
        
        # Remove any unwanted phrases from the corrected transcription
        for phrase in unwanted_phrases:
            corrected_transcription = corrected_transcription.replace(phrase, "").strip()

        return corrected_transcription
    else:
        return transcription


def generate_audio_with_sentence_timing(corrected_transcription, sentence_timings):



    client = texttospeech.TextToSpeechClient()

    # Split corrected transcription into sentences using nltk's sent_tokenize
    corrected_sentences = sent_tokenize(corrected_transcription)

    # Generate a silent track based on the last sentence's end time from the original audio
    final_duration_ms = int(sentence_timings[-1][2] * 1000)
    final_sound = AudioSegment.silent(duration=final_duration_ms)

    # Debugging info
    print("Corrected Sentences:", corrected_sentences)
    print("Original Sentence Timings:", sentence_timings)

    # Calculate total original duration
    total_original_duration_ms = int((sentence_timings[-1][2] - sentence_timings[0][1]) * 1000)

    # Calculate durations for corrected sentences proportionally
    total_corrected_sentences = len(corrected_sentences)
    if total_corrected_sentences == 0:
        return None

    # Distribute durations proportionally among corrected sentences
    corrected_sentence_durations = [total_original_duration_ms // total_corrected_sentences] * total_corrected_sentences

    current_time_ms = int(sentence_timings[0][1] * 1000)

    for idx, corrected_sentence in enumerate(corrected_sentences):
        corrected_duration_ms = corrected_sentence_durations[idx]

        # Synthesize speech for the corrected sentence
        synthesis_input = texttospeech.SynthesisInput(text=corrected_sentence)
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

        corrected_audio_segment = AudioSegment.from_wav(io.BytesIO(response.audio_content))

        # Adjust the audio segment to match the calculated duration
        if len(corrected_audio_segment) > corrected_duration_ms:
            corrected_audio_segment = corrected_audio_segment[:corrected_duration_ms]
        else:
            silence = AudioSegment.silent(duration=corrected_duration_ms - len(corrected_audio_segment))
            corrected_audio_segment = corrected_audio_segment + silence

        # Overlay the corrected audio at the current time
        final_sound = final_sound.overlay(corrected_audio_segment, position=current_time_ms)

        current_time_ms += corrected_duration_ms

    # Export the final audio with corrected sentences at correct timestamps
    output_audio_path = "output_with_sentence_timing.wav"
    final_sound.export(output_audio_path, format="wav")
    
    return output_audio_path



def replace_audio_in_video(video_path, audio_path, output_path):
    video = mp.VideoFileClip(video_path)
    audio = mp.AudioFileClip(audio_path)
    video = video.set_audio(audio)
    try:
        video.write_videofile(output_path, codec='libx264', audio_codec='aac', preset='ultrafast', threads=4)
    finally:
        video.close()
        audio.close()
        del video
        del audio


    # Cleanup temporary audio files
    if os.path.exists(audio_path):
        os.remove(audio_path)


    # Cleanup temporary audio files
    if os.path.exists(audio_path):
        os.remove(audio_path)


def main():
    # Title in red color with padding below
    st.markdown("<h1 style='color:#FF6347; font-weight: bold;'>AI-Generated Voice for Video</h1>", unsafe_allow_html=True)
    
    # Adding padding
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)  # 30px of space between title and uploader
    
    # File uploader for user-uploaded video
    video_file = st.file_uploader("Upload a video file", type=["mp4", "wav"])

    # Section for demo test with example videos
    st.markdown("<h3 style='color:#FF6347;'>Demo Test Videos</h3>", unsafe_allow_html=True)
    
    # Adding some demo videos for the user to choose from
    demo_videos = {
        "Sample 1": r"Sample_Videos/Sample_vid_01.mp4",
        "Sample 2": r"Sample_Videos/Sample_vid_02.mp4",
        "Sample 3": r"Sample_Videos/Sample_vid_03.mp4"
        # Add more demo video paths if needed
    }
    
    # Let user select a demo video or upload their own
    demo_video_choice = st.selectbox("Or select a demo video", options=["None"] + list(demo_videos.keys()))
    
    # Check if the user selects a demo video or uploads their own
    video_path = None
    if video_file is not None:
        video_path = video_file.name
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())
    elif demo_video_choice != "None":
        video_path = demo_videos[demo_video_choice]
    
    if video_path is not None:
        # Display the selected video (uploaded or demo)
        st.subheader("Original Video:")
        st.video(video_path)
        
        # "Processing audio..." below the original video
        st.markdown("<h3 style='color:#FF6347; font-weight: bold;'>Processing audio...</h3>", unsafe_allow_html=True)

        audio_path = "temp_audio.wav"
        video_clip = mp.VideoFileClip(video_path)
        video_clip.audio.write_audiofile(audio_path)
        convert_to_mono(audio_path)
        audio_chunks = split_audio(audio_path)
        sentence_timings = get_sentence_timing_from_audio(audio_chunks)
        transcription = transcribe_audio_chunks(audio_chunks)
        
        # Original Transcription
        st.markdown("<h5 style='color:#FF6347; font-weight: bold;'>Original Transcription:</h5>", unsafe_allow_html=True)
        st.text_area("Original Transcription", transcription, height=150, key="original_transcription")
        
        corrected_transcription = correct_transcription(transcription)
        
        # Corrected Transcription
        st.markdown("<h5 style='color:#FF6347; font-weight: bold;'>Corrected Transcription:</h5>", unsafe_allow_html=True)
        st.text_area("Corrected Transcription", corrected_transcription, height=150, key="corrected_transcription")
        
        # Generate new audio with sentence timing
        output_audio_path = generate_audio_with_sentence_timing(corrected_transcription, sentence_timings)
        if output_audio_path:
            output_video_path = "output_video_with_timing.mp4"
            replace_audio_in_video(video_path, output_audio_path, output_video_path)
            
            # Display the new video
            st.subheader("Modified Video:")
            st.video(output_video_path)
        
        # Cleanup video clip resources
        video_clip.close()
        del video_clip


if __name__ == "__main__":
    main()
