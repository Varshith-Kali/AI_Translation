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

@st.cache_data
def download_tokenizers():
    nltk.data.path.append(".")
    if os.path.exists("tokenizers/punkt_tab"):
        return
    
    nltk.download('punkt_tab', download_dir=".")


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
    # breakpoint()
    client = speech.SpeechClient()
    sentence_timings = []
    sentence = ""
    sentence_start_time = None
    filler_words = ["umm", "hmm", "uh", "ah", "erm"]  # Add more filler words if needed
    punctuation_marks = [".", "!", "?"]
    silence_threshold = 2  # Define a minimum gap length to consider as silence in seconds

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

        previous_end_time = None

        for result in response.results:
            words = result.alternatives[0].words
            for idx, word_info in enumerate(words):
                word = word_info.word.lower()
                start_time = word_info.start_time.total_seconds()
                end_time = word_info.end_time.total_seconds()

                # If there's a significant gap between the previous word and the current word, treat it as a pause
                if previous_end_time is not None and start_time - previous_end_time >= silence_threshold:
                    print(f"Detected SILENCE from {previous_end_time:.2f} to {start_time:.2f}")
                    sentence_timings.append(("SILENCE", previous_end_time, start_time))

                if sentence_start_time is None:
                    sentence_start_time = start_time

                sentence += " " + word

                # Check if word ends with a punctuation mark to denote the end of a sentence
                if any(word.endswith(punct) for punct in punctuation_marks):
                    print(f"Extracted sentence: '{sentence.strip()}' from {sentence_start_time:.2f} to {end_time:.2f}")
                    sentence_timings.append((sentence.strip(), sentence_start_time, end_time))
                    sentence = ""  # Reset the sentence
                    sentence_start_time = None  # Reset start time for the next sentence

                previous_end_time = end_time  # Update the previous end time

            # If the last sentence has no punctuation, append it with the final end_time of the last word
            if sentence:
                print(f"Extracted sentence: '{sentence.strip()}' from {sentence_start_time:.2f} to {end_time:.2f}")
                sentence_timings.append((sentence.strip(), sentence_start_time, end_time))
                sentence = ""
                sentence_start_time = None

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

def correct_transcription_with_timings(sentence_timings):
    azure_openai_key = "22ec84421ec24230a3638d1b51e3a7dc"
    azure_openai_endpoint = "https://internshala.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview"
    headers = {
        "Content-Type": "application/json",
        "api-key": azure_openai_key,
    }

    # Prepare the sentences to be corrected, excluding "SILENCE"
    sentences_to_correct = [sentence for sentence, _, _ in sentence_timings]

    # Create a prompt with instructions
    # Prompt is Please correct the following sentences by removing filler words like 'uh' and 'umm' and improving the grammar. Keep sentences marked as 'SILENCE' the same. Do not skip any sentences even if they are of one word. Do not reply with anything other than the corrected sentences:
    prompt = "Please correct the following sentences by removing filler words like 'uh' and 'umm' and improving the grammar. Keep sentences marked as 'SILENCE' the same. Do not skip any sentences even if they are of one word. Do not reply with anything other than the corrected sentences:\n\n"
    
    # Add each sentence to the prompt
    for sentence in sentences_to_correct:
        prompt += f"- {sentence}\n"

    # Prepare data for the API request
    data = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1000
    }

    # Send the request to Azure OpenAI
    response = requests.post(azure_openai_endpoint, headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        corrected_sentences = result['choices'][0]['message']['content'].strip().split("\n")

        corrected_sentence_timings = []
        corrected_index = 0

        # Rebuild sentence_timings with corrected sentences and original timings
        for sentence, start_time, end_time in sentence_timings:
            if sentence == "SILENCE":
                # Keep "SILENCE" as is
                corrected_sentence_timings.append((sentence, start_time, end_time))
                corrected_index += 1
            else:
                # Replace original sentence with the corrected one
                # Remove the "-" from the response
                corrected_sentence_timings.append((corrected_sentences[corrected_index].lstrip("-").strip(), start_time, end_time))
                corrected_index += 1

        return corrected_sentence_timings
    else:
        # Return the original timings if API call fails
        return sentence_timings


def generate_audio_with_sentence_timing(corrected_transcription, sentence_timings):
    client = texttospeech.TextToSpeechClient()

    # Generate a silent track based on the last sentence's end time from the original audio
    final_duration_ms = int(sentence_timings[-1][2] * 1000)
    final_sound = AudioSegment.silent(duration=final_duration_ms)

    current_time_ms = int(sentence_timings[0][1] * 1000)  # Start from the time of the first sentence


    for timing in sentence_timings:
        current_time_ms = int(timing[1] * 1000)

        if timing[0] == "SILENCE":
            # Insert a period of silence for the natural gap
            silence_duration_ms = int((timing[2] - timing[1]) * 1000)
            print(f"Adding SILENCE from {timing[1]:.2f} to {timing[2]:.2f} ({silence_duration_ms} ms)")
        else:

            # Synthesize speech for the corrected sentence
            synthesis_input = texttospeech.SynthesisInput(text=timing[0])
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

            # Calculate sentence duration and add silence if needed
            sentence_duration_ms = int((timing[2] - timing[1]) * 1000)
            print(f"Overlaying sentence '{timing[0]}' from {timing[1]:.2f} to {timing[2]:.2f} ({sentence_duration_ms} ms)")

            if len(corrected_audio_segment) > sentence_duration_ms:
                print(f"Trimming sentence audio to {sentence_duration_ms} ms")
                corrected_audio_segment = corrected_audio_segment[:sentence_duration_ms]
            else:
                silence = AudioSegment.silent(duration=sentence_duration_ms - len(corrected_audio_segment))
                corrected_audio_segment = corrected_audio_segment + silence

            # Overlay the corrected audio at the current time
            final_sound = final_sound.overlay(corrected_audio_segment, position=current_time_ms)

    # Export the final audio with corrected sentences at correct timestamps
    output_audio_path = "output_with_sentence_timing_and_gaps.wav"
    print(f"Exporting final audio to {output_audio_path}")
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
    
    download_tokenizers()

    video_file = st.file_uploader("Upload a video file", type=["mp4", "wav"])

    # Section for demo test with example videos
    st.markdown("<h3 style='color:#FF6347;'>Demo Videos</h3>", unsafe_allow_html=True)
    
    # Adding some demo videos for the user to choose from
    demo_videos = {
        "Sample 1": r"Sample_Videos/Sample_vid_01.mp4",
        "Sample 2": r"Sample_Videos/Sample_vid_02.mp4",
        "Sample 3": r"Sample_Videos/Sample_vid_03.mp4"
        # Add more demo video paths if needed
    }
    
    # Let user select a demo video or upload their own
    demo_video_choice = st.selectbox("Or select a demo video", options=["None"] + list(demo_videos.keys()))
    video_path = None
    if video_file is not None:
        video_path = video_file.name
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())
        

    elif demo_video_choice != "None":
        video_path = demo_videos[demo_video_choice]


    if video_path is not None:
        # Display original video
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
        correct_transcription_with_timings_obj = correct_transcription_with_timings(sentence_timings)
        
        # Corrected Transcription
        st.markdown("<h5 style='color:#FF6347; font-weight: bold;'>Corrected Transcription:</h5>", unsafe_allow_html=True)
        st.text_area("Corrected Transcription", corrected_transcription, height=150, key="corrected_transcription")
        
        # Generate new audio with sentence timing
        output_audio_path = generate_audio_with_sentence_timing(corrected_transcription, correct_transcription_with_timings_obj)
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
