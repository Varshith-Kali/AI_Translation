import os
import streamlit as st
import requests
from google.cloud import speech, texttospeech
import moviepy.editor as mp
from pydub import AudioSegment
import wave
import io


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

    # Split corrected transcription into sentences
    corrected_sentences = corrected_transcription.split(". ")  # Assuming sentences are separated by ". "
    
    # Generate a silent track based on the last sentence's end time from the original audio
    final_sound = AudioSegment.silent(duration=int(sentence_timings[-1][2] * 1000))  # Duration based on the last sentence

    # Debugging info
    print("Corrected Sentences:", corrected_sentences)
    print("Original Sentence Timings:", sentence_timings)

    for idx, (original_sentence, start_time, end_time) in enumerate(sentence_timings):
        original_sentence_duration = (end_time - start_time) * 1000  # Convert to milliseconds

        if idx < len(corrected_sentences):
            corrected_sentence = corrected_sentences[idx]
        else:
            # If there are fewer corrected sentences, fallback to the original sentence
            corrected_sentence = original_sentence

        # Debugging: print original and corrected sentences with timing info
        print(f"Original Sentence {idx}: '{original_sentence}' from {start_time}s to {end_time}s")
        print(f"Corrected Sentence {idx}: '{corrected_sentence}'")

        # Synthesize speech for the corrected sentence
        synthesis_input = texttospeech.SynthesisInput(text=corrected_sentence)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Wavenet-D",  # Customize the voice
            ssml_gender=texttospeech.SsmlVoiceGender.MALE
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        )
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        # Convert synthesized audio to an AudioSegment
        corrected_audio_segment = AudioSegment.from_wav(io.BytesIO(response.audio_content))

        # Generate speech for the remaining part of the original sentence
        # Remove the part of the original sentence that overlaps with the corrected sentence
        if corrected_sentence.lower() in original_sentence.lower():
            remaining_sentence = original_sentence.lower().replace(corrected_sentence.lower(), "").strip()
        else:
            remaining_sentence = original_sentence[len(corrected_sentence):].strip()

        print(f"Remaining Sentence {idx}: '{remaining_sentence}'")

        # Synthesize speech for the remaining part of the original sentence
        if remaining_sentence:
            synthesis_input_rem = texttospeech.SynthesisInput(text=remaining_sentence)
            response_rem = client.synthesize_speech(
                input=synthesis_input_rem, voice=voice, audio_config=audio_config
            )
            remaining_audio_segment = AudioSegment.from_wav(io.BytesIO(response_rem.audio_content))
        else:
            remaining_audio_segment = AudioSegment.silent(duration=0)

        # Combine the corrected audio and remaining original sentence
        combined_audio_segment = corrected_audio_segment + remaining_audio_segment

        # Debugging: print length of combined sentence audio and original duration
        print(f"Combined audio length: {len(combined_audio_segment)}ms, Original duration: {original_sentence_duration}ms")

        # Trim or pad the synthesized audio to match the original sentence duration
        if len(combined_audio_segment) > original_sentence_duration:
            # If the combined audio is longer, trim it
            combined_audio_segment = combined_audio_segment[:int(original_sentence_duration)]
        elif len(combined_audio_segment) < original_sentence_duration:
            # If the combined audio is shorter, pad with silence to match the duration
            silence = AudioSegment.silent(duration=int(original_sentence_duration - len(combined_audio_segment)))
            combined_audio_segment = combined_audio_segment + silence

        # Overlay the corrected sentence audio at the original start time
        final_sound = final_sound.overlay(combined_audio_segment, position=int(start_time * 1000))

    # Export the final audio with corrected sentences at correct timestamps
    output_audio_path = "output_with_sentence_timing.wav"
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
    
    video_file = st.file_uploader("Upload a video file", type=["mp4", "wav"])
    
    if video_file is not None:
        video_path = video_file.name
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())
        
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
