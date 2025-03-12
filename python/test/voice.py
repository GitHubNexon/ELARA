import argparse
import torch
from transformers import AutoProcessor, SpeechT5ForTextToSpeech
import scipy.io.wavfile as wavfile
import numpy as np
from datasets import load_dataset
import os
from pydub import AudioSegment 

def generate_voice(text):
    try:
        # Print status
        print(f"Generating speech for: '{text}'")
        
        processor = AutoProcessor.from_pretrained("microsoft/speecht5_tts")
        model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        
        dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        
        print(f"Dataset fields: {dataset.column_names}")
        print(f"First example: {dataset[0]}")

        xvector_data = dataset[0]["xvector"]
        
        speaker_embeddings = torch.tensor(xvector_data).unsqueeze(0)  # Add batch dimension

        inputs = processor(text=text, return_tensors="pt")
        
        with torch.no_grad():
            speech = model.generate(inputs["input_ids"], speaker_embeddings=speaker_embeddings)
        
        speech_np = speech.squeeze().cpu().numpy()
        
        speech_np = speech_np / np.max(np.abs(speech_np))  # Normalize to -1 to 1
        speech_np = speech_np * 0.9  # Keep a margin to prevent clipping
        
        sample_rate = 16000
        
        speech_np = np.int16(speech_np * 32767)  # Convert float to int16
        
        script_dir = os.path.dirname(os.path.realpath(__file__))
        raw_wav_output_file = os.path.join(script_dir, "output_audio_raw.wav")  # Save as raw WAV
        
        wavfile.write(raw_wav_output_file, sample_rate, speech_np)

        final_wav_output_file = os.path.join(script_dir, "output_audio_final.wav")
        audio = AudioSegment.from_wav(raw_wav_output_file)  # Load the raw PCM WAV
        audio.export(final_wav_output_file, format="wav")  # Export in proper WAV format
        
        os.remove(raw_wav_output_file)
        
        print(f"Audio generated successfully and saved to '{final_wav_output_file}'")
        print(f"Duration: {len(speech_np) / sample_rate:.2f} seconds")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Try installing required packages with: pip install transformers torch scipy numpy sentencepiece datasets pydub")

def main():
    parser = argparse.ArgumentParser(description="Generate voice from text")
    parser.add_argument("--text", type=str, required=True, help="Text to convert to speech")
    args = parser.parse_args()

    generate_voice(args.text)

if __name__ == "__main__":
    main()
