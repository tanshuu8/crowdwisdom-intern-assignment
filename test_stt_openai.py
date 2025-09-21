from openai import OpenAI

client = OpenAI()

# Change path if needed – this points to your latest client audio file
audio_path = "outputs/audio/client_1.wav"

with open(audio_path, "rb") as f:
    resp = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=f
    )

print("✅ Transcription:", resp.text)
