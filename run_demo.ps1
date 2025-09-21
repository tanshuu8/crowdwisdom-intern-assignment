# Activate virtual environment
. .\.venv312\Scripts\Activate.ps1

# Set environment variables
$env:CW_STT_FORCE_MOCK = "1"
# Set your API key if you want real GPT replies
# $env:OPENAI_API_KEY = "sk-..."

# Run conversation
python crew_main.py --turns 4 --stt-model tiny --tts-backend auto --phonikud

# Show outputs
Write-Host "==== Outputs ===="
Get-ChildItem -Recurse .\outputs | Sort-Object LastWriteTime | Select-Object LastWriteTime, FullName

# Open last stitched audio
$lastWav = Get-ChildItem .\outputs\full_conversation_*.wav | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($lastWav) { Start-Process $lastWav.FullName }

# Open last SRT
$lastSrt = Get-ChildItem .\outputs\transcripts\*.srt | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($lastSrt) { notepad $lastSrt.FullName }
