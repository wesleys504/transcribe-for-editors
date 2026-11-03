#!/bin/bash
# Transcribe installer. Safe to run again to update or repair.
# Copyright (c) 2026 TBD Media Post. All rights reserved.
set -e
cd "$(dirname "$0")"
SRC="$(pwd)"
APP_HOME="$HOME/Transcribe"
ARCH="$(uname -m)"

say_step() { echo; echo "==> $1"; }

clear
echo "Transcribe installer"
echo "This sets up local transcription in $APP_HOME and adds a Transcribe app to your Applications folder."
echo "It downloads about 2–3 GB the first time (Python, Whisper and speaker models)."
echo
read -r -p "Press Return to continue, or close this window to cancel. " _

mkdir -p "$APP_HOME/app/ui" "$APP_HOME/bin" "$APP_HOME/models" "$APP_HOME/Watch Folder"
cp "$SRC/transcribe.py" "$SRC/app.py" "$SRC/icon.png" "$APP_HOME/app/"
cp "$SRC/ui/index.html" "$APP_HOME/app/ui/index.html"

say_step "Installing uv (a self-contained Python manager) into $APP_HOME/bin"
if [ ! -x "$APP_HOME/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$APP_HOME/bin" UV_NO_MODIFY_PATH=1 sh
fi
UV="$APP_HOME/bin/uv"

say_step "Setting up Python 3.12"
"$UV" venv --python 3.12 --allow-existing "$APP_HOME/venv"
PY="$APP_HOME/venv/bin/python"

say_step "Installing transcription libraries"
COMMON="sherpa-onnx>=1.12,<3 av>=12 numpy python-docx certifi truststore pywebview>=5"
if [ "$ARCH" = "arm64" ]; then
  echo "Apple Silicon Mac — using MLX Whisper (fast, uses the GPU)."
  "$UV" pip install --python "$PY" $COMMON mlx-whisper
else
  echo "Intel Mac — using faster-whisper."
  "$UV" pip install --python "$PY" $COMMON faster-whisper
fi

say_step "Downloading speaker-label models"
REL="https://github.com/k2-fsa/sherpa-onnx/releases/download"
if [ ! -f "$APP_HOME/models/sherpa-onnx-pyannote-segmentation-3-0/model.onnx" ]; then
  curl -L --fail -o "$APP_HOME/models/seg.tar.bz2" "$REL/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
  tar xjf "$APP_HOME/models/seg.tar.bz2" -C "$APP_HOME/models" && rm "$APP_HOME/models/seg.tar.bz2"
fi
if [ ! -f "$APP_HOME/models/nemo_en_titanet_small.onnx" ]; then
  curl -L --fail -o "$APP_HOME/models/nemo_en_titanet_small.onnx" "$REL/speaker-recongition-models/nemo_en_titanet_small.onnx"
fi

say_step "Writing helper scripts"
cat > "$APP_HOME/run.sh" <<EOF
#!/bin/bash
export PATH="/usr/bin:/bin:/usr/sbin:/sbin"
exec "$PY" -I "$APP_HOME/app/transcribe.py" "\$@"
EOF
chmod +x "$APP_HOME/run.sh"

cat > "$APP_HOME/watch-on.sh" <<EOF
#!/bin/bash
PLIST="\$HOME/Library/LaunchAgents/com.transcribe.watchfolder.plist"
mkdir -p "\$HOME/Library/LaunchAgents" "$APP_HOME/Watch Folder"
cat > "\$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.transcribe.watchfolder</string>
  <key>ProgramArguments</key><array><string>$APP_HOME/run.sh</string><string>--watch</string></array>
  <key>WatchPaths</key><array><string>$APP_HOME/Watch Folder</string></array>
  <key>StartInterval</key><integer>120</integer>
  <key>LowPriorityIO</key><true/>
  <key>Nice</key><integer>5</integer>
  <key>StandardOutPath</key><string>$APP_HOME/watch.out.log</string>
  <key>StandardErrorPath</key><string>$APP_HOME/watch.out.log</string>
</dict></plist>
PL
launchctl bootout gui/\$(id -u)/com.transcribe.watchfolder 2>/dev/null || true
launchctl bootstrap gui/\$(id -u) "\$PLIST"
EOF
chmod +x "$APP_HOME/watch-on.sh"

[ -f "$APP_HOME/names.txt" ] || printf "# One name or term per line. Lines starting with # are ignored.\n" > "$APP_HOME/names.txt"
[ -f "$APP_HOME/settings.json" ] || echo '{}' > "$APP_HOME/settings.json"

say_step "Building the Transcribe app"
APP="$HOME/Applications/Transcribe.app"
mkdir -p "$HOME/Applications"
rm -rf "$APP" "$HOME/Desktop/Transcribe.app"   # also removes the older drag-and-drop version
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$SRC/icon.icns" "$APP/Contents/Resources/Transcribe.icns"
cat > "$APP/Contents/Info.plist" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>Transcribe</string>
  <key>CFBundleDisplayName</key><string>Transcribe</string>
  <key>CFBundleIdentifier</key><string>local.transcribe.app</string>
  <key>CFBundleVersion</key><string>2.1</string>
  <key>CFBundleShortVersionString</key><string>2.1</string>
  <key>NSHumanReadableCopyright</key><string>© 2026 TBD Media Post. All rights reserved.</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>Transcribe</string>
  <key>CFBundleIconFile</key><string>Transcribe</string>
  <key>LSMinimumSystemVersion</key><string>12.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict></plist>
PL
cat > "$APP/Contents/MacOS/Transcribe" <<LAUNCH
#!/bin/bash
export PATH="/usr/bin:/bin:/usr/sbin:/sbin"
exec "$PY" -I "$APP_HOME/app/app.py" "\$@" >> "$APP_HOME/app.log" 2>&1
LAUNCH
chmod +x "$APP/Contents/MacOS/Transcribe"
touch "$APP"
ln -sfn "$APP" "$HOME/Desktop/Transcribe"   # Desktop shortcut

say_step "Downloading the Whisper model (about 1.6 GB — this can take a few minutes)"
"$PY" - <<'EOF'
import platform, numpy as np
audio = np.zeros(16000, dtype=np.float32)
if platform.machine() == "arm64":
    import mlx_whisper
    mlx_whisper.transcribe(audio, path_or_hf_repo="mlx-community/whisper-large-v3-turbo")
else:
    from faster_whisper import WhisperModel
    WhisperModel("large-v3-turbo", device="auto", compute_type="int8")
print("Whisper model ready.")
EOF

say_step "Quick self-test"
"$PY" -c "import sherpa_onnx, av, docx, certifi, truststore, webview; print('Libraries OK')"
"$PY" - <<'EOF' || echo "⚠️  Couldn't reach Claude securely. Transcription still works; tell Claude (the assistant) about this message."
import ssl, urllib.request, truststore
try:
    urllib.request.urlopen("https://api.anthropic.com", timeout=15, context=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT))
except urllib.error.HTTPError:
    pass  # any HTTP reply means the secure connection worked
print("Secure connection to Claude OK")
EOF

echo
echo "✅ All set."
echo "   • Transcribe is in your Applications folder (and on your Desktop). Drag it to your Dock to keep it handy."
echo "   • Drop clips or folders onto its window. Settings has your API key, names, translation and more."
osascript -e 'display notification "Transcribe is installed and opening now." with title "Transcribe"' || true
open "$APP"
