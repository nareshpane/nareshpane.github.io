#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$PROJECT_DIR/build"
CLASS_DIR="$BUILD_DIR/classes"
FRAME_DIR="$BUILD_DIR/frames"

for tool in java javac; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Missing required tool: $tool. Install a JDK (Java 17 or newer is recommended)." >&2
    exit 1
  fi
done

mkdir -p "$CLASS_DIR" "$FRAME_DIR" "$PROJECT_DIR/traces" "$PROJECT_DIR/images" "$PROJECT_DIR/videos"
find "$CLASS_DIR" -type f -name '*.class' -delete
find "$FRAME_DIR" -type f -name 'frame-*.png' -delete

echo "Compiling Java sources..."
javac -encoding UTF-8 -d "$CLASS_DIR" "$PROJECT_DIR"/src/*.java

echo "Running correctness checks..."
java -Djava.awt.headless=true -cp "$CLASS_DIR" MatrixTests

echo "Generating JSON traces..."
java -Djava.awt.headless=true -cp "$CLASS_DIR" TraceWriter "$PROJECT_DIR/traces"

echo "Rendering Java2D poster, recap, and video frames..."
java -Djava.awt.headless=true -cp "$CLASS_DIR" MatrixAnimationGenerator "$FRAME_DIR" "$PROJECT_DIR/images"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg was not found. Java images and frames were generated, but no MP4 was created." >&2
  echo "Install ffmpeg, then rerun this script to encode videos/matrix-multiplication-in-motion.mp4." >&2
  exit 0
fi

echo "Encoding H.264/yuv420p MP4..."
ffmpeg -hide_banner -loglevel warning -y -framerate 20 -i "$FRAME_DIR/frame-%04d.png" \
  -c:v libx264 -preset medium -crf 23 -pix_fmt yuv420p -movflags +faststart \
  "$PROJECT_DIR/videos/matrix-multiplication-in-motion.mp4"

find "$FRAME_DIR" -type f -name 'frame-*.png' -delete
echo "Build complete. Temporary PNG frames were removed after encoding."
