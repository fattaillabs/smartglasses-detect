#!/bin/zsh
# 유튜브 영상에서 프레임을 뽑아 안경 띠로 잘라 저장한다. 원본 영상과 프레임은 지운다.
# 사용: src/yt_collect.sh <라벨> <영상ID> [영상ID ...]
# 결과: data/crops/yt/<라벨>/<영상ID>/{band,corner,meta.jsonl}  (YT_DEST=yt_test 로 주면 테스트 전용 폴더에 저장)
# 키프레임만 읽어 빠르게 뽑는다(대략 2~5초에 한 장).
set -u
ROOT="${0:A:h:h}"
PY="$ROOT/.venv/bin/python"
LABEL="$1"; shift
for VID in "$@"; do
  OUT="$ROOT/data/crops/${YT_DEST:-yt}/$LABEL/$VID"
  if [[ -f "$OUT/.done" ]]; then echo "skip $VID"; continue; fi
  TMP="$ROOT/data/raw/yt/$VID"
  mkdir -p "$TMP/frames" "$OUT"
  "$ROOT/.venv/bin/yt-dlp" -q --no-warnings --no-playlist -f "bv*[height<=1080][vcodec^=avc1]/bv*[height<=1080]" \
    -o "$TMP/v.%(ext)s" "https://www.youtube.com/watch?v=$VID" </dev/null || { echo "FAIL download $VID"; continue; }
  V=$(ls "$TMP"/v.* | head -1)
  ffmpeg -nostdin -loglevel error -skip_frame nokey -i "$V" -fps_mode vfr -vf "fps=1/2" -q:v 2 "$TMP/frames/${VID}_%04d.jpg"
  N=$(ls "$TMP/frames" | wc -l | tr -d ' ')
  R=$(arch -arm64 "$PY" "$ROOT/src/crop.py" "$TMP/frames" "$OUT" --label "$LABEL" 2>/dev/null | tail -1)
  echo "$VID frames=$N $R"
  rm -rf "$TMP"
  touch "$OUT/.done"
done
