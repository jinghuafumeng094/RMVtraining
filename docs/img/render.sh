#!/usr/bin/env bash
# 把本目录的 svg 渲染成 png。
#
# 为什么用 Chrome：这台机器上没有 rsvg-convert / inkscape / ImageMagick，
# Chrome 无头模式是唯一能用的 svg 光栅化器。
#
# 用法：./render.sh            渲染全部 svg
#       ./render.sh 坐标系方向   只渲染一个（不带 .svg 后缀）
set -euo pipefail

cd "$(dirname "$0")"

render() {
    local name="$1"
    # 尺寸从 svg 的 width/height 属性读，免得两边对不上
    local w h
    w=$(grep -oP '(?<=^<svg xmlns)[^>]*\bwidth="\K[0-9]+' "$name.svg")
    h=$(grep -oP '(?<=^<svg xmlns)[^>]*\bheight="\K[0-9]+' "$name.svg")
    google-chrome --headless --disable-gpu --no-sandbox --hide-scrollbars \
        --default-background-color=FFFFFFFF \
        --window-size="$w,$h" \
        --screenshot="$PWD/$name.png" \
        "file://$PWD/$name.svg" 2>/dev/null
    echo "$name.png  ${w}x${h}"
}

if [ $# -gt 0 ]; then
    for name in "$@"; do render "$name"; done
else
    for f in *.svg; do render "${f%.svg}"; done
fi
