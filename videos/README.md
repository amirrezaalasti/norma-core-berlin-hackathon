# Demo videos

Short clips of the ElRobot arm during the Berlin hackathon. **GIFs** and **MP4** versions are optimized for the root README; **MOV** files are the original phone recordings.

| File | MCP / voice command | Description |
|------|---------------------|-------------|
| `hi` | `say_hi` | Fast gripper wave hello |
| `dance` | `dance` | Arm sway + gripper flaps |
| `pickup` | `go_to_square` / `pick_object` | Pick from the board |
| `put` | `place_at_square` / `transfer_object` | Place on the board |

## Formats

| Format | Use |
|--------|-----|
| `.gif` | Embeds in GitHub README (autoplay loop) |
| `.mp4` | Smaller web preview; use with `<video autoplay loop muted>` |
| `.MOV` | Full-quality source |

## Regenerate web assets

```bash
cd videos
for f in dance hi pickup put; do
  ffmpeg -y -i "${f}.MOV" -vf "scale=360:-2" -an -movflags +faststart -pix_fmt yuv420p -crf 28 "${f}.mp4"
  ffmpeg -y -i "${f}.MOV" -vf "fps=6,scale=200:-1:flags=lanczos,palettegen=stats_mode=diff" -frames:v 1 "${f}_palette.png"
  ffmpeg -y -i "${f}.MOV" -i "${f}_palette.png" \
    -lavfi "fps=6,scale=200:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5" -an "${f}.gif"
  rm -f "${f}_palette.png"
done
```
