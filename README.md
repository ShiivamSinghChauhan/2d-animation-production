# 2D animation production

Tools for making 2D cut-out (rig) animation episodes, with AI image/music sites driven through the browser
instead of paid APIs.

## What is in here

- **Editor 1: Character studio** (`room/puppet.html`): arrange the parts of a character, soft-erase edges, colour
  brush, mark a part with a note to have it redrawn, Blender-style bones (extrude, attach parts), bending test.
- **Editor 2: Production editor** (`room/edit.html` + `room/engine.js`): timeline with shots, characters, voice,
  music and effects as separate clips; each clip keeps its prompt and tries; per-clip and per-row mute; notes;
  preview and final render use the same drawing/mixing code. `python cut.py render <ep>` writes an mp4
  (puppeteer + ffmpeg).
- **Browser automation** (`bridge/`): puppeteer-core drivers that use ChatGPT (images), Flow Music (music/SFX),
  Gemini and AI Studio TTS through *your own signed-in Chrome windows*. Selectors live in each driver's `SEL` block.
- **Production desk** (`prod.py`, `room.py`, `cut.py`, `fixdesk.py`, ): recipes -> jobs -> tries ->
  pick; the asset room is the page where you pick, redo and leave notes.

## Setup

```
pip install -r requirements.txt
cd bridge && npm install       # puppeteer-core
```
Needs Python 3.10+, Node 18+, Chrome, ffmpeg on PATH (set `CHROME` to the chrome.exe path if not default).

```
python room.py                 # http://127.0.0.1:8790  (asset room, /puppet?id=<id>, /edit?ep=<ep>)
python prod.py login           # opens Chrome windows: sign in by hand, once
```
Episodes live in `episodes/<ep>/` (lines.yaml, shots.yaml, cues.yaml, sound.yaml); this repo ships none.

## Notes

- Browser profiles are stored in `.browser/` and hold your cookies. They are git-ignored; never share them.
- The drivers imitate a person using the sites: one job at a time, on your own account. Check each site's terms
  of use before relying on this. Site layouts change, so expect to fix selectors (`node bridge/inspect.js <site>`).
- Work in progress, built for one channel's needs; rough edges expected. Paths in a few docstrings are examples.
