---
title: Hangpost Matching Engine Demo
emoji: 👥
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
short_description: Pick a profile, see who the ranker recommends and why.
---

# Hangpost Matching Engine — Live Demo

Pick a source profile and see the top-10 recommendations from the rules-based
ranker, with a full per-candidate score breakdown so you can see exactly which
signal pushed each match to the top.

- **Lane A — mutual friends.** Friends-of-friends always rank first.
- **Lane B — shared background.** Same hometown or same college.
- **Lane C — everyone else.** Ordered by weighted compatibility.

Source code: <https://github.com/ai-bryguy101/hangpost-app>
