---
title: Hangpost Matching Engine Demo
emoji: 👥
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
short_description: Pick a profile, see who the ranker recommends and why.
---

# Hangpost Matching Engine — Live Demo

Pick a source profile and see the top-10 recommendations from the rules-based
ranker, with a full per-candidate score breakdown so you can see exactly which
signal pushed each match to the top.

The ranker uses a **six-tier lexicographic sort** — candidates with mutual
friends always outrank candidates without them, and within each lane the
weighted compatibility score (hobbies, age closeness, semantic similarity)
decides ordering:

- 🌟 **Tier 1** — mutual friend + same hometown **and** same college
- 🟢 **Tier 2** — mutual friend + same hometown **or** same college
- 🟢 **Tier 3** — mutual friend, no background overlap
- 🔵 **Tier 4** — no mutual friend + same hometown **and** same college
- 🔵 **Tier 5** — no mutual friend + same hometown **or** same college
- ⚪ **Tier 6** — everyone else (ordered by weighted compatibility)

Source code: <https://github.com/ai-bryguy101/hangpost-app>
