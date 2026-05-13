# Hangpost — Product Vision

> **Why this file exists.** Every other doc in this repo (`CLAUDE.md`,
> `README.md`, the model/data cards) describes the *matching engine*. This
> file describes the *app the engine is for*. Read it first if you've never
> seen Hangpost before — the design choices in `scoring.py`,
> `embeddings.py`, and the radius pre-filter only make sense once you know
> what the product is trying to do for users.

---

## One-sentence pitch

**Hangpost is a social media app for finding new friends in your city —
the app you download when you move somewhere new.**

It looks and feels like social media, but instead of broadcasting to
friends you already have around the world, every post is location-scoped
to your current city and visible to nearby strangers you're likely to
get along with.

---

## The problem we're solving

Making friends as an adult — especially after moving to a new city — is
hard. Existing platforms are the wrong shape for it:

- **Instagram / Facebook / X** are optimized for keeping up with people
  you already know. The graph is global and historical, not local and
  forward-looking.
- **Dating apps** are romantic by default, which warps the interaction.
- **Meetup / Bumble BFF / Hinge "friend mode"** require a hard "let's
  meet a stranger" commitment up front. Most people won't take that step
  cold.

Hangpost sits in the gap: low-lift, location-scoped social discovery
that surfaces people you're statistically likely to actually become
friends with, and gives you an easy on-ramp (a hangout post, a comment
on a local-info post) instead of a 1:1 cold open.

---

## The core experience

### 1. A city-scoped posterboard, not a friends feed

Every post you see is from someone **physically near you right now**.
Your feed is essentially your city's bulletin board. Two kinds of posts
dominate:

- **Hangout opportunities** — "Grabbing a drink at 7 tonight, anyone in?",
  "Going to play pickup basketball Saturday at Dolores Park, need 2
  more." Posts can be **shared with linked profiles only** (good
  matches, friends-of-friends) or **open to the whole local area**.
- **Local information** — "This sports bar in the Mission is actually
  chill," "Avoid the 14 line between 5–6," "Free salsa night Thursdays."
  Content that makes a city feel less anonymous.

### 2. Matching surfaces *people you're likely to befriend*

Outside the open posterboard, the app shows you a curated set of
profiles each day — people the matching engine thinks you're a strong
candidate to actually befriend, ranked in this priority order:

1. **Friends of friends** — by far the strongest predictor of new
   friendships in the real world. Sourced from contacts + Instagram
   graph imports. This is why `mutual_friends` and the
   `friend_common_boost` lane dominate the score.
2. **Shared background** — hometown, college, where you grew up. People
   with shared origin context bond fast.
3. **Hobbies and interests** — what you actually like to do.

### 3. Low-lift hangout invitations

The atomic unit of action in Hangpost isn't "send a DM and hope they
reply." It's **post a hangout**. The poster takes on the social risk
once; everyone else just taps "I'm in." That asymmetry is the whole
product. Match quality matters because the hangout post is shown to
matched profiles first.

---

## Why "location-based" means two different things

This is the single most common point of confusion when someone (human or
AI) reads `scoring.py`. The repo uses "location" to mean two structurally
different things:

| | **Current location** (GPS) | **Hometown** |
|---|---|---|
| What it is | Where you are *right now* | Where you grew up |
| Role in the system | **Hard pre-filter.** Profiles outside the radius are removed before the ranker ever sees them. | **Soft ranking signal.** Same hometown = friendship cue, gets a positive weight. |
| Where it lives in code | Upstream of the matching engine (DB / geo-index query) | A feature inside `compute_match_score`, currently stored in `UserProfile.location` |

**The matching engine never sees GPS coordinates and never computes
Haversine distance.** By the time `rank_candidates` runs, the radius
filter has already done its job. The ranker's only job is "given these
already-nearby people, which ones is this user most likely to befriend?"

If you find yourself wanting to add a `distance_km` feature to the
ranker, stop — that belongs in the candidate-retrieval layer instead.

---

## Why the engine ranks the way it does

A summary of the design choices for someone reading the code cold:

- **Friends-of-friends dominates.** "Mutual friends" gets its own
  separate boost lane on top of the weighted score — socially-connected
  candidates are always shown before unconnected ones, regardless of how
  high a stranger's compatibility score is. This mirrors the real-world
  finding that friend-of-friend is the #1 path to new friendships.
- **Hometown / shared background ranks above generic interests.**
  Shared origin context is a stronger friendship predictor than
  overlapping hobby lists, and the weights reflect that.
- **Interests and liked topics use Jaccard overlap, not just counts.**
  Two people who both list 30 hobbies and share 3 are less similar than
  two people who list 5 hobbies and share 3 — Jaccard handles that
  natively.
- **Age compatibility uses a step-down ladder, not a continuous decay.**
  Easy to explain in the UI ("you're within 3 years"), easy to tune,
  product-appropriate.
- **Semantic similarity (Phase 2) embeds a synthesized text, not a bio.**
  Hangpost users don't write bios — they fill in structured fields
  (interests, hometown, age). `profile_to_text` assembles those into a
  natural-language sentence that the embedder turns into a vector. This
  way every user gets a useful semantic representation without typing
  anything extra.
- **The breakdown is explainable on purpose.** `MatchBreakdown` carries
  every component score so the product can eventually surface "you
  matched because: 2 mutual friends, same hometown, 4 shared interests"
  — transparency builds trust in a social product where users are
  trusting the algorithm with who they meet IRL.

---

## What this engine is *not*

- Not a dating-app matcher. No romantic-intent modeling.
- Not a content-recommendation engine. We rank **people**, not posts.
  (Post ranking is a separate, downstream concern.)
- Not a global graph. Everything is scoped to "people physically near
  you right now."
- Not a black box. Every score is decomposable into named components.

---

## Roadmap shape (product, not ML)

The ML roadmap lives in `CLAUDE.md` (Phase 1 rules → Phase 2 embeddings
→ Phase 3 learned ranker). The product roadmap on top of that ranker
looks roughly like:

1. **Profile + city posterboard MVP.** Structured profile capture,
   radius-filtered feed, basic hangout-post + local-info-post types.
2. **Friend-graph imports.** Contacts + Instagram → populate
   `mutual_friend_ids`. This is what unlocks the "friends of friends"
   signal in the real product, not just in evaluation data.
3. **Matched daily picks surface.** A small set of high-ranked profiles
   per day, with the `MatchBreakdown` explanation visible.
4. **Hangout-post targeting.** Let posters target "matched profiles only,"
   "friends of friends," or "open to the whole local area." The ranker
   picks the audience for the first two.
5. **Outcome instrumentation.** Log accept/decline, chats started,
   hangouts actually attended. These are the labels Phase 3 has been
   waiting for.
6. **Re-train Phase 3 on real labels.** Replace synthetic labels with
   measured outcomes; treat the synthetic generators as offline-only
   diagnostics.

---

## TL;DR for future agents reading this repo

If you only remember three things:

1. **The app helps people make new friends in their current city** —
   especially after moving. It looks like social media but the audience
   is "nearby strangers you'd probably like," not "people you already
   know."
2. **GPS distance is a hard filter, not a feature.** Hometown is a
   feature. Don't conflate them.
3. **Friends-of-friends > shared background > shared hobbies.** That
   priority order is intentional and product-driven, not arbitrary.
