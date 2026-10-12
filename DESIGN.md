---
name: Dragonwing Doorbell
description: An edge face-recognition doorbell that stands vigil in indigo and renders its verdict in light.
colors:
  indigo: "#31017D"
  indigo-deep: "#23015A"
  indigo-lift: "#420A96"
  violet: "#5B0CD9"
  violet-dim: "#461E96"
  violet-glow: "#AA78F5"
  ink: "#FFF0F5"
  mist: "#CD9BBE"
  unlock-green: "#5ADC78"
  lock-red: "#EB4646"
  scan-amber: "#3CBEEB"
typography:
  display:
    fontFamily: "FONT_HERSHEY_SIMPLEX (OpenCV vector stroke font)"
    fontSize: "scale 4.6-5.2 (recognition banner, ~140-170px cap height)"
    fontWeight: 2
    lineHeight: 1
    letterSpacing: "normal"
  wordmark:
    fontFamily: "FONT_HERSHEY_SIMPLEX"
    fontSize: "scale 0.95-1.1 (~30-36px)"
    fontWeight: 2
    lineHeight: 1
    letterSpacing: "0.3em (DRAGONWING) / wide (D O O R B E L L)"
  title:
    fontFamily: "FONT_HERSHEY_SIMPLEX"
    fontSize: "scale 1.0-1.2 (~34-40px)"
    fontWeight: 2
    lineHeight: 1
    letterSpacing: "normal"
  label:
    fontFamily: "FONT_HERSHEY_PLAIN"
    fontSize: "scale 1.0-1.1 (~16-20px)"
    fontWeight: 1
    lineHeight: 1
    letterSpacing: "normal"
  mono:
    fontFamily: "FONT_HERSHEY_PLAIN (telemetry / HUD readouts)"
    fontSize: "scale 0.75-1.1 (~12-20px)"
    fontWeight: 1
    lineHeight: 1
    letterSpacing: "normal"
components:
  banner-unlock:
    backgroundColor: "{colors.indigo-deep}"
    textColor: "{colors.unlock-green}"
    typography: "{typography.display}"
    padding: "48px 64px"
  banner-lock:
    backgroundColor: "{colors.indigo-deep}"
    textColor: "{colors.lock-red}"
    typography: "{typography.display}"
    padding: "48px 64px"
  corner-bracket:
    textColor: "{colors.violet}"
    size: "3px stroke, ~48px arms"
  stats-panel:
    backgroundColor: "{colors.indigo-deep}"
    textColor: "{colors.mist}"
    typography: "{typography.mono}"
    padding: "14px 18px"
  hud-bar:
    backgroundColor: "{colors.indigo-deep}"
    textColor: "{colors.mist}"
    typography: "{typography.mono}"
    height: "26px"
---

# Design System: Dragonwing Doorbell

## Overview

**Creative North Star: "The Threshold Sentinel"**

This is the face of a machine standing guard at a doorway. At rest it holds a calm, deep-indigo
vigilance — the Dragonwing mark glowing like a beacon over a slow, breathing field of violet
contour-waves — and the instant a face resolves it snaps to the certainty of a verdict: green for
*known*, red for *denied*. The whole system is engineered around that single dramatic moment of
recognition. Everything before it is patient atmosphere; the verdict is loud, immediate, and
unmistakable from across a trade-show floor.

The aesthetic is engineered and confident, not decorative. It is built to be read at distance on an
HDMI panel: bold, solid forms over thin fussy ones, high-contrast light type on indigo, and a strict
separation between *brand chrome* (indigo and violet — the identity) and *safety signal* (green, red,
amber — the result). The brand sets the stage; it never speaks the verdict. Because the canvas is
drawn with OpenCV vector-stroke primitives rather than CSS, every form here is a deliberate geometric
construction — gradients, polylines, filled ellipses — and the design language is expressed in stroke
weight, tonal indigo layering, and luminous accents rather than in web conventions.

Depth is tonal, never shadowed in the drop-shadow sense: a vertical indigo gradient grounds every
screen, panels lift via brighter indigo fills and translucency, and the wave field glows behind. The
one concession to pure flatness is a legibility halo — a soft dark indigo pad behind critical text and
the mark — so the message survives over a live camera feed or a busy wave field.

**Key Characteristics:**
- Indigo is the ground of every screen; nothing sits on pure black or gray.
- The brand palette and the access-result palette never trade places.
- Calm atmosphere at rest; a loud, decisive verdict on recognition.
- Bold and solid over thin and precise — built for across-the-room legibility.
- Every mark is a drawn vector construction, ASCII-only type, no bitmaps or Unicode glyphs.

## Colors

A two-world palette: an indigo-and-violet brand world that owns all chrome and atmosphere, and a
three-signal safety world (green/red/amber) reserved exclusively for the access verdict and live load.

### Primary
- **Midnight Indigo** (#31017D): The ground of the entire product — the base of every gradient, the
  fill behind panels and banners. The official Qualcomm Dragonwing indigo. If a surface has no other
  instruction, it is this color.
- **Dragon Violet** (#5B0CD9): The primary brand accent and chrome — corner brackets at rest, section
  rules, the active-backend highlight, the wave-field strokes, the glowing edge of the mark. The
  official Dragonwing violet.

### Secondary
- **Lifted Indigo** (#420A96): The top of the vertical gradient and the fill of lifted panels — how a
  surface reads as *raised* without a shadow.
- **Deep Indigo** (#23015A): The gradient base and the translucent pad behind panels, banners, and
  legibility halos — how a surface reads as *recessed* or *grounded*.
- **Violet Glow** (#AA78F5): Bright highlight for the mark's pulsing beacon and hover/active emphasis.
- **Dim Violet** (#461E96): Inactive lines, secondary rules, the keyboard-hint footer — violet turned
  down so it recedes without going gray.

### Neutral
- **Ink** (#FFF0F5): Near-white primary text and the wordmark. Faintly warm (a whisper of the violet
  hue), never a clinical pure white.
- **Mist** (#CD9BBE): Secondary text and telemetry labels. A violet-tinted gray — **tinted from the
  brand hue on purpose, never a neutral gray**.

### The Access Signal (reserved)
- **Signal Green** (#5ADC78): *Known / UNLOCKED* only. The affirmative verdict.
- **Alarm Red** (#EB4646): *Unknown / ACCESS DENIED* only. The negative verdict.
- **Scanning Amber** (#3CBEEB): *Recognizing / in-progress* only — a cyan-leaning amber chosen to stay
  legible against indigo. (Also the top of the per-core load meter, green→red.)

### Named Rules
**The Two-World Rule.** Indigo and violet own identity and chrome; green, red, and amber own the
verdict and live load. Neither world ever borrows the other's color. A violet "unlocked" state or a
green decorative frame is a defect, not a variation.

**The No-Gray Rule.** There are no neutral grays in this system. Every "gray" is Mist — violet tinted.
Secondary text, dividers, and dim lines all pull from the brand hue so the screen never reads generic.

## Typography

**Display / Title / Wordmark Font:** OpenCV `FONT_HERSHEY_SIMPLEX` — a clean single-stroke vector
sans, the closest canvas analog to the rounded humanist sans of the official Dragonwing wordmark.
**Telemetry / HUD / Label Font:** OpenCV `FONT_HERSHEY_PLAIN` — a lighter, denser stroke for dense
machine readouts.

**Character:** Engineered and legible. Because these are stroke fonts rendered with anti-aliasing on a
NumPy canvas, weight is carried by stroke *thickness* (1-2px) rather than a font-weight axis, and
emphasis by *letter-spacing* and *size*. **Hershey fonts are ASCII-only** — no Unicode punctuation,
arrows, or glyphs ever (they render as `?`); decorative marks are drawn, not typed.

### Hierarchy
- **Display** (thickness 2, scale 4.6-5.2, ~140-170px): the recognition verdict banner ("UNLOCKED",
  "ACCESS DENIED"). The single largest element on any screen; pulses subtly on appearance.
- **Wordmark** (thickness 2, scale ~1.0, letter-spaced): "DRAGONWING" over tracked "D O O R B E L L" —
  the brand lockup on loading and standby.
- **Title** (thickness 2, scale 1.0-1.2, ~34-40px): screen headlines ("DOORBELL READY", enrollment
  name).
- **Label** (thickness 1, scale 1.0-1.1, ~16-20px): status strings, progress counts, badges.
- **Mono** (thickness 1, scale 0.75-1.1, ~12-20px): telemetry panel, HUD readouts, FPS, per-core load,
  the keyboard-hint footer.

### Named Rules
**The Tracked-Caps Rule.** Brand lockups and key labels are uppercase with deliberate letter-spacing
(`put_tracked`). The spacing *is* the refinement — it signals brand chrome versus raw machine text.

**The ASCII-Only Rule.** Every string is 7-bit ASCII. A dot separator is a space or a drawn glyph; an
arrow is a drawn triangle (`fillConvexPoly`). Never a Unicode character in a Hershey string.

## Layout

A fixed **1280×720** full-screen canvas (16:9 HDMI), composed as horizontal bands. A top HUD bar
(~26px: state name left, status center, FPS badge right) and a bottom keyboard-hint strip frame a
central stage. Full-screen brand states (loading, standby, QR) center their content on the vertical
axis — mark above, wordmark/title below — over the gradient-and-wave ground. Camera states letterbox
the live feed into the indigo ground, overlay violet corner brackets, and dock the telemetry panel to
a lower corner. The inference frame is a small 320×240 working size scaled up for display. Spacing is
generous and centered; this is a display to be read, not a dense dashboard to be operated up close.

## Elevation & Depth

Tonal, with legibility halos — no drop shadows. Depth is built from a vertical indigo gradient
(Lifted Indigo at top → Deep Indigo at base), brighter-indigo panel fills that read as raised, and
translucent Deep-Indigo pads that read as grounded. Behind the brand states, the violet contour-wave
field glows and drifts, adding atmospheric depth the way a light haze would.

### Named Rules
**The Legibility-Halo Rule.** Critical text and the mark get a soft Deep-Indigo pad behind them when
they sit over the camera feed or the wave field — the one permitted "shadow", and it exists for
contrast, never for decoration.

**The Flat-Verdict Rule.** The access banner is a flat, bold block — no gradient, no glow, no halo
games on the verdict itself. Its certainty is conveyed by pure solid color and size.

## Shapes

The form language is a tension between **organic brand curves** and **rectilinear machine chrome**. The
Dragonwing mark and the wave field are all sweeping curves — filled ellipses, convex polygons, and
smooth polylines, built from the two interlocking swooshes and the negative-space crescent eye. The UI
chrome around them is strictly rectilinear and sharp-cornered: square-cut corner brackets (3px arms),
straight section rules, rectangular telemetry bars and panels. **No rounded-rectangle cards.** The
curves belong to the brand; the straight lines belong to the instrument. The mark reads as a near-
circular emblem (crest + crescent eye + body + flared tail curl) and is always a drawn vector
construction scaled from a center point, never a pasted bitmap.

## Components

### Access Banner (signature component)
- **Character:** The loud verdict — the whole system's climax.
- **Shape:** Full-width flat block, square corners, Deep-Indigo fill.
- **Unlock:** Signal Green display text ("UNLOCKED <name>"). **Lock:** Alarm Red ("ACCESS DENIED").
- **Behavior:** Display-scale type pulses subtly on appearance; no gradient or glow on the block.

### Corner Brackets
- **Shape:** Four L-shaped brackets, 3px stroke, ~48px arms, framing the detected-face / live-feed region.
- **Color:** Dragon Violet at rest (neutral), recolored to the state's signal color — Scanning Amber
  while recognizing, Signal Green / Alarm Red on verdict.

### Dragon Mark
- **Character:** The beacon — the brand's living presence.
- **Construction:** Drawn vector glyph (filled ellipses + convex polygons) with the negative-space
  crescent eye; the eye is load-bearing and must always read.
- **States:** Centered hero crop on loading/standby with a Violet-Glow pulsing halo; small lockup crop
  beside the wordmark elsewhere.

### Telemetry Panel
- **Shape:** Rectangular, square corners, translucent Deep-Indigo fill, docked to a lower corner.
- **Content:** Mono type in Mist; per-core CPU meter bars (green→red load gradient — a *meter*, not
  brand), RAM readout, active-backend row highlighted in Dragon Violet with a drawn triangle marker.

### HUD Bar
- **Shape:** Thin top band (~26px), Deep-Indigo, violet baseline rule.
- **Content:** State name (left), status (center) in Mist mono; FPS badge (right) — **Dragon Violet
  when NPU, Mist when CPU**, so the active backend is readable at a glance.

### Wave Field
- **Character:** The breathing atmosphere behind brand states.
- **Construction:** `cv2.polylines` contour strokes in Dim/Dragon Violet over the indigo gradient,
  drifting slowly with an animation clock. Thin lines; always behind content; never competes with the
  mark or the verdict.

## Do's and Don'ts

### Do:
- **Do** ground every screen in Midnight Indigo (#31017D) — gradient, panel, or flat fill.
- **Do** reserve Signal Green / Alarm Red / Scanning Amber exclusively for the access verdict and the
  load meter (the Two-World Rule).
- **Do** make the verdict the loudest thing on screen — display scale, flat bold block.
- **Do** tint every "gray" from the violet hue (Mist), never a neutral gray (the No-Gray Rule).
- **Do** draw the mark and all decorative marks as vector constructions; keep the crescent eye readable.
- **Do** letter-space brand lockups and keep all strings 7-bit ASCII.
- **Do** color the FPS badge violet on NPU / mist on CPU so the backend is legible at a glance.

### Don't:
- **Don't** let indigo or violet touch the verdict, or let green/red/amber become decorative chrome.
- **Don't** use a neutral gray anywhere.
- **Don't** put a drop shadow on anything; depth is tonal plus the one legibility halo.
- **Don't** use rounded-rectangle cards — chrome is sharp-cornered; only the brand curves are round.
- **Don't** let the wave field compete with the mark or the verdict — it stays thin and behind.
- **Don't** emit any Unicode glyph in a Hershey string (dots, arrows, bullets render as `?`).
- **Don't** drift toward a generic dark-dashboard look — gray panels, flat default charts, untinted type.
