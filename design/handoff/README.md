# Handoff: Choreo App — Mobile Redesign

## Overview
An iPhone-portrait redesign of the Choreo App (a choreography practice tool) applying an Apple-minimalist aesthetic. Covers two screens: **Library** (song list) and **Practice/Song** (video + Timeline HUD with waveform, A/B loop, speed, stem/onset selection, tap-counting).

## About the Design Files
The bundled `.html` files are **design references built as interactive HTML prototypes** — they show intended look, layout, and behavior, not production code to copy verbatim. The task is to **recreate these designs inside the existing Choreo App codebase** (`Neel-Karkhanis/Choreo-App`, React + TypeScript, `frontend/src`), reusing its existing component structure, state, and data (Timeline.tsx, Hud.tsx, controls.tsx, Library.tsx, types.ts, styles.ts, index.css) rather than introducing the prototype's inline styles wholesale.

## Fidelity
**High-fidelity.** Colors, typography, spacing, and interactions below are final; implement pixel-accurate.

## Source Repository
- repo: `Neel-Karkhanis/Choreo-App`, branch `main`, path `frontend/src`
- Screen → repo file mapping:
  - Library (song list) → `Library.tsx`, `index.css`
  - Practice/Song (video + HUD, loop, speed, timeline) → `Hud.tsx`, `controls.tsx`, `styles.ts`, `Timeline.tsx`, `types.ts`, `index.css`

## Screens / Views

### 1. Library
- **Purpose**: browse songs, jump into practice, see which songs still need beat-counting.
- **Layout**: full-width column, 85% container width centered. Header row: logo + "horeo" wordmark on left, "Import song" pill button on right (`justify-content: space-between`). Below, a vertical list of song rows, `gap: 15px` between rows... actually rows are separated by a bottom border, not gap — each row is `display:flex; align-items:center; gap:12px; padding:9px 6px`, with a `1.15px` bottom border in a purple/grey color mix (`color-mix(in oklab, accent 45%, cardBorder)`).
- **Components**:
  - **Logo**: 30×30 SVG octagon outline (two concentric octagon paths, `stroke-width:8`, accent color, `fill:none`) with 6 short radial "seam" divider lines connecting the two octagons at each vertex pair, plus wordmark "horeo" (the visible "C" is implied by the octagon shape itself), 21px/700 weight, `letter-spacing:-0.02em`.
  - **Import button**: dark pill, `background: oklch(0.22 0.01 90)` (near-black), white text, `padding:8px 14px`, `border-radius:980px` (full pill), 12px/600.
  - **Song row**: status dot (7×7px circle, green `oklch(0.6 0.13 155)` = counted, yellow `oklch(0.65 0.1 80)` = needs counting) → title (14px/600) + optional note ("Needs counting", 10px/600, `letter-spacing:0.02em`, color `oklch(0.5 0.1 80)`, amber/yellow) → duration (12px, muted, tabular-nums, right-aligned via `flex:1` on title block).
  - Row is fully clickable (`onClick` → opens that song in Practice view).

### 2. Practice / Song
- **Layout**: header with back arrow (←), song title (15px/700), logo icon (right). Below, a segmented tab control (Timeline / Video), centered, pill-shaped chip background. Bottom-fixed tab bar (Library / [current song name]) floating pill, `position:absolute; bottom:24px`, centered, drop shadow.
- **Video tab**:
  - 16:9 video preview placeholder, dark background (`oklch(0.15 0.005 90)`), rounded 12px corners, "video preview" label.
  - Scrubber bar overlay at bottom of video: track (`oklch(1 0 0 / 20%)`), filled progress in accent color, white playhead dot, plus **A/B loop pins** — green circle "A" (`oklch(0.6 0.18 145)`) and yellow circle "B" (`oklch(0.8 0.18 95)`), both 14px circles with drop shadow.
  - Fullscreen button, top-right of overlay area.
  - Playback row below video: play button (circular, bordered), time readout ("0:18 / 4:25", tabular-nums), **speed dial** button (clock-with-hand icon, toggles active/accent state), and a "Hear" stem-mode `<select>` (All/Vocals/Drums/Bass/Instrumental) right-aligned.
- **Timeline tab**:
  - Single card (`border-radius:12px`, bordered, padded 22px) containing:
    - **Waveform**: 64 vertical bars (grey `oklch(0.62 0.005 90)`), with thin colored onset ticks overlaid per-bar: blue (`oklch(0.5 0.18 260)`) = bass onset, red (`oklch(0.55 0.2 25)`) = drum onset. A translucent accent-colored band + two accent vertical lines mark the current A/B loop region.
    - A **faint 1px full-bleed divider** (`margin:20px -22px`) separates the waveform from the minimap/controls beneath it — this is a merged single card, not two.
    - **Minimap/scrub strip**: thin horizontal track showing loop region, with the same green "A" / yellow "B" pin circles as the video overlay, positioned proportionally.
    - **Controls row**: play button, time readout, "Onsets" `<select>` (Off/Bass/Drums/Both, colored option text matching bass/drum colors), "Hear" stem `<select>` (same options as Video tab), speed dial button below-left (intentionally offset via negative margin to sit under the row).
  - **Tap-counting panel** (Timeline tab only): appears below the card when active. Accent-bordered card, tinted background (`color-mix(accent 6%, cardBg)`), instruction text "Tap on the 1, then every count you feel.", a progress bar (0/16 min counted), and a "Begin"/"Accept" toggle button (accent-filled when idle, grey when tapping started).

## Interactions & Behavior
- **Library → Song**: click any song row, or the bottom "[song name]" tab, to open Practice view for that song. Back arrow or "Library" tab returns.
- **Timeline ↔ Video**: segmented control swaps between the two Practice sub-views; state persists per session (not per-song in the prototype).
- **Speed dial**: click toggles an active/inactive visual state (in the real app this would open a speed picker — prototype only mocks the toggle, not the picker UI).
- **Tap-counting**: "Begin" starts tap-counting mode (button becomes "Accept", grey); tapping the button while active would (in the real implementation) accept the counted taps and save them, clearing the "Needs counting" note in Library.
- **Onset/Stem selects**: plain native `<select>` dropdowns; Onset options are color-coded to match the bass (blue) / drums (red) onset tick colors used in the waveform.
- **Dark mode**: global light/dark theme toggle swaps background, card, border, text, and input colors (see Design Tokens below for both variants). Accent color stays constant across modes.
- No animations beyond native `<select>`/button defaults; hover/active states were not designed beyond `cursor:pointer` — add standard press/hover feedback consistent with the rest of the app.

## State Management
- `view`: `'library' | 'song'` — which top-level screen is shown.
- `songScreen`: `'timeline' | 'video'` — active Practice sub-tab.
- `tapStarted`: boolean — tap-counting session active/accepted.
- `showSpeed`: boolean — speed-dial active visual state (prototype-only; wire to real speed-picker state in the app).
- `darkMode`: boolean — theme toggle.
- `accent`: string (hex) — accent color, tweakable; default purple `#9333ea`.
- Real app additionally needs: current song id/data, playback position, loop A/B points, stem selection, onset mode, count/tap data — all present in the existing Choreo App codebase (`Timeline.tsx`, `types.ts`, `Hud.tsx`) and should be wired to this new layout rather than re-invented.

## Design Tokens

### Colors — Light mode
| Token | Value |
|---|---|
| Page background | `oklch(0.98 0.002 90)` |
| Text | `oklch(0.2 0.005 90)` |
| Card background | `white` |
| Card border | `oklch(0.91 0.003 90)` |
| Chip/tab-bar background | `oklch(0.94 0.003 90)` |
| Input background | `white` |
| Input border | `oklch(0.85 0.003 90)` |
| Muted text | `oklch(0.55 0.005 90)` |
| Divider | `oklch(0.88 0.003 90)` |

### Colors — Dark mode
| Token | Value |
|---|---|
| Page background | `oklch(0.18 0.006 290)` |
| Text | `oklch(0.95 0.004 90)` |
| Card background | `oklch(0.24 0.008 290)` |
| Card border | `oklch(0.34 0.012 290)` |
| Chip/tab-bar background | `oklch(0.28 0.01 290)` |
| Input background | `oklch(0.26 0.009 290)` |
| Input border | `oklch(0.4 0.012 290)` |
| Muted text | `oklch(0.68 0.008 90)` |
| Divider | `oklch(0.36 0.012 290)` |

### Colors — fixed (both modes)
| Token | Value |
|---|---|
| Accent (purple, tweakable) | `#9333ea` |
| Status dot — counted | `oklch(0.6 0.13 155)` (green) |
| Status dot — needs counting | `oklch(0.65 0.1 80)` (yellow) |
| "Needs counting" note text | `oklch(0.5 0.1 80)` |
| Loop pin A | `oklch(0.6 0.18 145)` (green), white "A" label |
| Loop pin B | `oklch(0.8 0.18 95)` (yellow), white "B" label |
| Bass onset tick | `oklch(0.5 0.18 260)` (blue) |
| Drum onset tick | `oklch(0.55 0.2 25)` (red) |
| Import button background | `oklch(0.22 0.01 90)` (near-black), white text |

### Typography
- Font stack: `-apple-system, BlinkMacSystemFont, 'SF Pro Display', Helvetica, Arial, sans-serif`
- Wordmark "horeo": 21px / 700 / `letter-spacing:-0.02em`
- Song title (header): 15px / 700 / `letter-spacing:-0.01em`
- Song row title: 14px / 600
- Song row note / duration: 10–12px / 600, tabular-nums for numeric time values
- Tab labels, buttons: 11–12px / 600

### Spacing / Shape
- Device frame: 407×656px (iPhone-portrait mock)
- Page padding: `52px 18px 18px` (top clears status bar)
- Library container width: 85%, `margin: 25px auto 0`
- Card border radius: 12px; pill buttons: `border-radius:980px` or `9999px`-equivalent full pill; input/select radius: 8px
- Song row padding: `9px 6px`; row divider: `1.15px solid`
- Timeline card padding: 22px; internal divider margin: `20px -22px` (full-bleed within padded card)
- Loop pin size: 14px circle; scrubber playhead: 11px circle; status dot: 7px circle

## Assets
- Logo: hand-drawn inline SVG (two concentric octagon `<path>` outlines + 6 short seam lines connecting their vertices), no external asset file. Reproduce as SVG or export to an icon asset in the app's asset pipeline — coordinates are in the DC file's `dividerPairs` array and octagon path `d` attributes.
- No photos/icons beyond this logo and inline Unicode/SVG glyphs (←, ▶, speed-dial clock icon, fullscreen corners icon).

## Files
- `Choreo Redesign.dc.html` — full interactive prototype (Library + Practice/Timeline/Video views, dark mode toggle, tap-counting panel). Open directly in a browser.
- `ios-frame.jsx` — iPhone device bezel used only for prototype presentation; not part of the design itself, skip when implementing in the app.
