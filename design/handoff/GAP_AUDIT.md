# Gap Audit: Design Handoff vs. Current Frontend

Read-only audit. No code changes. Sources: `design/handoff/README.md`,
`design/handoff/Choreo Redesign.dc.html`, and every file under `frontend/src`
(main.tsx, App.tsx, Library.tsx, Song.tsx, Timeline.tsx, TimelineOverview.tsx,
VideoScreen.tsx, VideoScrubber.tsx, Hud.tsx, controls.tsx, LoopBoundaryHandle.tsx,
TapOverlay.tsx, tapSession.ts, tapGrid.ts, tapGrid.test.ts, manualGrid.ts,
eightCount.ts, snap.ts, playback.ts, stemEngine.ts, videoSync.ts, videotest.ts,
dropdown.ts, styles.ts, format.ts, types.ts, index.css).

`videotest.ts` is a standalone A/V-sync measurement harness (own DOM ids,
no import from `App.tsx`/`main.tsx`) — not a rendered app screen, so it is
out of scope for design comparison and not discussed further below.

---

## 1. Design → Code Map

### Library screen

| Design element | Current file / component | Match? |
|---|---|---|
| 85%-width centered container, header row (logo+wordmark left, Import pill right) | `Library.tsx` `<main className="library">` / `.library-head` (`index.css:287-303`) | Layout present but not the 85%-centered/pill-button treatment — plain flex header, default button styling |
| Logo: 30×30 octagon-outline SVG + wordmark "horeo" | **No corresponding component.** `Library.tsx:110` renders plain `<h1>Choreo</h1>`, no SVG, no logo anywhere in the app | **Missing** |
| Import button (dark pill, near-black bg) | `Library.tsx:119-121` `<button>` — unstyled default button, no pill/near-black treatment in `index.css` | Present, unstyled |
| Song row: 7×7 status dot (green/yellow) | **No dot rendered.** State is instead carried by `.song-row` left border color + a `.song-state` text badge (`Library.tsx:145`, `index.css:320-330,365-378`) | **Different mechanism**, see §3 |
| Song row title (14px/600) | `Library.tsx:135-144` `.song-open` button (`index.css:339-348`, 1rem/600, blue link color) | Present, different visual treatment |
| "Needs counting" note (10px amber) | `Library.tsx:23-54` `<StateNote>` — much longer prose per state, 3 states not 2 | Present but richer/different, see §2, §3 |
| Duration (12px muted, tabular-nums, right-aligned) | `Library.tsx:147-149` `.song-duration` (`index.css:388-393`) | Match |
| Row divider only (1.15px, purple-tinted), no other border | `Library.tsx:133` `.song-row` (`index.css:311-316`) — full box border + 4px left accent border, not a bottom-only divider | Present, different treatment |

### Practice / Song screen

| Design element | Current file / component | Match? |
|---|---|---|
| Header: back arrow, song title, logo icon | `Song.tsx:148-151` back button + `<h1>{title}</h1>`; no logo icon | Partial — no logo |
| Segmented Timeline/Video tab (pill/chip, centered) | `Song.tsx:265-278` `.screen-tabs`/`.screen-tab` (`index.css:409-430`) — top tab-strip with bottom border, not a centered pill | Present, different visual treatment; also conditionally rendered, see §3 |
| Bottom-fixed floating Library/[song] tab bar | **No corresponding component anywhere in the app.** | **Missing** |
| Video 16:9 placeholder ("video preview" label) | `VideoScreen.tsx:52-67` — renders a **real** `<video>` element (`.video-pane`) when a URL exists, or `.video-pane-empty` ("no video for this track") only when there is none | Different semantics: design shows a placeholder always; code shows real video or a real-absence message |
| Scrubber overlaid on bottom of video, with A/B pins | `VideoScrubber.tsx` — rendered as a **separate bar below** the video element (`VideoScreen.tsx:55-61`), not overlaid on it | **Positioned differently** |
| Fullscreen button, top-right of video overlay | **No corresponding component.** | **Missing** |
| Play button (circular bordered), time, speed-dial (clock icon toggle), Hear select — video tab | `Hud.tsx:44-51` → `PlayPauseButton`, `TimeReadout`, `HearControl`, `SpeedControl` (`controls.tsx`) | Present; PlayPauseButton is a plain text button, not circular; SpeedControl is a full slider dropdown, not a toggle icon; HearControl is a custom dropdown, not a native `<select>` — see §3 |
| Timeline tab: single bordered/padded card | `Timeline.tsx:691-693` `<div className="timeline"><div ref={containerRef} /></div>` — **no such card styling exists in `index.css`**; no border/padding/radius wrapper at all | **Missing visual container** |
| Waveform: 64 grey bars + blue/red per-bar onset ticks | `Timeline.tsx:195-212` (`useWavesurfer`) renders a continuous wavesurfer.js peak path (not fixed-64-bar) plus onset-tick regions added at `Timeline.tsx:414-439` | Present, conceptually equivalent, structurally very different — see §2, §4 |
| Translucent accent band + 2 lines marking A/B region on waveform | `Timeline.tsx:631-671` (loop wash + edges via `RegionsPlugin`) | Match (conceptually) |
| Full-bleed 1px divider between waveform and minimap | **No corresponding element** — no divider in `index.css` or `Timeline.tsx` between the wavesurfer container and `TimelineOverview` | **Missing** |
| Minimap/scrub strip with A/B pins | `TimelineOverview.tsx` — explicitly **not** a "minimap": no waveform, no 8-count shading, no read-only viewport box (see its own doc comment, `TimelineOverview.tsx:6-13`) | Present as a differently-scoped component — see §4 conflict #1 |
| Controls row: play, time, "Onsets" select (Off/Bass/Drums/Both) | `Timeline.tsx:702-735` — "Onset ▾" dropdown containing two independent Bass/Drums toggle buttons, not a 4-way select | **Different state shape** — see §3 |
| "Hear" stem select | `HearControl` (`controls.tsx:157-198`) — custom dropdown of radio-style buttons, same 5 options | Present, different DOM |
| Speed dial, offset below-left of controls row | `SpeedControl` (`controls.tsx:99-143`) — inline in the controls row, not offset | Present, different layout/behavior — see §3 |
| Tap-counting panel: instruction text, 0/16 progress bar, Begin/Accept toggle | `TapOverlay.tsx` — full tap pad w/ live 1..8 count-in-phrase, progress to 32 (`RECOMMENDED_TAPS`), BPM/kept/dropped stats, error messages, Accept/Cancel/Start-over | Present, vastly more capable — see §2, §4 conflict #6 |

**Flagged as having no corresponding component at all:** logo SVG (Library
header + Song header), bottom floating Library/song tab bar, fullscreen
button on the video overlay, the Timeline-tab's outer bordered card, and the
full-bleed divider between waveform and the strip beneath it.

---

## 2. Implemented but Absent from the Design

Exhaustive list of things the app currently renders/holds as state that the
design does not depict at all.

1. **8-count alternating shading.** `Timeline.tsx:137-159`
   (`addEightCountShading`), invoked at `Timeline.tsx:351`. Alternating grey
   regions (`rgba(110,110,110,0.1)`) added via `RegionsPlugin` across the
   whole waveform. The design's waveform has no such banding at all.

2. **Downbeat emphasis.** `Timeline.tsx:383-403` — the `beats.forEach` pulse-tick
   loop; downbeats (`downbeats.has(i)`) render taller (55% vs 25% height) and
   darker (`rgba(40,40,40,0.85)` vs `rgba(110,110,110,0.55)`). The design has
   no beat-tick layer at all, so no downbeat/beat distinction exists there.

3. **Active-8-count highlight with 1..8 count labels.** `Timeline.tsx:479-577`
   ("Layer 6") — yellow wash + border (`ACTIVE_BLOCK_FILL`/`ACTIVE_BLOCK_BORDER`,
   `Timeline.tsx:79-80`) over the 8-count window under the playhead, with
   per-beat numeric labels (`COUNT_LABEL_STYLE`, `Timeline.tsx:85-93`) and "&"
   labels between them (`Timeline.tsx:542-567`). Entirely absent from the
   design.

4. **Half-beat subdivision ticks with "&" labels.** `Timeline.tsx:266-275`
   (`subdivisions` memo) + `Timeline.tsx:366-376` (tick rendering) +
   `Timeline.tsx:542-567` ("&" label rendering). No equivalent in the design.

5. **Bookmark markers.** **Not implemented anywhere in `frontend/src`.** The
   only trace is a stale comment in `playback.ts:24` ("...the brown
   bookmarks") enumerating visual treatments — there is no bookmark state,
   store, or rendered marker in any file (main view or `TimelineOverview`).
   Not depicted in the design either.

6. **Read-only minimap viewport box.** **Not implemented.**
   `TimelineOverview.tsx:6-13`'s own doc comment states a prior "minimap"
   (its own waveform peaks, its own 8-count shading, and a read-only viewport
   box) was deliberately scrapped in favor of the current plain
   playhead+A/B bar. So neither the current code nor (in this specific
   "viewport box" sense) the design has this element — the design's
   "minimap" is a plain strip with pins, which the current `TimelineOverview`
   already approximates. Flagged here only because the terminology
   ("minimap") in `README.md:40` invites recreating the scrapped viewport-box
   version — see §4 conflict #1.

7. **Per-song(-session) snap-mode selection.** `Song.tsx:249-253` (`snapMode`
   state in `SongShell`) + `snap.ts` (`SnapMode`: `'none'|'beat'|'4-count'|
   '8-count'`) + `LoopBoundaryHandle.tsx:99-120` (tap-to-open menu on either
   A/B handle). Note this is **not actually persisted per song** — it is a
   single `useState` scoped to one open `Song` mount (a fresh mount per
   `App.tsx:24` `key={view.song.md5}`), reset every time the song is
   reopened. The design has zero representation of snap mode: no UI, no
   state field, nothing.

8. **Onset-visibility-aware tick suppression.** `Timeline.tsx:236-249`
   (`tickTags` memo) + `Timeline.tsx:357-359` (`suppressed`) — a beat's pulse
   tick is hidden when a *currently visible* bass/drum onset lands within
   `ABSORPTION_TOLERANCE_MS` (`Timeline.tsx:65`) of it. The design's mock
   onset ticks are hardcoded per-bar (`i % 9 === 3`, `i % 5 === 1`,
   `Choreo Redesign.dc.html:253-254`) and always shown; there is no
   toggle-driven suppression concept at all.

9. **Onset "Bass"/"Drums" as two independent booleans**
   (`bassVisible`/`drumsVisible`, `Timeline.tsx:183-184`), not a single select
   value — see §3 for the mismatch, listed here because the *combination
   space* (any of 4 states reachable via 2 independent clicks) has no design
   analog beyond the single `<select>`'s 4 named options.

10. **The full re-tap / tap-session feature set.** `tapSession.ts` (tapping
    open/close, live taps array, debounced fit via `TAP_SETTLE_MS`,
    preview grid) + `tapGrid.ts` (`MIN_TAPS`=16 gate, `RECOMMENDED_TAPS`=32
    nag, outlier rejection with user-facing error text, BPM readout) +
    `TapOverlay.tsx` (live 1..8 count-in-phrase on the tap pad, Space-bar
    tapping via `tapSession.ts:103-115`, Accept/Cancel vs. Start-over
    wording). The design's tap panel is a static mock: one boolean
    (`tapStarted`) that only ever flips `true→true` (see §3 item 4), a fixed
    "0/16 min" bar that never animates, and no BPM/fit/error concepts at all.

11. **Three-way library state (`ready`/`needs_tap`/`stems_evicted`)** vs. the
    design's binary dot. `types.ts:30` (`LibraryState`), `Library.tsx:17-21`
    (`STATE_LABEL`) and `Library.tsx:23-54` (`StateNote`). The
    `stems_evicted` ("File missing") state — including its two sub-messages
    depending on `grid_present` (`Library.tsx:28-38`) — has **no visual
    equivalent whatsoever** in the design (no third dot color, no note text).

12. **Video-kind badge on library rows.** `Library.tsx:146`
    (`song.media_kind === 'video' && <span className="song-kind">video</span>`).
    The design's song rows carry no indicator of whether a song has an
    associated video.

13. **Import progress/error feedback.** `Library.tsx:60-105` (`importing`
    boolean disables the button and swaps its label to "Importing…";
    `error` state renders an error paragraph, `Library.tsx:125`). The
    design's Import button has no loading or error state.

14. **Analysis/stem-loading interstitial states.** `Song.tsx:156-159` —
    "Analyzing {title}… (first run on a new track takes minutes)" and
    "Loading stems… (decoding five buffers; sizes logged to console)". No
    design equivalent; the design assumes the song is already fully present.

15. **Error surfacing throughout the Song shell.** `onsetsError`, `stemError`,
    `manual.error` (`Song.tsx:152-155`), `loop.error` (`Hud.tsx:52`,
    `Timeline.tsx:776`). No error-state concept exists in the design.

16. **Audio-only songs render a single screen (no tab control at all).**
    `Song.tsx:260` (`screens: Screen[] = mediaKind === 'video' ? ['timeline',
    'video'] : ['timeline']`). The design's segmented Timeline/Video control
    is unconditional — it has no concept of `media_kind`.

17. **Full speed-control picker: 0.25×–2× stepped slider with reset and tick
    labels.** `controls.tsx:99-143` (`SpeedControl`), `playback.ts:16-21`
    (bounds/step/default). The design's "speed dial" is a single toggle
    button only, with the picker itself explicitly out of scope
    (`README.md:47,57`: "prototype only mocks the toggle, not the picker
    UI").

18. **Loop is always-on, spans the whole track by default, and is directly
    draggable on both bars.** `playback.ts:149-176` (`useLoop` doc + impl),
    `LoopBoundaryHandle.tsx`. The design's A/B pins are non-interactive,
    hardcoded-position decoration (`Choreo Redesign.dc.html:82-84,134-135` —
    plain divs, no pointer handlers at all) — see §4 conflict #2.

19. **Tap-to-open snap-mode menu on the loop handles themselves.**
    `LoopBoundaryHandle.tsx:34-53,99-120` — offers Beat/4-count/8-count
    (`BOUNDARY_SNAP_MODES`, `snap.ts:14`). No design equivalent (follows
    from #7/#18 — the design's pins aren't interactive at all).

20. **Keyboard (Space-bar) tap support.** `tapSession.ts:103-115`. The design
    is a click-only mockup with no keyboard path.

21. **Fixed-zoom "2 eight-counts visible" slice with auto-scroll/auto-center
    following playback.** `Timeline.tsx:587-605` (zoom effect) and
    `Timeline.tsx:208-212` (`autoScroll`/`autoCenter` options). The design's
    waveform is a static, non-scrolling, fixed-64-bar strip — no zoom or
    scroll concept.

22. **Loop validation error message** ("Loop needs two different points — A
    and B snapped to the same spot.", `playback.ts:223`), surfaced in both
    `Hud.tsx:52` and `Timeline.tsx:776`. No design equivalent.

23. **Per-mode waveform re-render, including the derived "instrumental"
    mix.** `stemEngine.ts:46-52` (`MODE_DISPLAY`), `197-216` (`peaksOfMix`),
    consumed at `Timeline.tsx:615-619`. The design's Hear select lists the
    same 5 labels as plain text with no indication the waveform itself
    would re-render per mode.

---

## 3. Behavioral Mismatches

Places where the design's control implies different (usually simpler) state
than the code actually holds.

1. **Onset select (single 4-way enum) vs. two independent booleans.**
   Design: one native `<select>` "Onsets" with mutually exclusive options
   Off/Bass/Drums/Both (`Choreo Redesign.dc.html:141-149`, oddly defaulting
   to `defaultValue="Drums"` rather than "Off"). Code: `bassVisible` and
   `drumsVisible` are two independent `useState<boolean>`
   (`Timeline.tsx:183-184`), exposed as two separate checkbox-style buttons
   inside one "Onset ▾" dropdown (`Timeline.tsx:705-735`). This is the exact
   "single 4-way select standing in for two independent booleans" case: the
   4 combinations are informationally equivalent, but the code's shape lets
   each flag change independently rather than through one selection.

2. **"Hear" control: native `<select>` vs. custom button+dropdown.** Design:
   plain `<select>` with `<option>` children (`Choreo Redesign.dc.html:103-
   107,152-156`). Code: `HearControl` (`controls.tsx:157-198`) is a custom
   button opening a list of `role="menuitemradio"` buttons — same
   single-select semantics and same 5 options, so *not* a state mismatch,
   but a literal `<select>` cannot reproduce the design's own per-option
   color-coding for Onsets (`Choreo Redesign.dc.html:145-146`) as easily as
   the current custom-dropdown approach already does for other controls.

3. **Speed control: boolean toggle (design) vs. continuous stepped value
   (code).** Design's `showSpeed` (`Choreo Redesign.dc.html:199,235-238`) is
   a boolean purely for the button's active/inactive look; per
   `README.md:47,57` it is explicitly a mock that doesn't open a real picker.
   Code's `SpeedController` (`playback.ts:265-297`) holds a real numeric
   `speed` (0.25–2, step 0.25) with a range slider, dial tick labels, and a
   reset button (`controls.tsx:99-143`). This is the "design control has
   fewer options than the implementation supports" case named in the
   prompt: a literal port of "the dial just toggles" would have to also
   reopen/reuse the existing real picker underneath, since the design
   deliberately left that part unspecified rather than depicting fewer
   states than exist.

4. **Tap-counting: one-way boolean (design) vs. a small state machine
   (code).** Design's `tapStarted` (`Choreo Redesign.dc.html:199`) only ever
   transitions `false → true`
   (`toggleTapStarted: () => this.setState({ tapStarted: true })`,
   line 229) — there is no path back to `false` in the prototype at all.
   Code's `tapSession.ts` models `tapping` (open/closed, both directions),
   `taps: number[]`, `fit: TapFit | null`, `error: string | null`, and
   `preview: GridData | null`, with `enter`/`exit`/`record`/`accept` all
   independently callable. The design's model cannot represent exiting
   without accepting, re-tapping an already-counted song, a rejected
   (too-uneven) tap run, or the live count-in-phrase readout — all of which
   the real feature does.

5. **Screen-tab visibility: unconditional (design) vs. gated on
   `media_kind` (code).** Design always renders the Timeline/Video
   segmented control, driven purely by `songScreen`
   (`Choreo Redesign.dc.html:71-74,76,112`). Code only renders the tab
   control when `mediaKind === 'video'` (`Song.tsx:260,265`) — an
   audio-only song has just one screen and no tab strip at all.

6. **Library "note": nullable string (design) vs. 3-way enum with several
   distinct messages per state (code).** Design's `song.note` is
   `null | string`, one short label 1:1 with the yellow dot
   (`Choreo Redesign.dc.html:43-45,202-204`). Code's `LibraryState`
   (`types.ts:30`) is a 3-way union, and `stems_evicted` alone produces two
   different message bodies depending on `grid_present`
   (`Library.tsx:28-38`) plus the separate `stems_present` sub-message
   (`Library.tsx:45-52`). The design's single-nullable-string model cannot
   carry this branching.

7. **Loop A/B pins: static decoration (design) vs. live, draggable, always-
   on region (code).** Design's pins are hardcoded at fixed percentages
   with no event handlers at all (`Choreo Redesign.dc.html:82-84,134-135`).
   Code's loop (`playback.ts:128-142,177-263`) is always active, spans the
   whole track by default, and is directly draggable via
   `LoopBoundaryHandle.tsx` with directional (floor/ceil) snapping. See §4
   conflict #2 for what a literal port would delete.

8. **Snap mode has no home in the design's control surface at all.** Since
   the design's loop pins aren't interactive (#7), there is nowhere in the
   design to even discover, let alone change, `snapMode` — a state that in
   code is threaded through `Song.tsx`, `snap.ts`, and both
   `LoopBoundaryHandle` instances (video scrubber + overall timeline) so
   that a change from either surface is instantly reflected in the other.

9. **`darkMode`/`accent`: DC editor props (design) vs. no theming system
   (code).** The design's `darkMode`/`accent` are `.dc.html` **component
   props** (`Choreo Redesign.dc.html:194-197`), editable only from the DC
   prop panel — the rendered prototype itself has no toggle button that
   flips `darkMode` (despite `README.md:50` calling it a "global light/dark
   theme toggle"). Code's `index.css` is a single flat light theme with no
   dark-mode media query, no CSS custom properties, and no accent-color
   token anywhere. A literal implementation has no existing UI control in
   the design to wire a real toggle onto — one would have to be invented.

---

## 4. Conflicts

Places where implementing the design literally would delete something from
§2 or collapse state from §3.

1. **Minimap-as-waveform-strip vs. the documented decision to scrap it.**
   Design's Timeline card holds waveform + a 1px divider + a
   minimap/scrub-strip, all inside one bordered card
   (`Choreo Redesign.dc.html:113-168`). `TimelineOverview.tsx:6-13`'s own
   comment states a *former* waveform-based minimap (with its own peaks,
   its own 8-count shading, and a read-only viewport box) was **deliberately
   scrapped** specifically to avoid a second copy of grid-rendering logic,
   replaced by the current plain playhead+A/B bar. Implementing the
   design's minimap literally — especially "inside the same card, separated
   only by a divider" — risks recreating exactly the duplicated-rendering
   pattern the codebase already removed on purpose.

2. **Static A/B pins vs. draggable loop + snap-menu.** `README.md:10` calls
   for pixel-accurate, literal implementation. Read literally, the design's
   A/B circles are inert absolutely-positioned dots with no pointer
   handlers. Rebuilding them as such would delete drag-to-set-loop and
   tap-to-open-snap-menu entirely (`LoopBoundaryHandle.tsx`) — a functional
   regression, not merely a visual one, and it is not something the design
   copy calls out (only close reading of the prototype's actual DOM/JS
   reveals the pins are non-interactive).

3. **Fixed 64-bar waveform mock vs. continuous, time-indexed rendering.**
   Design's waveform is exactly 64 discrete bars with per-bar boolean
   bass/drum flags from `i % 9`/`i % 5`
   (`Choreo Redesign.dc.html:114-125,248-256`) — a bar count independent of
   song length. The real `Timeline.tsx:195-212` renders continuous
   wavesurfer.js peaks at ~200/sec (`stemEngine.ts:92`,
   `PEAKS_PER_SECOND`) with onsets positioned by real timestamp, and a
   fixed-eight-count zoom/scroll built on wavesurfer's own pixel math
   (`Timeline.tsx:587-605`). A literal 64-bar rebuild would either
   re-quantize continuous data down to 64 buckets — discarding the
   zoom/scroll behavior and onset time-precision — or have no natural
   mapping onto `GridData`/`OnsetData` at all, conflicting with
   `README.md:7`'s instruction to reuse existing state and data.

4. **Six layered overlays (code) vs. one overlay kind (design), same visual
   region.** Design shows exactly one overlay on the waveform: per-bar
   blue/red onset ticks, centered on each bar. Code layers six kinds of
   marks in that same space — 8-count shading, subdivision hairlines,
   beat/downbeat pulse ticks, onset markers, the active-block highlight +
   1..8/"&" labels, and the loop band + edges (`Timeline.tsx`'s own
   comments label these "Layer 2" through "Layer 6"+, e.g. lines 130-159,
   251-265, 378-403, 475-577, 621-630). A literal rebuild of the design's
   simpler waveform card must either omit five of these six §2 features, or
   graft them onto a fixed-64-bar layout that was never designed to host
   them — the design's bars are vertically centered
   (`Choreo Redesign.dc.html:117`), while the code's count labels and
   subdivision ticks are deliberately top/bottom-anchored against a real
   96px-tall wavesurfer container (`Timeline.tsx:84-93,104-106`) to avoid
   colliding with onset markers — an anchoring scheme with no equivalent
   coordinate system in the design's percentage-height bars.

5. **Collapsing the Onset booleans into one select changes suppression
   logic's inputs.** §3 item 1 already flags the shape mismatch; the
   conflict is that `Timeline.tsx:357-359` (`suppressed`) reads
   `bassVisible` and `drumsVisible` **independently** — either one alone can
   suppress a pulse tick near its own onset type. Forcing a single
   Off/Bass/Drums/Both enum would still need to answer "is bass on" and "is
   drums on" as two separate lookups internally, so the enum can only be a
   thin wrapper over the same two booleans — replacing the booleans outright
   (rather than wrapping them) would break tick suppression for whichever
   combination the new enum's cases don't preserve.

6. **Tap panel always visible on the Timeline tab (design) vs. conditional
   on `tap.tapping` (code), with a separate permanent "Re-tap" entry
   point.** Design: `showTapPanel: this.state.songScreen === 'timeline'`
   (`Choreo Redesign.dc.html:241`) — the panel is docked under the card any
   time the Timeline tab is open, tapping active or not. Code:
   `{tap.tapping && <TapOverlay ... />}` (`Timeline.tsx:755`), with a
   separate "Re-tap the count" button shown only when `hasSavedGrid`
   (`Timeline.tsx:744-753`), whose own comment states re-tap must be
   "always here... never locked away" but explicitly **hidden during first
   run** (`Timeline.tsx:736-743`). A literal port that always shows the tap
   panel would surface tap UI even for fully-counted songs that aren't
   being re-tapped, directly conflicting with that comment's intent about
   when tap UI should appear.

7. **Bottom floating Library/[song] tab bar (design) vs. no "last song"
   memory (code).** Design's bottom pill nav lets a user jump from Library
   directly back into whichever song was last open
   (`Choreo Redesign.dc.html:187-190`, `currentSongName`). `App.tsx:8-13`'s
   own comment states the router model is deliberately mount/unmount-only:
   "closed by UNMOUNTING... anything that kept a Song mounted 'in the
   background' would keep an AudioContext alive with it." `App.tsx:26`'s
   `onExit: () => setView({ kind: 'library' })` discards the song entirely
   on exit — there is no "last song" state to point a bottom tab at.
   Implementing the design's bottom bar literally requires either (a)
   adding new "last song" state to `App.tsx` that the router was
   specifically designed not to need, or (b) having the bottom tab just
   remount the song fresh, which is safe for the AudioContext but silently
   loses any open tap session or scroll position the design's copy implies
   would persist ("[song name]" reads as a live tab back into an
   *unchanged* session, not a fresh one).

8. **Segmented control unconditional (design) vs. `media_kind`-gated
   (code).** Already flagged in §3 item 5; the conflict is that the design
   has no concept of `media_kind` to gate on at all, so a literal port must
   either invent gating logic that doesn't exist in the design, or always
   show a Video tab for audio-only songs — which `VideoScreen.tsx:63-66`
   already handles gracefully (`.video-pane-empty`) but which contradicts
   the design's implicit assumption (video preview always has real
   content) and duplicates a decision (`Song.tsx:260`) the code already
   made deliberately in the opposite direction.
