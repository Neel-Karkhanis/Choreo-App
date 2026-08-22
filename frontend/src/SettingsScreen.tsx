import { useAccentColor } from './accentColor'
import { SNAP_MODE_LABELS, SNAP_MODES, useDefaultSnapMode } from './snap'
import { toggleStyle } from './styles'

// LEVEL 1 sibling of Library (App.tsx's View): its own screen, not a
// dropdown — at the user's explicit request, reversed from an earlier
// dropdown-popover version once it grew a second, unrelated option (beat
// snapping joining accent color). Two independent settings read better as
// sections on a screen than stacked in one small menu. No design mock covers
// Settings at all (the redesign doesn't have one yet), so this screen's
// layout is invented, same as the Timeline's own canvas palette elsewhere in
// this app — kept deliberately plain (page head + two labeled sections) to
// match the rest of the app's placeholder-styling-pending-a-design-pass
// look (see e.g. SpeedControl's own comment in controls.tsx).
export default function SettingsScreen({ onExit }: { onExit: () => void }) {
  const { accentId, setAccentId, options } = useAccentColor()
  const { defaultSnapMode, setDefaultSnapMode } = useDefaultSnapMode()

  return (
    <main className="settings">
      <header className="song-head">
        <div className="song-head-left">
          <button className="song-back" onClick={onExit} aria-label="Back to library">
            ←
          </button>
          <h1>Settings</h1>
        </div>
      </header>

      <section className="settings-section">
        <h2 className="settings-section-title">Accent color</h2>
        <p className="settings-section-note">
          Colors the app's buttons and the Timeline's progress fill, active-count
          highlight, and loop band together.
        </p>
        {/* Curated swatches, not a free color picker — see accentColor.ts for
            why: an arbitrary pick could land too close to the onset markers'
            own tuned hues. */}
        <div className="accent-swatch-row">
          {options.map((option) => (
            <button
              key={option.id}
              type="button"
              className="accent-swatch"
              style={{ backgroundColor: option.hex }}
              aria-label={option.label}
              aria-pressed={option.id === accentId}
              onClick={() => setAccentId(option.id)}
            />
          ))}
        </div>
      </section>

      <section className="settings-section">
        <h2 className="settings-section-title">Beat snapping</h2>
        <p className="settings-section-note">Snap mode for looping boundaries.</p>
        <div className="settings-snap-row" role="radiogroup" aria-label="Default beat snapping">
          {SNAP_MODES.map((mode) => (
            <button
              key={mode}
              type="button"
              role="radio"
              aria-checked={defaultSnapMode === mode}
              onClick={() => setDefaultSnapMode(mode)}
              style={toggleStyle(defaultSnapMode === mode, 'var(--color-accent)')}
            >
              {SNAP_MODE_LABELS[mode]}
            </button>
          ))}
        </div>
      </section>
    </main>
  )
}
