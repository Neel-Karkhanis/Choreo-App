// The "C" of Choreo: two concentric octagon outlines left open on their right
// edge (hence "horeo" carries the wordmark — the octagons' gap reads as the
// "C"), plus 6 short seam lines connecting the two rings at each vertex pair,
// plus 2 short caps that close the open ends into rounded tips. Path `d` and
// `dividerPairs` copied verbatim from design/handoff/Choreo Redesign.dc.html
// (search `dividerPairs`) — do not hand-tune these coordinates.
const OUTER_RING = 'M92.5,67.6 L67.6,92.5 L32.4,92.5 L7.5,67.6 L7.5,32.4 L32.4,7.5 L67.6,7.5 L92.5,32.4'
const INNER_RING = 'M77.7,61.5 L61.5,77.7 L38.5,77.7 L22.3,61.5 L22.3,38.5 L38.5,22.3 L61.5,22.3 L77.7,38.5'
const CAP_TOP = 'M77.7,38.5 L92.5,32.4'
const CAP_BOTTOM = 'M77.7,61.5 L92.5,67.6'

const DIVIDER_PAIRS: Array<[number, number, number, number]> = [
  [67.6, 92.5, 61.5, 77.7],
  [32.4, 92.5, 38.5, 77.7],
  [7.5, 67.6, 22.3, 61.5],
  [7.5, 32.4, 22.3, 38.5],
  [32.4, 7.5, 38.5, 22.3],
  [67.6, 7.5, 61.5, 22.3],
]

export default function Logo({ size = 30 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
      <path d={OUTER_RING} fill="none" stroke="var(--color-accent)" strokeWidth={8} />
      <path d={INNER_RING} fill="none" stroke="var(--color-accent)" strokeWidth={8} />
      {DIVIDER_PAIRS.map(([x1, y1, x2, y2], i) => (
        <line
          key={i}
          x1={x1}
          y1={y1}
          x2={x2}
          y2={y2}
          stroke="var(--color-accent)"
          strokeWidth={4}
          strokeLinecap="round"
        />
      ))}
      <path d={CAP_BOTTOM} fill="none" stroke="var(--color-accent)" strokeWidth={8} />
      <path d={CAP_TOP} fill="none" stroke="var(--color-accent)" strokeWidth={8} />
    </svg>
  )
}
