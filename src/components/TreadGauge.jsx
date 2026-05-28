/**
 * Semi-circular gauge showing tread percentage remaining.
 */
export default function TreadGauge({ pct, color }) {
  const r = 52
  const circumference = Math.PI * r

  const offset = circumference - (pct / 100) * circumference

  return (
    <svg
      width="128"
      height="80"
      viewBox="0 0 128 88"
      style={{ display: 'block', margin: '0 auto' }}
      aria-label={`${Math.round(pct)}% tread remaining`}
      role="img"
    >
      {/* Background arc */}
      <path
        d={`M 12 64 A ${r} ${r} 0 0 1 116 64`}
        fill="none"
        stroke="#e5e7eb"
        strokeWidth="10"
        strokeLinecap="round"
      />
      {/* Filled arc */}
      <path
        d={`M 12 64 A ${r} ${r} 0 0 1 116 64`}
        fill="none"
        stroke={color}
        strokeWidth="10"
        strokeLinecap="round"
        strokeDasharray={circumference}
        strokeDashoffset={offset}
        style={{ transition: 'stroke-dashoffset 0.8s cubic-bezier(.4,0,.2,1), stroke 0.4s' }}
      />
      <text
        x="64" y="62"
        textAnchor="middle"
        fontSize="20"
        fontWeight="700"
        fill={color}
        fontFamily="'JetBrains Mono', monospace"
      >
        {Math.round(pct)}%
      </text>
      <text
        x="64" y="78"
        textAnchor="middle"
        fontSize="10"
        fill="#9ca3af"
        fontFamily="system-ui"
      >
        tread remaining
      </text>
    </svg>
  )
}
