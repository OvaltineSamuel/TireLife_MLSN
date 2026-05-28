import TreadGauge from './TreadGauge.jsx'
import {
  derivePredictionSnapshot,
  formatInt,
  formatOneDecimal,
} from '../utils/prediction.js'

const metricCard = {
  background: '#f8fafc',
  border: '1px solid #e5e7eb',
  borderRadius: 8,
  padding: '10px 8px',
  textAlign: 'center',
}

export default function ResultCard({ prediction }) {
  const snapshot = derivePredictionSnapshot(prediction)
  if (!snapshot) return null

  const metrics = [
    { label: 'Miles Left', value: formatInt(snapshot.predictedMiles), unit: 'mi' },
    { label: 'Years Left', value: formatOneDecimal(snapshot.yearsLeft), unit: 'yrs' },
    { label: 'Tread Depth', value: formatOneDecimal(snapshot.remainingTread), unit: 'mm' },
  ]

  return (
    <div style={{
      marginTop: 10,
      border: '1px solid #d7dde5',
      background: 'linear-gradient(180deg, #ffffff 0%, #f8fafc 100%)',
      borderRadius: 8,
      padding: 14,
      boxShadow: '0 2px 10px rgba(15, 23, 42, 0.07)',
    }}>
      <div style={{ fontSize: 11, color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.09em' }}>
        Final Tire Life Estimate
      </div>
      <div style={{ marginTop: 3, fontSize: 34, fontWeight: 800, color: '#0f172a', lineHeight: 1.05 }}>
        {formatInt(snapshot.predictedKm)} km
      </div>
      <div style={{ marginTop: 5, fontSize: 14, color: '#475569' }}>
        {formatInt(snapshot.predictedMiles)} miles | model: {snapshot.model}
      </div>

      <div style={{ marginTop: 14, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <TreadGauge pct={snapshot.pctLeft} color={snapshot.color} />
        <div style={{
          marginTop: 6,
          padding: '4px 16px',
          borderRadius: 999,
          border: `1.5px solid ${snapshot.color}`,
          background: `${snapshot.color}1F`,
          color: snapshot.color,
          fontWeight: 700,
          fontSize: 12,
          letterSpacing: '0.08em',
          textTransform: 'uppercase',
        }}>
          {snapshot.status}
        </div>
      </div>

      <div style={{ marginTop: 14, display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(108px, 1fr))', gap: 8 }}>
        {metrics.map(metric => (
          <div key={metric.label} style={metricCard}>
            <div style={{ fontSize: 10, color: '#94a3b8', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
              {metric.label}
            </div>
            <div style={{ marginTop: 3, fontSize: 21, fontWeight: 800, color: '#111827', fontFamily: "'JetBrains Mono', monospace" }}>
              {metric.value}
            </div>
            <div style={{ fontSize: 10, color: '#94a3b8' }}>{metric.unit}</div>
          </div>
        ))}
      </div>

      <div style={{ marginTop: 10, fontSize: 12, color: '#64748b', fontWeight: 600 }}>
        Defaults used: {snapshot.defaultsUsed}
      </div>
    </div>
  )
}
