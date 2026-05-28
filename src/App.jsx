import { useEffect, useState } from 'react'
import StructuredForm    from './components/StructuredForm.jsx'
import ChatMode          from './components/ChatMode.jsx'
import { getHealth } from './api/tirelifeApi.js'

const TABS = [
  { id: 'form', label: 'Input Form' },
  { id: 'chat', label: 'Chat' },
]

function UnitToggle({ label, value, options, onChange }) {
  return (
    <div>
      <div style={{ fontSize: 10, color: '#94a3b8', fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 5 }}>
        {label}
      </div>
      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        background: '#f3f4f6',
        borderRadius: 8,
        padding: 3,
        border: '1px solid #e5e7eb',
      }}>
        {options.map(option => (
          <button
            key={option.value}
            type="button"
            onClick={() => onChange(option.value)}
            style={{
              border: 'none',
              borderRadius: 6,
              padding: '7px 8px',
              background: value === option.value ? '#111827' : 'transparent',
              color: value === option.value ? '#fff' : '#64748b',
              fontSize: 12,
              fontWeight: 700,
              cursor: 'pointer',
            }}
          >
            {option.label}
          </button>
        ))}
      </div>
    </div>
  )
}

export default function App() {
  const [tab, setTab] = useState('form')
  const [availableModels, setAvailableModels] = useState(['lightgbm'])
  const [distanceUnit, setDistanceUnit] = useState('km')
  const [treadUnit, setTreadUnit] = useState('mm')

  useEffect(() => {
    let ignore = false

    getHealth()
      .then(data => {
        const models = Array.isArray(data.models_available) ? data.models_available : []
        if (!ignore && models.length) {
          setAvailableModels(models)
        }
      })
      .catch(() => {
        if (!ignore) {
          setAvailableModels(['lightgbm'])
        }
      })

    return () => {
      ignore = true
    }
  }, [])

  return (
    <div style={{
      minHeight: '100vh',
      background: '#fafafa',
      display: 'flex',
      alignItems: 'flex-start',
      justifyContent: 'center',
      padding: '2.5rem 1rem',
    }}>
      <div style={{ width: '100%', maxWidth: 520 }}>

        <header style={{ marginBottom: '2rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div style={{
              width: 36, height: 36, borderRadius: '50%',
              border: '3px solid #111',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: 16,
            }}>
              TL
            </div>
            <div>
              <div style={{ fontSize: 22, fontWeight: 800, letterSpacing: 0, color: '#111', lineHeight: 1 }}>
                TireLife
              </div>
              <div style={{ fontSize: 12, color: '#9ca3af', letterSpacing: '0.1em', textTransform: 'uppercase' }}>
                Remaining Useful Life Estimator
              </div>
            </div>
          </div>
        </header>

        <main style={{
          background: '#fff',
          borderRadius: 8,
          border: '1px solid #e5e7eb',
          padding: '1.5rem',
          boxShadow: '0 2px 12px rgba(0,0,0,0.06)',
        }}>

          <div style={{
            display: 'flex',
            background: '#f3f4f6',
            borderRadius: 8,
            padding: 4,
            marginBottom: 12,
          }}>
            {TABS.map(t => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                style={{
                  flex: 1,
                  padding: '8px 0',
                  borderRadius: 7,
                  border: 'none',
                  background: tab === t.id ? '#fff' : 'transparent',
                  color: tab === t.id ? '#111' : '#9ca3af',
                  fontWeight: tab === t.id ? 600 : 400,
                  fontSize: 13,
                  cursor: 'pointer',
                  boxShadow: tab === t.id ? '0 1px 4px rgba(0,0,0,0.10)' : 'none',
                  transition: 'all 0.15s',
                }}
              >
                {t.label}
              </button>
            ))}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginBottom: '1.5rem' }}>
            <UnitToggle
              label="Distance Unit"
              value={distanceUnit}
              options={[
                { value: 'km', label: 'km' },
                { value: 'miles', label: 'miles' },
              ]}
              onChange={setDistanceUnit}
            />
            <UnitToggle
              label="Tread Unit"
              value={treadUnit}
              options={[
                { value: 'mm', label: 'mm' },
                { value: 'inch32', label: 'inch' },
              ]}
              onChange={setTreadUnit}
            />
          </div>

          <div style={{ display: tab === 'form' ? 'block' : 'none' }}>
            <StructuredForm
              availableModels={availableModels}
              distanceUnit={distanceUnit}
              treadUnit={treadUnit}
            />
          </div>
          <div style={{ display: tab === 'chat' ? 'block' : 'none' }}>
            <ChatMode
              availableModels={availableModels}
              distanceUnit={distanceUnit}
              treadUnit={treadUnit}
            />
          </div>
        </main>

        {/* ── Footer ── */}
        <footer style={{ marginTop: '1.5rem', textAlign: 'center', fontSize: 11, color: '#c4c4c4', lineHeight: 1.7 }}>
          Estimates are produced by the configured backend model.<br />
          Always consult a professional for safety-critical decisions.
        </footer>

      </div>
    </div>
  )
}
