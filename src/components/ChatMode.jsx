import { useEffect, useRef, useState } from 'react'
import { sendChatMessage } from '../api/tirelifeApi.js'
import ResultCard from './ResultCard.jsx'

const starterMessage =
  "Describe your tyre condition in plain English. Include tread depth, distance driven, age, and pressure if you know them."

const MODEL_LABELS = {
  lightgbm: 'LightGBM',
  deeplearning_test: 'Deep Learning Test',
}

const FRIENDLY_FIELD_LABELS = {
  'current_tread_depth(mm)': 'tread depth',
  'kilometers_driven(km)': 'distance driven',
  'average_inflation_pressure(psi)': 'tire pressure',
  'tyre_age(years)': 'tire age',
  vehicle_model: 'vehicle model',
  tyre_brand: 'tire brand',
  number_of_punctures: 'puncture count',
  road_condition: 'road condition',
  weather_condition: 'weather condition',
  'recommended_inflation_pressure(psi)': 'recommended pressure',
  'Standard_tread_depth(mm)': 'new tire tread depth',
}

function friendlyFieldName(field) {
  return FRIENDLY_FIELD_LABELS[field] || field
}

function renderInline(text, keyPrefix = 'inline') {
  const chunks = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g)
  return chunks.map((chunk, i) => {
    const key = `${keyPrefix}-${i}`
    if (!chunk) return null
    if (chunk.startsWith('**') && chunk.endsWith('**')) {
      return <strong key={key}>{chunk.slice(2, -2)}</strong>
    }
    if (chunk.startsWith('`') && chunk.endsWith('`')) {
      return (
        <code key={key} style={{
          background: '#eef2ff',
          border: '1px solid #c7d2fe',
          borderRadius: 6,
          padding: '1px 5px',
          fontSize: '0.92em',
          color: '#1e1b4b',
        }}>
          {chunk.slice(1, -1)}
        </code>
      )
    }
    return <span key={key}>{chunk}</span>
  })
}

function AssistantText({ text }) {
  const blocks = text.split(/\n{2,}/).map(part => part.trim()).filter(Boolean)

  return (
    <div style={{ display: 'grid', gap: 10 }}>
      {blocks.map((block, idx) => {
        const lines = block.split('\n').map(l => l.trim()).filter(Boolean)
        const bulletLines = lines.filter(l => l.startsWith('- '))

        if (bulletLines.length === lines.length && bulletLines.length > 0) {
          return (
            <ul key={`block-${idx}`} style={{ margin: 0, paddingLeft: 18, display: 'grid', gap: 6 }}>
              {bulletLines.map((line, bulletIdx) => (
                <li key={`bullet-${idx}-${bulletIdx}`} style={{ margin: 0 }}>
                  {renderInline(line.slice(2), `bullet-${idx}-${bulletIdx}`)}
                </li>
              ))}
            </ul>
          )
        }

        return (
          <p key={`block-${idx}`} style={{ margin: 0, lineHeight: 1.65 }}>
            {renderInline(block, `block-${idx}`)}
          </p>
        )
      })}
    </div>
  )
}

export default function ChatMode({
  availableModels = ['lightgbm'],
  distanceUnit = 'km',
  treadUnit = 'mm',
}) {
  const [model, setModel] = useState('lightgbm')
  const [sessionId, setSessionId] = useState(null)
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [messages, setMessages] = useState([
    { role: 'assistant', text: starterMessage, prediction: null },
  ])
  const [missingFields, setMissingFields] = useState([])
  const [suggestAutofill, setSuggestAutofill] = useState(false)
  const inputRef = useRef(null)

  useEffect(() => {
    const el = inputRef.current
    if (!el) return

    el.style.height = '0px'
    const nextHeight = Math.min(160, Math.max(88, el.scrollHeight))
    el.style.height = `${nextHeight}px`
  }, [input])

  const placeholder = treadUnit === 'inch32'
    ? `Example: tread is 6/32, driven 28000 ${distanceUnit === 'miles' ? 'miles' : 'km'}, pressure 32 psi, tyre age 3 years`
    : `Example: tread is 4.5 mm, driven 28000 ${distanceUnit === 'miles' ? 'miles' : 'km'}, pressure 32 psi, tyre age 3 years`

  async function sendMessage(rawMessage, forcePredict = false) {
    const message = rawMessage.trim()
    if (!message || loading) return

    setMessages(m => [...m, { role: 'user', text: message, prediction: null }])
    setInput('')
    setLoading(true)

    try {
      const data = await sendChatMessage({ sessionId, message, model, forcePredict })
      setSessionId(data.session_id)
      setMissingFields(data.missing_fields || [])
      setSuggestAutofill(Boolean(data.suggest_autofill))
      setMessages(m => [
        ...m,
        {
          role: 'assistant',
          text: data.assistant_message,
          prediction: data.prediction || null,
        },
      ])
    } catch (err) {
      setSuggestAutofill(false)
      setMessages(m => [
        ...m,
        {
          role: 'assistant',
          text: `Backend error: ${err.message}`,
          prediction: null,
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  function onSubmit(e) {
    e.preventDefault()
    sendMessage(input)
  }

  function onInputKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage(input)
    }
  }

  function resetChat() {
    setSessionId(null)
    setMissingFields([])
    setSuggestAutofill(false)
    setMessages([{ role: 'assistant', text: starterMessage, prediction: null }])
    setInput('')
  }

  return (
    <div>
      <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
        <select
          value={model}
          onChange={e => setModel(e.target.value)}
          style={{
            flex: 1,
            border: '1px solid #d1d5db',
            borderRadius: 8,
            padding: '8px 10px',
            fontSize: 13,
            background: '#fff',
          }}
        >
          {availableModels.map(modelName => (
            <option key={modelName} value={modelName}>
              {MODEL_LABELS[modelName] || modelName}
            </option>
          ))}
        </select>
        <button
          type="button"
          onClick={resetChat}
          style={{
            border: '1px solid #d1d5db',
            borderRadius: 8,
            background: '#fff',
            padding: '8px 12px',
            fontSize: 13,
            cursor: 'pointer',
          }}
        >
          New Case
        </button>
      </div>

      <div style={{
        border: '1px solid #e5e7eb',
        background: '#fff',
        borderRadius: 12,
        padding: 14,
        maxHeight: 430,
        overflowY: 'auto',
      }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {messages.map((m, idx) => (
            <div
              key={idx}
              style={{
                alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start',
                maxWidth: '92%',
              }}
            >
              <div style={{
                fontSize: 11,
                color: '#94a3b8',
                marginBottom: 4,
                textAlign: m.role === 'user' ? 'right' : 'left',
                fontWeight: 600,
                letterSpacing: '0.04em',
                textTransform: 'uppercase',
              }}>
                {m.role === 'user' ? 'You' : 'TireLife AI'}
              </div>
              <div
                style={{
                  borderRadius: 12,
                  padding: '11px 13px',
                  fontSize: 14,
                  lineHeight: 1.65,
                  whiteSpace: 'pre-wrap',
                  overflowWrap: 'anywhere',
                  border: m.role === 'user' ? '1px solid #0f172a' : '1px solid #dbe2ea',
                  background: m.role === 'user' ? '#0f172a' : '#f8fafc',
                  color: m.role === 'user' ? '#fff' : '#1f2937',
                }}
              >
                {m.role === 'assistant' ? <AssistantText text={m.text} /> : m.text}
              </div>
              <ResultCard prediction={m.prediction} distanceUnit={distanceUnit} treadUnit={treadUnit} />
            </div>
          ))}
          {loading && (
            <div style={{ fontSize: 12, color: '#6b7280', animation: 'pulse 1.2s ease-in-out infinite' }}>
              Thinking...
            </div>
          )}
        </div>
      </div>

      {missingFields.length > 0 && (
        <div style={{ marginTop: 10 }}>
          <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 6 }}>
            Requested fields: {missingFields.map(friendlyFieldName).join(', ')}
          </div>
          {suggestAutofill ? (
            <div style={{ display: 'flex', gap: 8 }}>
              <button
                type="button"
                onClick={() => sendMessage('use defaults', true)}
                disabled={loading}
                style={{
                  flex: 1,
                  border: '1px solid #d1d5db',
                  borderRadius: 8,
                  background: '#f8fafc',
                  padding: '9px 10px',
                  fontSize: 13,
                  cursor: loading ? 'not-allowed' : 'pointer',
                }}
              >
                Predict Now With Defaults
              </button>
            </div>
          ) : (
            <div style={{ display: 'grid', gap: 6 }}>
              <div style={{ fontSize: 12, color: '#9ca3af' }}>
                Share one more detail for a better estimate.
              </div>
              <button
                type="button"
                onClick={() => sendMessage("that's all I know")}
                disabled={loading}
                style={{
                  width: '100%',
                  border: '1px solid #d1d5db',
                  borderRadius: 8,
                  background: '#fff',
                  padding: '9px 10px',
                  fontSize: 13,
                  cursor: loading ? 'not-allowed' : 'pointer',
                }}
              >
                I Don't Know More
              </button>
            </div>
          )}
        </div>
      )}

      <form onSubmit={onSubmit} style={{ marginTop: 10, display: 'flex', gap: 8, alignItems: 'stretch' }}>
        <textarea
          ref={inputRef}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={onInputKeyDown}
          rows={1}
          placeholder={placeholder}
          style={{
            flex: 1,
            minHeight: 88,
            maxHeight: 160,
            border: '1px solid #d1d5db',
            borderRadius: 8,
            padding: '10px 12px',
            fontSize: 14,
            lineHeight: 1.45,
            resize: 'none',
            overflowY: 'auto',
            boxSizing: 'border-box',
            fontFamily: 'inherit',
          }}
        />
        <div style={{ width: 72, display: 'grid', gridTemplateRows: '1fr 1fr', gap: 6 }}>
          <button
            type="submit"
            disabled={loading}
            style={{
              border: 'none',
              borderRadius: 8,
              padding: '0 10px',
              fontSize: 14,
              fontWeight: 600,
              background: '#111',
              color: '#fff',
              cursor: loading ? 'not-allowed' : 'pointer',
            }}
          >
            Send
          </button>
          <button
            type="button"
            onClick={resetChat}
            disabled={loading}
            style={{
              border: '1px solid #fecaca',
              borderRadius: 8,
              padding: '0 10px',
              fontSize: 13,
              fontWeight: 700,
              background: '#fef2f2',
              color: '#b91c1c',
              cursor: loading ? 'not-allowed' : 'pointer',
            }}
          >
            Reset
          </button>
        </div>
      </form>
    </div>
  )
}
