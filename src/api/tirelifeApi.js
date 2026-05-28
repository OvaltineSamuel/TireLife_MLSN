function apiUrl(path) {
  const base = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')
  return base ? `${base}${path}` : path
}

async function postJson(path, payload) {
  const response = await fetch(apiUrl(path), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })

  const data = await response.json().catch(() => ({}))
  if (!response.ok) {
    throw new Error(data?.detail || `Request failed with status ${response.status}`)
  }

  return data
}

async function getJson(path) {
  const response = await fetch(apiUrl(path))
  const data = await response.json().catch(() => ({}))
  if (!response.ok) {
    throw new Error(data?.detail || `Request failed with status ${response.status}`)
  }
  return data
}

export function getHealth() {
  return getJson('/api/health')
}

export function predictTireLife({ features, model = 'lightgbm' }) {
  return postJson('/api/predict', { features, model })
}

export function sendChatMessage({ sessionId, message, model, forcePredict = false }) {
  return postJson('/api/chat', {
    session_id: sessionId,
    message,
    model,
    force_predict: forcePredict,
  })
}
