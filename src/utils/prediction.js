export function formatInt(value) {
  return Number(value || 0).toLocaleString(undefined, { maximumFractionDigits: 0 })
}

export function formatOneDecimal(value) {
  return Number(value || 0).toLocaleString(undefined, {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  })
}

export function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value))
}

export function derivePredictionSnapshot(prediction) {
  if (!prediction) return null
  const normalized = prediction.normalized_features || {}

  const predictedKm = Number(prediction.predicted_rul_km || 0)
  const predictedMiles = Number(prediction.predicted_rul_miles || predictedKm * 0.621371)
  const expectedLifeKm = Number(normalized['expected_tyre_life(km)'] || 0)
  const treadUsedPct = Number(normalized.tread_depth_used_pct)
  const standardDepthMm = Number(normalized['Standard_tread_depth(mm)'] || 0)
  const currentDepthMm = Number(normalized['current_tread_depth(mm)'] || 0)
  const tireAgeYears = Number(normalized['tyre_age(years)'] || 0)
  const kmDriven = Number(normalized['kilometers_driven(km)'] || 0)

  let pctLeft = 50
  if (Number.isFinite(treadUsedPct) && treadUsedPct >= 0) {
    pctLeft = clamp((1 - treadUsedPct) * 100, 0, 100)
  } else if (standardDepthMm > 0 && currentDepthMm >= 0) {
    pctLeft = clamp((currentDepthMm / standardDepthMm) * 100, 0, 100)
  } else if (expectedLifeKm > 0) {
    pctLeft = clamp((predictedKm / expectedLifeKm) * 100, 0, 100)
  }

  let remainingTread = 0
  if (currentDepthMm > 0) {
    remainingTread = currentDepthMm
  } else if (standardDepthMm > 0) {
    remainingTread = (pctLeft / 100) * standardDepthMm
  }

  const annualKmFromUsage = tireAgeYears > 0 && kmDriven > 0 ? kmDriven / tireAgeYears : 0
  const annualKm = annualKmFromUsage > 300 ? annualKmFromUsage : 12000
  const yearsLeft = annualKm > 0 ? predictedKm / annualKm : 0

  let status = 'Low'
  let color = '#ef4444'
  if (pctLeft >= 70) {
    status = 'Good'
    color = '#22c55e'
  } else if (pctLeft >= 40) {
    status = 'Fair'
    color = '#f59e0b'
  }

  return {
    predictedKm,
    predictedMiles,
    defaultsUsed: Number(prediction.defaults_used_count || 0),
    model: prediction.model,
    pctLeft,
    status,
    color,
    yearsLeft,
    remainingTread,
  }
}
