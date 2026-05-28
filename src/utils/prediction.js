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

const KM_TO_MILES = 0.621371
const MILES_TO_KM = 1.60934
const MM_PER_INCH = 25.4
const THIRTY_SECONDS_PER_INCH = 32
const DEFAULT_LEGAL_MINIMUM_TREAD_DEPTH_MM = 1.6
const DEFAULT_RECOMMENDED_REPLACEMENT_TREAD_DEPTH_MM = 3.0

export function distanceUnitLabel(distanceUnit) {
  return distanceUnit === 'miles' ? 'mi' : 'km'
}

export function treadUnitLabel(treadUnit) {
  return treadUnit === 'inch32' ? '/32 in' : 'mm'
}

export function displayDistanceToKm(value, distanceUnit) {
  const numeric = Number(value || 0)
  return distanceUnit === 'miles' ? numeric * MILES_TO_KM : numeric
}

export function kmToDisplayDistance(value, distanceUnit) {
  const numeric = Number(value || 0)
  return distanceUnit === 'miles' ? numeric * KM_TO_MILES : numeric
}

export function displayTreadToMm(value, treadUnit) {
  const numeric = Number(value || 0)
  return treadUnit === 'inch32'
    ? (numeric / THIRTY_SECONDS_PER_INCH) * MM_PER_INCH
    : numeric
}

export function mmToDisplayTread(value, treadUnit) {
  const numeric = Number(value || 0)
  return treadUnit === 'inch32'
    ? (numeric / MM_PER_INCH) * THIRTY_SECONDS_PER_INCH
    : numeric
}

export function formatTreadWithUnit(valueMm, treadUnit) {
  const displayValue = mmToDisplayTread(valueMm, treadUnit)
  return treadUnit === 'inch32'
    ? `${formatOneDecimal(displayValue)}${treadUnitLabel(treadUnit)}`
    : `${formatOneDecimal(displayValue)} ${treadUnitLabel(treadUnit)}`
}

export function derivePredictionSnapshot(prediction) {
  if (!prediction) return null
  const normalized = prediction.normalized_features || {}

  const predictedKm = Number(prediction.predicted_rul_km || 0)
  const predictedMiles = Number(prediction.predicted_rul_miles || predictedKm * KM_TO_MILES)
  const rawModelKm = Number(prediction.raw_model_rul_to_zero_km || predictedKm)
  const rawModelMiles = Number(prediction.raw_model_rul_to_zero_miles || rawModelKm * KM_TO_MILES)
  const legalMinimumRulKm = Number(prediction.legal_minimum_rul_km || predictedKm)
  const legalMinimumRulMiles = Number(prediction.legal_minimum_rul_miles || legalMinimumRulKm * KM_TO_MILES)
  const legalMinimumTreadDepthMm = Number(
    prediction.legal_minimum_tread_depth_mm || DEFAULT_LEGAL_MINIMUM_TREAD_DEPTH_MM
  )
  const recommendedReplacementTreadDepthMm = Number(
    prediction.recommended_replacement_tread_depth_mm || DEFAULT_RECOMMENDED_REPLACEMENT_TREAD_DEPTH_MM
  )
  const replacementThresholdMm = Number(
    prediction.replacement_threshold_mm || recommendedReplacementTreadDepthMm
  )
  const expectedLifeKm = Number(normalized['expected_tyre_life(km)'] || 0)
  const treadUsedPct = Number(normalized.tread_depth_used_pct)
  const standardDepthMm = Number(normalized['Standard_tread_depth(mm)'] || 0)
  const currentDepthMm = Number(normalized['current_tread_depth(mm)'] || 0)
  const tireAgeYears = Number(normalized['tyre_age(years)'] || 0)
  const kmDriven = Number(normalized['kilometers_driven(km)'] || 0)
  const usableTread = Math.max(0, currentDepthMm - replacementThresholdMm)
  const totalUsableTread = Math.max(0, standardDepthMm - replacementThresholdMm)

  let pctLeft = 50
  if (totalUsableTread > 0 && currentDepthMm >= 0) {
    pctLeft = clamp((usableTread / totalUsableTread) * 100, 0, 100)
  } else if (rawModelKm > 0) {
    pctLeft = clamp((predictedKm / rawModelKm) * 100, 0, 100)
  } else if (Number.isFinite(treadUsedPct) && treadUsedPct >= 0) {
    pctLeft = clamp((1 - treadUsedPct) * 100, 0, 100)
  } else if (expectedLifeKm > 0) {
    pctLeft = clamp((predictedKm / expectedLifeKm) * 100, 0, 100)
  }

  let currentTreadDisplay = 0
  if (currentDepthMm > 0) {
    currentTreadDisplay = currentDepthMm
  } else if (standardDepthMm > 0) {
    currentTreadDisplay = replacementThresholdMm + (pctLeft / 100) * totalUsableTread
  }

  const annualKmFromUsage = tireAgeYears > 0 && kmDriven > 0 ? kmDriven / tireAgeYears : 0
  const annualKm = annualKmFromUsage > 300 ? annualKmFromUsage : 12000
  const yearsLeft = annualKm > 0 ? predictedKm / annualKm : 0

  let status = 'Low'
  let color = '#ef4444'
  if (currentDepthMm > 0 && currentDepthMm <= legalMinimumTreadDepthMm) {
    status = 'Below Legal'
    color = '#dc2626'
  } else if (currentDepthMm > 0 && currentDepthMm <= replacementThresholdMm) {
    status = 'Replace Now'
    color = '#ef4444'
  } else if (pctLeft >= 70) {
    status = 'Good'
    color = '#22c55e'
  } else if (pctLeft >= 40) {
    status = 'Fair'
    color = '#f59e0b'
  } else if (pctLeft >= 20) {
    status = 'Low'
    color = '#f97316'
  }

  return {
    predictedKm,
    predictedMiles,
    rawModelKm,
    rawModelMiles,
    legalMinimumRulKm,
    legalMinimumRulMiles,
    defaultsUsed: Number(prediction.defaults_used_count || 0),
    model: prediction.model,
    pctLeft,
    status,
    color,
    yearsLeft,
    currentTread: currentTreadDisplay,
    usableTread,
    replacementThresholdMm,
    legalMinimumTreadDepthMm,
    recommendedReplacementTreadDepthMm,
  }
}
