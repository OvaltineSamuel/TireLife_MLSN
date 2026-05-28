import { useEffect, useRef, useState } from 'react'
import { TIRE_BRANDS } from '../constants.js'
import { predictTireLife } from '../api/tirelifeApi.js'
import ResultCard from './ResultCard.jsx'
import {
  displayDistanceToKm,
  displayTreadToMm,
  kmToDisplayDistance,
  mmToDisplayTread,
} from '../utils/prediction.js'

const DEFAULT_FORM = {
  model: 'lightgbm',
  vehicleModel: '',
  brand: 'Michelin Pilot Sport 4S',
  yearsOwned: '',
  milesPerDay: '',
  currentTread: '',
  pressure: '',
  roadCondition: 'smooth',
  weatherCondition: 'dry',
}

const MODEL_LABELS = {
  lightgbm: 'LightGBM',
  deeplearning_test: 'Deep Learning Test',
}

const labelStyle = {
  fontSize: 12,
  fontWeight: 600,
  color: '#6b7280',
  letterSpacing: '0.07em',
  textTransform: 'uppercase',
  marginBottom: 5,
  display: 'block',
}

function inputStyle(hasError) {
  return {
    width: '100%',
    padding: '9px 12px',
    borderRadius: 8,
    border: `1px solid ${hasError ? '#ef4444' : '#d1d5db'}`,
    fontSize: 14,
    background: '#fff',
    boxSizing: 'border-box',
  }
}

function optionalNumber(value) {
  return value === '' ? null : Number(value)
}

function formatEditableNumber(value, fractionDigits = 1) {
  if (value === '') return ''
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return value
  return String(Number(numeric.toFixed(fractionDigits)))
}

function validate(form, distanceUnit, treadUnit) {
  const errors = {}
  const yearsOwned = Number(form.yearsOwned)
  const distancePerDay = Number(form.milesPerDay)
  const currentTread = optionalNumber(form.currentTread)
  const pressure = optionalNumber(form.pressure)
  const maxTread = treadUnit === 'inch32' ? 20 : 15
  const treadLabel = treadUnit === 'inch32' ? '32nds of an inch' : 'mm'
  const distanceLabel = distanceUnit === 'miles' ? 'miles' : 'kilometers'

  if (form.vehicleModel.trim().length > 80) {
    errors.vehicleModel = 'Keep vehicle model under 80 characters'
  }
  if (!form.yearsOwned || Number.isNaN(yearsOwned) || yearsOwned < 0) {
    errors.yearsOwned = 'Enter a valid number of years'
  }
  if (!form.milesPerDay || Number.isNaN(distancePerDay) || distancePerDay <= 0) {
    errors.milesPerDay = `Enter ${distanceLabel} per day greater than 0`
  }
  if (currentTread !== null && (Number.isNaN(currentTread) || currentTread < 0 || currentTread > maxTread)) {
    errors.currentTread = `Enter 0 to ${maxTread} ${treadLabel}, or leave blank`
  }
  if (pressure !== null && (Number.isNaN(pressure) || pressure < 15 || pressure > 60)) {
    errors.pressure = 'Enter 15 to 60 psi, or leave blank'
  }

  return errors
}

function buildFeatures(form, distanceUnit, treadUnit) {
  const yearsOwned = Number(form.yearsOwned)
  const distancePerDay = Number(form.milesPerDay)
  const kilometersDriven = yearsOwned * 365.25 * displayDistanceToKm(distancePerDay, distanceUnit)
  const features = {
    tyre_brand: form.brand,
    road_condition: form.roadCondition,
    weather_condition: form.weatherCondition,
    'tyre_age(years)': yearsOwned,
    'kilometers_driven(km)': kilometersDriven,
  }

  if (form.vehicleModel.trim()) {
    features.vehicle_model = form.vehicleModel.trim()
  }
  if (form.currentTread !== '') {
    features['current_tread_depth(mm)'] = displayTreadToMm(Number(form.currentTread), treadUnit)
  }
  if (form.pressure !== '') {
    features['average_inflation_pressure(psi)'] = Number(form.pressure)
  }

  return features
}

export default function StructuredForm({
  availableModels = ['lightgbm'],
  distanceUnit = 'km',
  treadUnit = 'mm',
}) {
  const [form, setForm] = useState(DEFAULT_FORM)
  const [errors, setErrors] = useState({})
  const [result, setResult] = useState(null)
  const [apiError, setApiError] = useState('')
  const [loading, setLoading] = useState(false)
  const previousDistanceUnit = useRef(distanceUnit)
  const previousTreadUnit = useRef(treadUnit)

  const set = (key, value) => setForm(current => ({ ...current, [key]: value }))

  useEffect(() => {
    if (previousDistanceUnit.current === distanceUnit) return

    setForm(current => {
      if (current.milesPerDay === '') return current
      const kmPerDay = displayDistanceToKm(current.milesPerDay, previousDistanceUnit.current)
      return {
        ...current,
        milesPerDay: formatEditableNumber(kmToDisplayDistance(kmPerDay, distanceUnit), 1),
      }
    })
    setErrors(current => ({ ...current, milesPerDay: undefined }))
    previousDistanceUnit.current = distanceUnit
  }, [distanceUnit])

  useEffect(() => {
    if (previousTreadUnit.current === treadUnit) return

    setForm(current => {
      if (current.currentTread === '') return current
      const treadMm = displayTreadToMm(current.currentTread, previousTreadUnit.current)
      return {
        ...current,
        currentTread: formatEditableNumber(mmToDisplayTread(treadMm, treadUnit), 1),
      }
    })
    setErrors(current => ({ ...current, currentTread: undefined }))
    previousTreadUnit.current = treadUnit
  }, [treadUnit])

  function resetForm() {
    setForm(DEFAULT_FORM)
    setErrors({})
    setResult(null)
    setApiError('')
  }

  async function handleSubmit(event) {
    event.preventDefault()
    const nextErrors = validate(form, distanceUnit, treadUnit)
    if (Object.keys(nextErrors).length) {
      setErrors(nextErrors)
      return
    }

    setErrors({})
    setApiError('')
    setLoading(true)

    try {
      const prediction = await predictTireLife({
        model: form.model,
        features: buildFeatures(form, distanceUnit, treadUnit),
      })
      setResult(prediction)
    } catch (error) {
      setResult(null)
      setApiError(error.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <form onSubmit={handleSubmit}>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
        <div style={{ gridColumn: '1 / -1' }}>
          <label style={labelStyle}>Backend Model</label>
          <select
            value={form.model}
            onChange={event => set('model', event.target.value)}
            style={{ ...inputStyle(false), cursor: 'pointer' }}
          >
            {availableModels.map(modelName => (
              <option key={modelName} value={modelName}>
                {MODEL_LABELS[modelName] || modelName}
              </option>
            ))}
          </select>
        </div>

        <div style={{ gridColumn: '1 / -1' }}>
          <label style={labelStyle}>Vehicle Model</label>
          <input
            type="text"
            placeholder="Example: Toyota Camry"
            value={form.vehicleModel}
            onChange={event => set('vehicleModel', event.target.value)}
            style={inputStyle(errors.vehicleModel)}
          />
          {errors.vehicleModel && <div style={{ fontSize: 11, color: '#ef4444', marginTop: 3 }}>{errors.vehicleModel}</div>}
        </div>

        <div style={{ gridColumn: '1 / -1' }}>
          <label style={labelStyle}>Tire Brand and Model</label>
          <select
            value={form.brand}
            onChange={event => set('brand', event.target.value)}
            style={{ ...inputStyle(false), cursor: 'pointer' }}
          >
            {Object.keys(TIRE_BRANDS).map(brand => <option key={brand}>{brand}</option>)}
          </select>
        </div>

        <div>
          <label style={labelStyle}>Years Owned</label>
          <input
            type="number"
            min="0"
            step="0.1"
            placeholder="Example: 2"
            value={form.yearsOwned}
            onChange={event => set('yearsOwned', event.target.value)}
            style={inputStyle(errors.yearsOwned)}
          />
          {errors.yearsOwned && <div style={{ fontSize: 11, color: '#ef4444', marginTop: 3 }}>{errors.yearsOwned}</div>}
        </div>

        <div>
          <label style={labelStyle}>{distanceUnit === 'miles' ? 'Miles' : 'Kilometers'} per Day</label>
          <input
            type="number"
            min="0.1"
            step="0.1"
            placeholder={distanceUnit === 'miles' ? 'Example: 30' : 'Example: 48'}
            value={form.milesPerDay}
            onChange={event => set('milesPerDay', event.target.value)}
            style={inputStyle(errors.milesPerDay)}
          />
          {errors.milesPerDay && <div style={{ fontSize: 11, color: '#ef4444', marginTop: 3 }}>{errors.milesPerDay}</div>}
        </div>

        <div>
          <label style={labelStyle}>
            Current Tread Depth ({treadUnit === 'inch32' ? '32nds in' : 'mm'})
          </label>
          <input
            type="number"
            min="0"
            max={treadUnit === 'inch32' ? '20' : '15'}
            step={treadUnit === 'inch32' ? '1' : '0.1'}
            placeholder="Strongly Recommended"
            value={form.currentTread}
            onChange={event => set('currentTread', event.target.value)}
            style={inputStyle(errors.currentTread)}
          />
          {errors.currentTread && <div style={{ fontSize: 11, color: '#ef4444', marginTop: 3 }}>{errors.currentTread}</div>}
        </div>

        <div>
          <label style={labelStyle}>Average Pressure (psi)</label>
          <input
            type="number"
            min="15"
            max="60"
            step="0.5"
            placeholder="Optional"
            value={form.pressure}
            onChange={event => set('pressure', event.target.value)}
            style={inputStyle(errors.pressure)}
          />
          {errors.pressure && <div style={{ fontSize: 11, color: '#ef4444', marginTop: 3 }}>{errors.pressure}</div>}
        </div>

        <div>
          <label style={labelStyle}>Road Condition</label>
          <select
            value={form.roadCondition}
            onChange={event => set('roadCondition', event.target.value)}
            style={{ ...inputStyle(false), cursor: 'pointer' }}
          >
            <option value="smooth">Smooth</option>
            <option value="mixed">Mixed</option>
            <option value="rough">Rough</option>
          </select>
        </div>

        <div>
          <label style={labelStyle}>Weather</label>
          <select
            value={form.weatherCondition}
            onChange={event => set('weatherCondition', event.target.value)}
            style={{ ...inputStyle(false), cursor: 'pointer' }}
          >
            <option value="dry">Dry</option>
            <option value="rainy">Rainy</option>
            <option value="snowy">Snowy</option>
            <option value="mixed">Mixed</option>
          </select>
        </div>
      </div>

      {apiError && (
        <div style={{
          marginTop: '1rem',
          padding: '10px 12px',
          background: '#fef2f2',
          border: '1px solid #fecaca',
          borderRadius: 8,
          color: '#991b1b',
          fontSize: 13,
        }}>
          {apiError}
        </div>
      )}

      <div style={{ marginTop: '1.25rem', display: 'flex', gap: 8 }}>
        <button
          type="submit"
          disabled={loading}
          style={{
            flex: 1,
            padding: '11px',
            background: loading ? '#6b7280' : '#111',
            color: '#fff',
            border: 'none',
            borderRadius: 8,
            fontSize: 14,
            fontWeight: 600,
            cursor: loading ? 'not-allowed' : 'pointer',
            letterSpacing: '0.05em',
          }}
        >
          {loading ? 'Predicting...' : 'Calculate Remaining Life'}
        </button>
        <button
          type="button"
          onClick={resetForm}
          disabled={loading}
          style={{
            width: 82,
            padding: '11px 10px',
            background: '#fff',
            color: '#475569',
            border: '1px solid #d1d5db',
            borderRadius: 8,
            fontSize: 13,
            fontWeight: 700,
            cursor: loading ? 'not-allowed' : 'pointer',
          }}
        >
          Reset
        </button>
      </div>

      <ResultCard prediction={result} distanceUnit={distanceUnit} treadUnit={treadUnit} />
    </form>
  )
}
