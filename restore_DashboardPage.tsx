import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Bar } from "react-chartjs-2";
import {
  BarElement,
  CategoryScale,
  Chart as ChartJS,
  Legend,
  LinearScale,
  Tooltip,
  Filler
} from "chart.js";

import useAudioAssistant from "../hooks/useAudioAssistant";
import { useSession } from "../stores/session";
import { useNavigate } from "react-router-dom";
import formatCurrency from "../utils/formatCurrency";

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Legend, Filler);

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://127.0.0.1:8000";

const SOUND_PROMPTS = {
  success: "positive confirmation chime with futuristic tone",
  warning: "soft notification alert tone"
} as const;

type FinancialDetail = {
  flight_id: string;
  product_id: string;
  product_name?: string | null;
  unit_cost: number;
  recommended_load: number;
  baseline_returns: number;
  spir_returns: number;
  weight_difference_units?: number;
  fuel_weight_saved_kg?: number;
};

type FinancialImpact = {
  assumptions: {
    fuel_cost_per_liter: number;
    fuel_burn_liters_per_kg: number;
    buffer_factor: number;
    waste_cost_multiplier: number;
    unit_margin_factor: number;
  };
  waste_cost_baseline: number;
  waste_cost_spir: number;
  waste_savings: number;
  fuel_weight_reduction_kg: number;
  fuel_cost_savings: number;
  recovered_retail_value: number;
  total_impact: number;
  details?: FinancialDetail[];
};

const DEFAULT_ASSUMPTIONS = {
  waste_cost_multiplier: 1.0,
  fuel_cost_per_liter: 25,
  unit_margin_factor: 1.0,
  buffer_factor: 1.05
};

const DashboardPage = () => {
  const [assumptions, setAssumptions] = useState(DEFAULT_ASSUMPTIONS);
  const [data, setData] = useState<FinancialImpact | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [detailsLimit] = useState(8);

  const previousImpact = useRef<number | null>(null);
  const { enabled, toggleEnabled, speak, playEffect, speechSupported } = useAudioAssistant({ defaultEnabled: true });

  const chartData = useMemo(() => {
    const baseline = (data?.waste_cost_baseline ?? 0) * assumptions.waste_cost_multiplier;
    const spir = (data?.waste_cost_spir ?? 0) * assumptions.waste_cost_multiplier;
    return {
      labels: ["Costo medio actual", "Costo medio SPIR"],
      datasets: [
        {
          label: "Costo",
          data: [baseline, spir],
          backgroundColor: ["rgba(162, 133, 93, 0.85)", "rgba(255, 255, 255, 0.85)"],
          borderRadius: 14,
          borderSkipped: false
        }
      ]
    };
  }, [data, assumptions.waste_cost_multiplier]);

  const chartOptions = useMemo(
    () => ({
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (context: any) => ` ${formatCurrency(context.parsed.y)}`
          }
        }
      },
      scales: {
        y: {
          beginAtZero: true,
          ticks: {
            color: "rgba(244,246,251,0.6)",
            callback: (value: number) => `$${(value / 1000).toFixed(1)}K`
          },
          grid: { color: "rgba(244,246,251,0.08)" }
        },
        x: {
          ticks: { color: "rgba(244,246,251,0.8)" },
          grid: { display: false }
        }
      }
    }),
    []
  );

  const { selectedFlight, selectedLot } = useSession();
  const navigate = useNavigate();

  const fetchFinancialImpact = useCallback(async () => {
    setLoading(true);
    setError(null);
    if (!selectedFlight) {
      // if no flight selected, go back to home
      navigate("/");
      setLoading(false);
      return;
    }

    try {
      const response = await fetch(`${API_BASE}/predict/financial-impact`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          fuel_cost_per_liter: assumptions.fuel_cost_per_liter,
          waste_cost_multiplier: assumptions.waste_cost_multiplier,
          unit_margin_factor: assumptions.unit_margin_factor,
          buffer_factor: assumptions.buffer_factor,
          include_details: true,
          max_details: detailsLimit,
          context: {
            flight_id: selectedFlight?.id,
            lot_id: selectedLot?.id
          }
        })
      });

      if (!response.ok) {
        throw new Error(response.statusText);
      }

      const json: FinancialImpact = await response.json();
      setData(json);

      const previous = previousImpact.current;
      previousImpact.current = json.total_impact;

      const summary = `Con los supuestos actuales, el impacto total estimado es de ${formatCurrency(json.total_impact)}. 
        El ahorro por desperdicio alcanza ${formatCurrency(json.waste_savings)} 
        y los ingresos recuperados ${formatCurrency(json.recovered_retail_value)}.`;

      if (previous === null) {
        await speak(summary);
      } else {
        const delta = json.total_impact - previous;
        if (Math.abs(delta) > 5000) {
          await speak(
            `Actualizaci├│n r├ípida: el impacto anual cambi├│ ${delta > 0 ? "positivamente" : "negativamente"} a ${formatCurrency(
              json.total_impact
            )}.`
          );
          if (delta > 0) {
            await playEffect(SOUND_PROMPTS.success);
          } else {
            await playEffect(SOUND_PROMPTS.warning, 0.6);
          }
        }
      }
    } catch (err) {
      console.error(err);
      setError("No se pudo obtener el impacto financiero.");
    } finally {
      setLoading(false);
    }
  }, [assumptions, detailsLimit, playEffect, speak]);

  useEffect(() => {
    const timeout = setTimeout(() => {
      void fetchFinancialImpact();
    }, 250);
    return () => clearTimeout(timeout);
  }, [fetchFinancialImpact]);

  const handleSliderChange = (key: keyof typeof DEFAULT_ASSUMPTIONS) => (event: React.ChangeEvent<HTMLInputElement>) => {
    const value = Number(event.target.value);
    setAssumptions((prev) => ({ ...prev, [key]: value }));
  };

  const handlePreviewSummary = async () => {
    if (!data) return;
    const summary = `SPIR reporta un ahorro total de ${formatCurrency(
      data.total_impact
    )}, con ${formatCurrency(data.fuel_cost_savings)} en combustible 
    y ${formatCurrency(data.recovered_retail_value)} en ventas recuperadas.`;
    await speak(summary);
  };

  const handlePreviewSuccess = async () => {
    await playEffect(SOUND_PROMPTS.success);
  };

  return (
    <div className="page-wrapper">
      <section className="hero" style={{ padding: "40px 48px", marginBottom: "36px" }}>
        <h1>Simulador financiero SPIR</h1>
        <p>Configura los supuestos clave, cuantifica el ahorro en tiempo real y demuestra el valor de negocio de SPIR.</p>
          <div className="hero__actions">
            <button
              className="primary-button"
              onClick={handlePreviewSummary}
              disabled={!speechSupported || !enabled}
              title={!speechSupported ? "Narraci├│n no soportada en este navegador" : undefined}
            >
              Narrar hallazgos
            </button>
            <button className="primary-button" style={{ background: "rgba(162, 133, 93, 0.9)" }} onClick={handlePreviewSuccess}>
              Sonido de ├®xito
            </button>
          </div>
      </section>

      <section className="card-grid flight-grid" style={{ marginBottom: "32px" }}>
        <article className="flight-card" style={{ padding: "22px 26px" }}>
          <span className="flight-card__label">Ahorro por combustible</span>
          <p className="flight-card__route" style={{ fontSize: "1.8rem", margin: "16px 0" }}>
            {formatCurrency(data?.fuel_cost_savings ?? 0)}
          </p>
          <p style={{ color: "rgba(244,246,251,0.6)", fontSize: "0.88rem" }}>Derivado de la reducci├│n de peso cargado innecesariamente.</p>
        </article>
        <article className="flight-card" style={{ padding: "22px 26px" }}>
          <span className="flight-card__label">Ahorro anual estimado</span>
          <p className="flight-card__route" style={{ fontSize: "1.8rem", margin: "16px 0" }}>
            {formatCurrency(data?.total_impact ?? 0)}
          </p>
          <p style={{ color: "rgba(244,246,251,0.6)", fontSize: "0.88rem" }}>Incluye desperdicio, combustible e ingresos retail recuperados.</p>
        </article>
        <article className="flight-card" style={{ padding: "22px 26px" }}>
          <span className="flight-card__label">Ahorro directo por desperdicio</span>
          <p className="flight-card__route" style={{ fontSize: "1.8rem", margin: "16px 0" }}>
            {formatCurrency(data?.waste_savings ?? 0)}
          </p>
          <p style={{ color: "rgba(244,246,251,0.6)", fontSize: "0.88rem" }}>Calculado con el multiplicador de coste por comida desperdiciada.</p>
        </article>
      </section>

      <div className="card" style={{ borderRadius: "20px", background: "var(--color-surface-alt)", boxShadow: "var(--shadow-soft)" }}>
        <div style={{ display: "flex", gap: "36px", flexWrap: "wrap" }}>
          <div style={{ flex: "1 1 380px", minWidth: "320px" }}>
            <p className="graph-title" style={{ marginBottom: "12px" }}>
              Actual vs SPIR
            </p>
            <div style={{ height: "280px" }}>
              <Bar data={chartData} options={chartOptions} />
            </div>
          </div>
          <div style={{ flex: "1 1 320px", minWidth: "300px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "18px" }}>
              <h3 style={{ margin: 0, fontSize: "1rem" }}>Controles del simulador</h3>
              <label className="toggle" title="Activar narraci├│n y sonidos">
                <input type="checkbox" checked={enabled} onChange={(event) => toggleEnabled(event.target.checked)} />
                <span className="toggle-slider" />
              </label>
            </div>

            <div className="slider-group">
              <span className="slider-label">Costo medio por comida desperdiciada</span>
              <span className="slider-qty-label">{assumptions.waste_cost_multiplier.toFixed(2)}x</span>
              <input
                type="range"
                className="slider"
                min={0.5}
                max={2}
                step={0.05}
                value={assumptions.waste_cost_multiplier}
                onChange={handleSliderChange("waste_cost_multiplier")}
              />
            </div>

            <div className="slider-group">
              <span className="slider-label">Precio del combustible (MXN/L)</span>
              <span className="slider-qty-label">${assumptions.fuel_cost_per_liter.toFixed(0)}</span>
              <input
                type="range"
                className="slider"
                min={10}
                max={60}
                step={1}
                value={assumptions.fuel_cost_per_liter}
                onChange={handleSliderChange("fuel_cost_per_liter")}
              />
            </div>

            <div className="slider-group">
              <span className="slider-label">Margen retail</span>
              <span className="slider-qty-label">{assumptions.unit_margin_factor.toFixed(1)}x</span>
              <input
                type="range"
                className="slider"
                min={0.5}
                max={4}
                step={0.1}
                value={assumptions.unit_margin_factor}
                onChange={handleSliderChange("unit_margin_factor")}
              />
            </div>

            <div className="slider-group" style={{ borderBottom: "none" }}>
              <span className="slider-label">Buffer de seguridad</span>
              <span className="slider-qty-label">{assumptions.buffer_factor.toFixed(2)}x</span>
              <input
                type="range"
                className="slider"
                min={1}
                max={1.3}
                step={0.01}
                value={assumptions.buffer_factor}
                onChange={handleSliderChange("buffer_factor")}
              />
            </div>

            {loading && <p style={{ color: "rgba(244,246,251,0.7)", marginTop: "16px" }}>Calculando impactoÔÇª</p>}
            {error && (
              <p style={{ color: "#ff7b7b", marginTop: "16px" }}>
                {error} <br /> Revisa que el backend est├® corriendo en {API_BASE}.
              </p>
            )}
          </div>
        </div>

        <div style={{ marginTop: "32px" }}>
          <h3 style={{ marginBottom: "12px", fontSize: "1rem" }}>Top oportunidades por vuelo</h3>
          <div
            style={{
              overflowX: "auto",
              borderRadius: "12px",
              border: "1px solid rgba(255,255,255,0.05)"
            }}
          >
            <table style={{ width: "100%", borderCollapse: "collapse", minWidth: "720px" }}>
              <thead>
                <tr style={{ textAlign: "left", background: "rgba(255,255,255,0.04)" }}>
                  {["Vuelo", "Producto", "Spec est├índar", "Carga recomendada", "Devoluciones actuales", "Devoluciones con SPIR"].map((header) => (
                    <th key={header} style={{ padding: "12px 16px", fontSize: "0.78rem", letterSpacing: "0.08em", textTransform: "uppercase" }}>
                      {header}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data?.details?.slice(0, detailsLimit).map((detail) => (
                  <tr key={`${detail.flight_id}-${detail.product_id}`} style={{ borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
                    <td style={{ padding: "12px 16px", fontWeight: 600 }}>{detail.flight_id}</td>
                    <td style={{ padding: "12px 16px" }}>{detail.product_name ?? detail.product_id}</td>
                    <td style={{ padding: "12px 16px" }}>{(detail.recommended_load + detail.spir_returns).toFixed(1)}</td>
                    <td style={{ padding: "12px 16px" }}>{detail.recommended_load.toFixed(1)}</td>
                    <td style={{ padding: "12px 16px" }}>{detail.baseline_returns.toFixed(1)}</td>
                    <td style={{ padding: "12px 16px" }}>{detail.spir_returns.toFixed(1)}</td>
                  </tr>
                ))}
                {!data?.details?.length && (
                  <tr>
                    <td colSpan={6} style={{ padding: "16px", textAlign: "center", color: "rgba(244,246,251,0.6)" }}>
                      Ejecuta el notebook de consumo o habilita la opci├│n <strong>include_details</strong> para ver recomendaciones espec├¡ficas.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DashboardPage;
