import { useMemo } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useFlights } from "../hooks/useFlights";
import { useSession } from "../stores/session";

const stats = [
  {
    title: "Ahorro por combustible",
    description: "Reducción de peso no utilizado en trayectos de largo alcance",
    value: "$2.4M MXN"
  },
  {
    title: "Ahorro en desperdicio",
    description: "Optimización de la carga con buffer dinámico por vuelo",
    value: "$1.1M MXN"
  },
  {
    title: "Ingresos recuperados",
    description: "Prevención de faltantes retail identificados por la tripulación",
    value: "$860K MXN"
  }
];

const HomePage = () => {
  const heroKpis = useMemo(() => stats, []);
  const { flights, loading } = useFlights();
  const { setSelectedFlight } = useSession();
  const navigate = useNavigate();

  const handleSelectFlight = (flight: any) => {
    setSelectedFlight(flight);
    navigate("/lots");
  };

  return (
    <div className="page-wrapper">
      <section className="hero">
        <h1>Welcome to SPIR</h1>
        <p>Sistema de Predicción Inteligente de Recursos. Combina forecasting, simulaciones y análisis financiero en un solo lugar.</p>
        <div className="hero__actions">
          <Link to="/dashboard" className="primary-button">
            Ir al simulador financiero
          </Link>
          <Link to="/lots" className="primary-button" style={{ background: "rgba(162, 133, 93, 0.85)" }}>
            Optimizar lotes
          </Link>
        </div>
      </section>

      <section style={{ marginTop: "48px" }}>
        <div className="section-heading">
          <h2>Impacto Estratégico</h2>
          <span style={{ color: "rgba(244,246,251,0.5)", fontSize: "0.85rem" }}>Resultados del piloto en 4 hubs críticos</span>
        </div>
        <div className="card-grid flight-grid">
          {heroKpis.map((kpi) => (
            <article key={kpi.title} className="flight-card" style={{ padding: "24px 26px" }}>
              <span className="flight-card__label">{kpi.title}</span>
              <p className="flight-card__route" style={{ fontSize: "1.6rem", margin: "12px 0" }}>
                {kpi.value}
              </p>
              <p style={{ color: "rgba(244,246,251,0.68)", fontSize: "0.9rem", lineHeight: 1.5 }}>{kpi.description}</p>
            </article>
          ))}
        </div>
      </section>

      <section style={{ marginTop: "48px" }}>
        <div className="section-heading">
          <h2>Próximos vuelos monitoreados</h2>
          <Link to="/dashboard" className="nav-bar__link">
            Ver recomendaciones →
          </Link>
        </div>
        <div className="card-grid flight-grid">
          {loading && <div style={{ color: "rgba(244,246,251,0.6)" }}>Cargando vuelos…</div>}
          {!loading && flights.map((flight) => (
            <article key={flight.id} className="flight-card" onClick={() => handleSelectFlight(flight)} style={{ cursor: "pointer" }}>
              <span className="flight-card__label">{flight.airline}</span>
              <p className="flight-card__route">{flight.route}</p>
              <div className="flight-card__meta">
                <span>{flight.date}</span>
                <span>{flight.time ?? "--"}h</span>
              </div>
              <div className="flight-card__meta" style={{ marginTop: "auto", fontSize: "0.8rem" }}>
                <span>ID vuelo: {flight.id}</span>
              </div>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
};

export default HomePage;
