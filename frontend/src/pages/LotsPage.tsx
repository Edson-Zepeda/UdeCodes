import { useMemo, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useSession } from "../stores/session";
import { useLots } from "../hooks/useLots";

interface LotCard {
  id: string;
  lot: string;
  product: string;
  expiry: string;
  bestOption?: boolean;
  origin: string;
}

const sampleLots: LotCard[] = [
  { id: "LOT-9821", lot: "B47-2025", product: "Chips Clásicas 50g", expiry: "12 Nov 2025", bestOption: true, origin: "MEX" },
  { id: "LOT-1042", lot: "C12-2025", product: "Energy Bar 60g", expiry: "18 Nov 2025", origin: "DOH" },
  { id: "LOT-1183", lot: "A03-2025", product: "Jugos Premium 250ml", expiry: "26 Nov 2025", origin: "LHR" },
  { id: "LOT-1267", lot: "E08-2025", product: "Galletas Integrales", expiry: "02 Dic 2025", origin: "ZRH" }
];

const LotsPage = () => {
  const navigate = useNavigate();
  const { selectedFlight, setSelectedLot } = useSession();
  const { lots, loading } = useLots(selectedFlight?.id ?? null, selectedFlight?.origin ?? null);

  useEffect(() => {
    if (!selectedFlight) {
      navigate("/");
    }
  }, [selectedFlight]);

  const availableLots = lots && lots.length > 0 ? lots : sampleLots;
  const [message, setMessage] = useState<string | null>(null);

  const exportFEFO = () => {
    try {
      const rows = availableLots.map((l) => ({
        lot_id: (l as any).id ?? (l as any).lot ?? "",
        lot: (l as any).lot ?? (l as any).lot_number ?? "",
        product: (l as any).product ?? (l as any).product_name ?? "",
        expiry: (l as any).expiry ?? (l as any).expiry_date ?? "",
        origin: (l as any).origin ?? "",
        recommended: String((l as any).bestOption ?? (l as any).recommended ?? false),
        unit_cost: (l as any).unit_cost ?? "",
        quantity_consumed: (l as any).quantity_consumed ?? "",
      }));

      if (!rows.length) return;

      const headers = Object.keys(rows[0]);
      const csv = [headers.join(",")].concat(rows.map((r) => headers.map((h) => {
        const v = r[h as keyof typeof r];
        if (v === null || v === undefined) return "";
        const s = String(v).replace(/"/g, '""');
        return `"${s}"`;
      }).join(",")) ).join("\n");

      const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      const flightTag = selectedFlight?.id ?? "noflight";
      a.href = url;
      a.download = `fefo_plan_${flightTag}.csv`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Export FEFO failed", err);
      setMessage("No se pudo exportar el plan FEFO.");
      setTimeout(() => setMessage(null), 3000);
    }
  };

  return (
    <div className="page-wrapper">
      <section className="hero" style={{ padding: "36px 40px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 20 }}>
          <div>
            <h1 style={{ fontSize: "1.9rem", margin: 0 }}>Optimización de lotes y caducidad</h1>
            <p style={{ maxWidth: "540px", marginTop: 8 }}>
              Priorizamos embarques bajo la estrategia FEFO y presentamos recomendaciones con impacto financiero. Selecciona un lote para asignarlo al vuelo seleccionado.
            </p>
            <div style={{ marginTop: 14 }}>
              <strong>Vuelo seleccionado:</strong> {selectedFlight ? `${selectedFlight.id} · ${selectedFlight.route} (${selectedFlight.airline})` : "Ninguno"}
            </div>
          </div>
          <div style={{ display: "flex", gap: 12 }}>
            <button className="primary-button" style={{ background: "rgba(162, 133, 93, 0.88)" }} onClick={exportFEFO}>
              Exportar plan FEFO
            </button>
          </div>
        </div>
      </section>

      <section style={{ marginTop: "28px" }}>
        <div className="section-heading">
          <h2>Lotes priorizados</h2>
          <span style={{ fontSize: "0.85rem", color: "rgba(244,246,251,0.6)" }}>Ordenados por urgencia y compatibilidad con vuelo</span>
        </div>

        {message && (
          <div style={{ marginBottom: 12, padding: 12, borderRadius: 8, background: "rgba(162,133,93,0.12)", color: "#fff" }}>{message}</div>
        )}

        <div className="card-grid lots-grid">
          {loading && <div style={{ color: "rgba(244,246,251,0.6)" }}>Buscando lotes recomendados…</div>}
          {!loading && availableLots.map((lot: any) => (
            <article key={(lot.product_id ?? lot.id ?? Math.random()) as any} className="lot-card">
              <div className="lot-card__image" />
              {((lot.recommended ?? lot.bestOption) as boolean) && <span className="lot-card__badge">Mejor opción</span>}
              <div className="lot-card__content">
                <p className="lot-card__title">{lot.product_name ?? lot.product}</p>
                <p className="lot-card__meta"><strong>Lote:</strong> {lot.lot_number ?? lot.lot ?? (lot.id ?? "-")}</p>
                <p className="lot-card__meta"><strong>Caducidad:</strong> {lot.expiry_date ?? lot.expiry ?? "-"}</p>
                <p className="lot-card__meta"><strong>Origen:</strong> {lot.origin ?? "-"}</p>
                <p className="lot-card__meta"><strong>Spec estándar:</strong> {lot.standard_spec_qty ?? "-"}</p>
                <p className="lot-card__meta"><strong>Consumo esperado:</strong> {lot.quantity_consumed ?? "-"}</p>
                <p className="lot-card__meta"><strong>Precio unitario:</strong> {lot.unit_cost ? `$${Number(lot.unit_cost).toFixed(2)}` : "-"}</p>
                {lot.crew_feedback && <p className="lot-card__meta"><em>{lot.crew_feedback}</em></p>}
                <button
                  className="primary-button"
                  style={{ marginTop: "14px", width: "100%", background: "rgba(62,81,112,0.85)" }}
                  onClick={() => {
                    setSelectedLot(lot as any);
                    setMessage("Lote asignado al vuelo");
                    setTimeout(() => setMessage(null), 2200);
                    navigate("/dashboard");
                  }}
                >
                  Asignar al vuelo
                </button>
              </div>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
};

export default LotsPage;

