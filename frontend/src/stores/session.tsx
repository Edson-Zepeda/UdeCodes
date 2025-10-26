import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://127.0.0.1:8000";

export type Flight = {
  id: string;
  route: string;
  airline: string;
  date: string;
  time?: string;
  baseline?: number;
  origin?: string;
};

export type Lot = {
  id: string;
  lot: string;
  product: string;
  expiry: string;
  origin?: string;
  bestOption?: boolean;
};

type SessionState = {
  flights: Flight[];
  selectedFlight: Flight | null;
  selectedLot: Lot | null;
  setFlights: (f: Flight[]) => void;
  setSelectedFlight: (f: Flight | null) => void;
  setSelectedLot: (l: Lot | null) => void;
  clear: () => void;
};

const STORAGE_KEY = "spir-session";

const SessionContext = createContext<SessionState | undefined>(undefined);

const readInitial = (): { flights: Flight[]; selectedFlight: Flight | null; selectedLot: Lot | null } => {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return { flights: [], selectedFlight: null, selectedLot: null };
    const parsed = JSON.parse(raw);
    return {
      flights: parsed.flights ?? [],
      selectedFlight: parsed.selectedFlight ?? null,
      selectedLot: parsed.selectedLot ?? null,
    };
  } catch (err) {
    console.warn("Failed reading session from storage", err);
    return { flights: [], selectedFlight: null, selectedLot: null };
  }
};

export const SessionProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const initial = useMemo(() => readInitial(), []);

  const [flights, setFlightsState] = useState<Flight[]>(initial.flights);
  const [selectedFlight, setSelectedFlightState] = useState<Flight | null>(initial.selectedFlight);
  const [selectedLot, setSelectedLotState] = useState<Lot | null>(initial.selectedLot);

  useEffect(() => {
    const payload = { flights, selectedFlight, selectedLot };
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
    } catch (err) {
      // ignore
    }
  }, [flights, selectedFlight, selectedLot]);

  const setFlights = useCallback((f: Flight[]) => setFlightsState(f), []);
  const setSelectedFlight = useCallback((f: Flight | null) => setSelectedFlightState(f), []);
  const setSelectedLot = useCallback((l: Lot | null) => setSelectedLotState(l), []);
  const clear = useCallback(() => {
    setFlightsState([]);
    setSelectedFlightState(null);
    setSelectedLotState(null);
    try {
      localStorage.removeItem(STORAGE_KEY);
    } catch (err) {
      // ignore
    }
  }, []);

  return (
    <SessionContext.Provider value={{ flights, selectedFlight, selectedLot, setFlights, setSelectedFlight, setSelectedLot, clear }}>
      {children}
    </SessionContext.Provider>
  );
};

export const useSession = (): SessionState => {
  const ctx = useContext(SessionContext);
  if (!ctx) throw new Error("useSession must be used within SessionProvider");
  return ctx;
};

export const fetchFlightsFromBackend = async (): Promise<Flight[]> => {
  // Primary API: backend dataset-based list
  try {
    const res = await fetch(`${API_BASE}/flights/list`);
    if (res.ok) {
      const json = await res.json();
      const flights = (json?.flights ?? []).map((f: any) => ({
        id: f.flight_id ?? f.id,
        route: f.route,
        airline: f.airline,
        date: f.date,
        time: f.departure_time,
        origin: f.origin,
      })) as Flight[];
      return flights;
    }
  } catch {}

  // Demo fallback
  try {
    const res = await fetch(`${API_BASE}/flights`);
    if (res.ok) {
      const json = await res.json();
      const flights = (json?.flights ?? json ?? []).map((f: any) => ({
        id: f.flight_id ?? f.id,
        route: f.route,
        airline: f.airline,
        date: f.date,
        time: f.departure_time ?? f.time,
        origin: f.origin,
      })) as Flight[];
      return flights;
    }
  } catch {}

  // Static fallback
  return [
    { id: "AM109", route: "MEX - DOH", airline: "AeroMexico", date: "27 Oct 2025", time: "08:45", origin: "MEX" },
    { id: "LH432", route: "FRA - JFK", airline: "Lufthansa", date: "27 Oct 2025", time: "13:20", origin: "FRA" },
    { id: "BA249", route: "LHR - GRU", airline: "British Airways", date: "27 Oct 2025", time: "18:10", origin: "LHR" },
  ];
};

export const recommendLotsForFlight = async (flight_id: string, origin?: string): Promise<Lot[]> => {
  try {
    const url = origin
      ? `${API_BASE}/lots/recommend?flight_id=${encodeURIComponent(flight_id)}&origin=${encodeURIComponent(origin)}`
      : `${API_BASE}/lots/recommend?flight_id=${encodeURIComponent(flight_id)}`;
    const res = await fetch(url);
    if (res.ok) {
      const json = await res.json();
      const lots = (json?.lots ?? json ?? []).map((l: any, idx: number) => ({
        id: l.product_id ?? `LOT-${idx + 1}`,
        lot: l.lot_number ?? l.lot ?? "",
        product: l.product_name ?? l.product ?? "",
        expiry: l.expiry_date ?? l.expiry ?? "",
        origin,
        bestOption: Boolean(l.recommended),
      })) as Lot[];
      return lots;
    }
  } catch {}

  // fallback sample
  return [
    { id: "LOT-9821", lot: "B47-2025", product: "Chips Clasicas 50g", expiry: "12 Nov 2025", bestOption: true, origin: origin ?? "MEX" },
    { id: "LOT-1042", lot: "C12-2025", product: "Energy Bar 60g", expiry: "18 Nov 2025", origin: origin ?? "DOH" },
  ];
};

export default null as unknown as void;

