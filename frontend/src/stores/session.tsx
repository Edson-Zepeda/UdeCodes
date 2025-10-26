import React, { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://127.0.0.1:8000";

export type Flight = {
  id: string;
  route: string;
  airline: string;
  date: string;
  time?: string;
  baseline?: number;
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
      selectedLot: parsed.selectedLot ?? null
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
  // try common endpoints (projects may expose different routes). This function centralizes backend calls.
  try {
    const res = await fetch(`${API_BASE}/flights`);
    if (res.ok) return (await res.json()) as Flight[];
  } catch (err) {
    // ignore and try fallback
  }

  try {
    const res = await fetch(`${API_BASE}/simulate/generative`);
    if (res.ok) {
      const json = await res.json();
      // assume response contains flights list
      return json.flights ?? json;
    }
  } catch (err) {
    // ignore
  }

  // fallback sample
  return [
    { id: "AM109", route: "MEX â†’ DOH", airline: "AeroMÃ©xico", date: "27 Oct 2025", time: "08:45" },
    { id: "LH432", route: "FRA â†’ JFK", airline: "Lufthansa", date: "27 Oct 2025", time: "13:20" },
    { id: "BA249", route: "LHR â†’ GRU", airline: "British Airways", date: "27 Oct 2025", time: "18:10" }
  ];
};

export const recommendLotsForFlight = async (flight_id: string): Promise<Lot[]> => {
  try {
    const res = await fetch(`${API_BASE}/lots/recommend?flight_id=${encodeURIComponent(flight_id)}`);
    if (res.ok) return (await res.json()) as Lot[];
  } catch (err) {
    // ignore
  }

  // fallback sample
  return [
    { id: "LOT-9821", lot: "B47-2025", product: "Chips ClÃ¡sicas 50g", expiry: "12 Nov 2025", bestOption: true, origin: "MEX" },
    { id: "LOT-1042", lot: "C12-2025", product: "Energy Bar 60g", expiry: "18 Nov 2025", origin: "DOH" }
  ];
};

export default null as unknown as void;

