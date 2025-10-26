import { useCallback, useEffect, useState } from "react";
import { fetchFlightsFromBackend } from "../stores/session";
import { useSession } from "../stores/session";

export const useFlights = () => {
  const { flights, setFlights } = useSession();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchFlightsFromBackend();
      setFlights(data);
    } catch (err: any) {
      setError(err?.message ?? "Error al obtener vuelos");
    } finally {
      setLoading(false);
    }
  }, [setFlights]);

  useEffect(() => {
    if (!flights || flights.length === 0) void refresh();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  return { flights, loading, error, refresh } as const;
};

export default null as unknown as void;
