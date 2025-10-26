import { useCallback, useEffect, useState } from "react";
import { recommendLotsForFlight, Lot } from "../stores/session";

export const useLots = (flightId?: string | null) => {
  const [lots, setLots] = useState<Lot[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(
    async (id?: string | null) => {
      const useId = id ?? flightId;
      if (!useId) return;
      setLoading(true);
      setError(null);
      try {
        const res = await recommendLotsForFlight(useId);
        setLots(res);
      } catch (err: any) {
        setError(err?.message ?? "Error al obtener lotes");
      } finally {
        setLoading(false);
      }
    },
    [flightId]
  );

  useEffect(() => {
    if (flightId) void refresh(flightId);
  }, [flightId]);

  return { lots, loading, error, refresh } as const;
};

export default null as unknown as void;
