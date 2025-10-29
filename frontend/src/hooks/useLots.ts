import { useCallback, useEffect, useState } from "react";
import { recommendLotsForFlight, Lot } from "../stores/session";

export const useLots = (flightId?: string | null, origin?: string | null) => {
  const [lots, setLots] = useState<Lot[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(
    async (id?: string | null, o?: string | null) => {
      const useId = id ?? flightId;
      const useOrigin = o ?? origin ?? undefined;
      if (!useId) return;
      setLoading(true);
      setError(null);
      try {
        const res = await recommendLotsForFlight(useId, useOrigin);
        setLots(res);
      } catch (err: any) {
        setError(err?.message ?? "Error al obtener lotes");
      } finally {
        setLoading(false);
      }
    },
    [flightId, origin]
  );

  useEffect(() => {
    if (flightId) void refresh(flightId, origin ?? null);
  }, [flightId, origin]);

  return { lots, loading, error, refresh } as const;
};

export default null as unknown as void;
