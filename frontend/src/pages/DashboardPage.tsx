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
