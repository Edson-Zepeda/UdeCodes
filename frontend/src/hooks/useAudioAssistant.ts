import { useCallback, useRef, useState } from "react";
import { useToast } from "../components/ToastProvider";

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? "http://127.0.0.1:8000";

const STREAM_HEADERS = {
  "Content-Type": "application/json"
} as const;

interface Options {
  defaultEnabled?: boolean;
}

const useAudioAssistant = ({ defaultEnabled = true }: Options = {}) => {
  const [enabled, setEnabled] = useState(defaultEnabled);
  const playbackRef = useRef<HTMLAudioElement | null>(null);
  const requestRef = useRef<AbortController | null>(null);
  const toast = useToast();
  const speechSupported = typeof window !== "undefined" && "speechSynthesis" in window;

  const stopPlayback = useCallback(() => {
    playbackRef.current?.pause();
    playbackRef.current?.removeAttribute("src");
    playbackRef.current?.load();
    playbackRef.current = null;
    requestRef.current?.abort();
    requestRef.current = null;
    // cancel any ongoing SpeechSynthesis utterance
    try {
      if (typeof window !== "undefined" && "speechSynthesis" in window) {
        window.speechSynthesis.cancel();
      }
    } catch (e) {
      // ignore
    }
  }, []);

  const streamAndPlay = useCallback(
    async (endpoint: string, body: Record<string, unknown>) => {
      if (!enabled) {
        return;
      }

      // show a small toast indicating audio action started
      try {
        toast.show("Reproduciendo audio…", "info");
      } catch (e) {
        // no-op if toast provider missing
      }

      requestRef.current?.abort();
      const controller = new AbortController();
      requestRef.current = controller;

      try {
        const response = await fetch(`${API_BASE}${endpoint}`, {
          method: "POST",
          headers: STREAM_HEADERS,
          body: JSON.stringify(body),
          signal: controller.signal
        });

        if (!response.ok) {
          throw new Error(`Audio request failed: ${response.statusText}`);
        }

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        playbackRef.current = audio;
        audio.play()
          .then(() => {
            try {
              toast.show("Reproducción iniciada", "success");
            } catch (e) {
              // ignore
            }
          })
          .catch(() => {
            // Autoplay restrictions
            console.warn("Reproducción bloqueada por el navegador. Se requiere interacción del usuario.");
            try {
              toast.show("Interacción requerida para reproducir audio", "error");
            } catch (e) {}
          });
        audio.onended = () => {
          URL.revokeObjectURL(url);
          if (playbackRef.current === audio) {
            playbackRef.current = null;
          }
        };
      } catch (error) {
        if (!(error instanceof DOMException && error.name === "AbortError")) {
          console.warn("No se pudo reproducir audio:", error);
        }
      }
    },
    [enabled]
  );

  const speak = useCallback(
    async (text: string) => {
      if (!text.trim()) return;

      // Prefer client-side SpeechSynthesis for real spoken audio when available
      try {
        if (typeof window !== "undefined" && "speechSynthesis" in window && enabled) {
          // cancel any previous
          window.speechSynthesis.cancel();
          const utter = new SpeechSynthesisUtterance(text);
          // try to prefer Spanish voice if available
          try {
            const voices = window.speechSynthesis.getVoices();
            const spanish = voices.find((v) => /es(-|_)?/i.test(v.lang) || /spanish/i.test(v.name));
            if (spanish) utter.voice = spanish;
          } catch (e) {
            // ignore voice selection errors
          }

          utter.lang = utter.lang || "es-ES";
          utter.onstart = () => {
            try {
              toast.show("Narración iniciada", "success");
            } catch (e) {}
          };
          utter.onend = () => {
            try {
              toast.show("Narración finalizada", "info");
            } catch (e) {}
          };
          utter.onerror = (ev) => {
            console.warn("SpeechSynthesis error", ev);
            try {
              toast.show("Error en la narración", "error");
            } catch (e) {}
          };

          window.speechSynthesis.speak(utter);
          return;
        }
      } catch (err) {
        console.warn("SpeechSynthesis not available or failed:", err);
      }

      // Fallback: ask backend to return audio blob
      await streamAndPlay("/assist/speak", { text });
    },
    [streamAndPlay, enabled, toast]
  );

  const playEffect = useCallback(
    async (prompt: string, durationSeconds = 0.5) => {
      if (!prompt.trim()) return;
      await streamAndPlay("/assist/sound", { prompt, duration_seconds: durationSeconds });
    },
    [streamAndPlay]
  );

  const toggleEnabled = useCallback(
    (value: boolean) => {
      setEnabled(value);
      if (!value) {
        stopPlayback();
      }
    },
    [stopPlayback]
  );

  return {
    enabled,
    speak,
    playEffect,
    toggleEnabled
    ,speechSupported
  };
};

export default useAudioAssistant;
