"use client";
import { useCallback, useEffect, useLayoutEffect, useState } from "react";
import { createPortal } from "react-dom";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { X } from "lucide-react";

export interface TourStep {
  /** Matches a `data-tour="..."` attribute on the element to spotlight. */
  target: string;
  title: string;
  body: string;
}

interface Rect {
  top: number;
  left: number;
  width: number;
  height: number;
}

const PAD = 8;

/**
 * A lightweight spotlight product tour. Auto-starts once per `id` (localStorage),
 * and can be re-triggered by dispatching `window` event `start-tour` with
 * `{ detail: { id } }`. Reduced-motion aware; no dependencies beyond motion.
 */
export function ProductTour({
  id,
  steps,
  autoStart = true,
}: {
  id: string;
  steps: TourStep[];
  autoStart?: boolean;
}) {
  const reduce = useReducedMotion();
  const [mounted, setMounted] = useState(false);
  const [active, setActive] = useState(false);
  const [index, setIndex] = useState(0);
  const [rect, setRect] = useState<Rect | null>(null);

  const storageKey = `tour:${id}`;

  useEffect(() => setMounted(true), []);

  const start = useCallback(() => {
    setIndex(0);
    setActive(true);
  }, []);

  const finish = useCallback(() => {
    setActive(false);
    try {
      localStorage.setItem(storageKey, "1");
    } catch {
      /* no-op */
    }
  }, [storageKey]);

  // Auto-start on first visit + listen for manual replay.
  useEffect(() => {
    if (!mounted) return;
    let seen = false;
    try {
      seen = !!localStorage.getItem(storageKey);
    } catch {
      /* no-op */
    }
    let t: ReturnType<typeof setTimeout> | undefined;
    if (autoStart && !seen) t = setTimeout(start, 650);

    const onStart = (e: Event) => {
      const detail = (e as CustomEvent).detail;
      if (!detail || detail.id === id) start();
    };
    window.addEventListener("start-tour", onStart);
    return () => {
      if (t) clearTimeout(t);
      window.removeEventListener("start-tour", onStart);
    };
  }, [mounted, autoStart, start, storageKey, id]);

  // Measure the current target (following scroll/resize).
  const measure = useCallback(() => {
    const step = steps[index];
    if (!step) return;
    const el = document.querySelector<HTMLElement>(`[data-tour="${step.target}"]`);
    if (!el) {
      setRect(null);
      return;
    }
    const r = el.getBoundingClientRect();
    setRect({ top: r.top, left: r.left, width: r.width, height: r.height });
  }, [index, steps]);

  useLayoutEffect(() => {
    if (!active) return;
    const step = steps[index];
    const el = step && document.querySelector<HTMLElement>(`[data-tour="${step.target}"]`);
    if (el) el.scrollIntoView({ block: "center", behavior: reduce ? "auto" : "smooth" });
    const t = setTimeout(measure, reduce ? 0 : 320);
    return () => clearTimeout(t);
  }, [active, index, measure, steps, reduce]);

  useEffect(() => {
    if (!active) return;
    const onMove = () => measure();
    window.addEventListener("scroll", onMove, true);
    window.addEventListener("resize", onMove);
    return () => {
      window.removeEventListener("scroll", onMove, true);
      window.removeEventListener("resize", onMove);
    };
  }, [active, measure]);

  if (!mounted || !active) return null;

  const step = steps[index];
  const last = index === steps.length - 1;
  const vh = typeof window !== "undefined" ? window.innerHeight : 800;
  const vw = typeof window !== "undefined" ? window.innerWidth : 400;

  // Tooltip placement: below the target unless it sits low on screen.
  const below = !rect || rect.top + rect.height < vh * 0.55;
  const tipWidth = Math.min(340, vw - 24);
  const tipLeft = rect
    ? Math.min(Math.max(rect.left, 12), vw - tipWidth - 12)
    : (vw - tipWidth) / 2;
  const tipTop = rect
    ? below
      ? rect.top + rect.height + PAD + 10
      : Math.max(12, rect.top - PAD - 176)
    : vh / 2 - 90;

  return createPortal(
    <div className="fixed inset-0 z-[100]" role="dialog" aria-modal="true" aria-label="Guided tour">
      {/* Dimmed backdrop with a spotlight hole (via big box-shadow). Click = next. */}
      <button
        aria-label="Next step"
        onClick={() => (last ? finish() : setIndex((i) => i + 1))}
        className="absolute inset-0 h-full w-full cursor-default"
      />
      {rect && (
        <motion.div
          initial={reduce ? false : { opacity: 0 }}
          animate={{
            opacity: 1,
            top: rect.top - PAD,
            left: rect.left - PAD,
            width: rect.width + PAD * 2,
            height: rect.height + PAD * 2,
          }}
          transition={reduce ? { duration: 0 } : { type: "spring", stiffness: 300, damping: 32 }}
          className="pointer-events-none absolute rounded-2xl ring-2 ring-la-gold"
          style={{ boxShadow: "0 0 0 9999px rgba(3, 7, 15, 0.82)" }}
        />
      )}

      {/* Tooltip */}
      <AnimatePresence mode="wait">
        <motion.div
          key={index}
          initial={reduce ? false : { opacity: 0, y: below ? -6 : 6 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0 }}
          className="absolute rounded-2xl border border-hairline bg-elevated p-4 shadow-2xl"
          style={{ left: tipLeft, top: tipTop, width: tipWidth }}
        >
          <button
            onClick={finish}
            aria-label="Skip tour"
            className="absolute right-3 top-3 text-ink-faint transition hover:text-ink"
          >
            <X size={16} />
          </button>
          <div className="mb-1 flex items-center gap-1.5">
            <span className="h-1.5 w-1.5 rounded-full bg-la-gold" />
            <span className="text-[10px] font-bold uppercase tracking-wide text-la-gold">
              {index + 1} / {steps.length}
            </span>
          </div>
          <h3 className="pr-6 text-sm font-bold">{step.title}</h3>
          <p className="mt-1 text-[13px] leading-snug text-ink-dim">{step.body}</p>

          <div className="mt-3 flex items-center justify-between">
            <button
              onClick={finish}
              className="text-[11px] font-semibold text-ink-faint transition hover:text-ink-dim"
            >
              Skip
            </button>
            <div className="flex items-center gap-2">
              {index > 0 && (
                <button
                  onClick={() => setIndex((i) => i - 1)}
                  className="rounded-lg border border-hairline px-3 py-1.5 text-xs font-semibold text-ink-dim"
                >
                  Back
                </button>
              )}
              <button
                onClick={() => (last ? finish() : setIndex((i) => i + 1))}
                className="rounded-lg la-gradient px-3.5 py-1.5 text-xs font-bold text-navy-950"
              >
                {last ? "Got it" : "Next"}
              </button>
            </div>
          </div>
        </motion.div>
      </AnimatePresence>
    </div>,
    document.body,
  );
}

/** Fire this to (re)start a tour by id. */
export function startTour(id: string) {
  window.dispatchEvent(new CustomEvent("start-tour", { detail: { id } }));
}
