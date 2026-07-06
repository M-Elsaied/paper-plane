"use client";
import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { X, MousePointerClick, SlidersHorizontal, Trophy } from "lucide-react";

const KEY = "rtla28:onboarded";

/**
 * 60-second welcome — a one-time, dismissible explainer that frames the app's
 * three-beat loop. Shows only on first visit; never nags.
 */
export function OnboardingHint() {
  const [show, setShow] = useState(false);

  useEffect(() => {
    try {
      if (!localStorage.getItem(KEY)) setShow(true);
    } catch {
      /* no-op */
    }
  }, []);

  function dismiss() {
    try {
      localStorage.setItem(KEY, "1");
    } catch {
      /* no-op */
    }
    setShow(false);
  }

  return (
    <AnimatePresence>
      {show && (
        <motion.div
          initial={{ opacity: 0, y: -8, height: 0 }}
          animate={{ opacity: 1, y: 0, height: "auto" }}
          exit={{ opacity: 0, height: 0 }}
          className="mb-5"
        >
          <div className="card relative p-4">
            <button
              onClick={dismiss}
              aria-label="Dismiss"
              className="absolute right-3 top-3 text-ink-faint transition hover:text-ink"
            >
              <X size={16} />
            </button>
            <h2 className="mb-3 pr-6 text-sm font-bold">
              Watch a real athlete chase the Olympics
            </h2>
            <ol className="space-y-2.5">
              <Step icon={<MousePointerClick size={15} />} n={1} text="Pick your athlete below." />
              <Step icon={<Trophy size={15} />} n={2} text="See their live rank and gap to the qualification line." />
              <Step icon={<SlidersHorizontal size={15} />} n={3} text="Tap Simulate to play out a race and watch the ranking move." />
            </ol>
            <button
              onClick={dismiss}
              className="mt-4 w-full rounded-xl la-gradient py-2.5 text-sm font-bold text-navy-950"
            >
              Let&apos;s go
            </button>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

function Step({ icon, n, text }: { icon: React.ReactNode; n: number; text: string }) {
  return (
    <li className="flex items-center gap-3">
      <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-white/8 text-electric-bright">
        {icon}
      </span>
      <span className="text-[13px] text-ink-dim">
        <span className="font-bold text-ink">{n}.</span> {text}
      </span>
    </li>
  );
}
