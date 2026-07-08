"use client";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";

const COLORS = ["#ff5c6c", "#8b5cf6", "#f5c518", "#35c8ff", "#21d07a"];

/**
 * The line-cross moment. Fires a short gradient burst + confetti when an athlete
 * crosses into the qualifying zone. Respects prefers-reduced-motion (static glow).
 */
export function Celebration({ show }: { show: boolean }) {
  const reduce = useReducedMotion();

  return (
    <AnimatePresence>
      {show && (
        <motion.div
          key="celebrate"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="pointer-events-none absolute inset-0 z-30 flex items-center justify-center overflow-hidden"
        >
          {/* radial glow */}
          <motion.div
            initial={{ scale: 0.6, opacity: 0.9 }}
            animate={{ scale: 1.6, opacity: 0 }}
            transition={{ duration: 1.1, ease: "easeOut" }}
            className="absolute h-40 w-40 rounded-full"
            style={{ background: "radial-gradient(circle, #f5c51888, transparent 70%)" }}
          />
          {!reduce &&
            Array.from({ length: 16 }).map((_, i) => {
              const angle = (i / 16) * Math.PI * 2;
              const dist = 120 + (i % 3) * 40;
              return (
                <motion.span
                  key={i}
                  initial={{ x: 0, y: 0, opacity: 1, scale: 1 }}
                  animate={{
                    x: Math.cos(angle) * dist,
                    y: Math.sin(angle) * dist + 40,
                    opacity: 0,
                    scale: 0.4,
                  }}
                  transition={{ duration: 1.2, ease: "easeOut" }}
                  className="absolute h-2.5 w-2.5 rounded-sm"
                  style={{ background: COLORS[i % COLORS.length] }}
                />
              );
            })}
          <motion.div
            initial={{ scale: 0.7, y: 8, opacity: 0 }}
            animate={{ scale: 1, y: 0, opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ type: "spring", stiffness: 400, damping: 18 }}
            className="rounded-2xl bg-elevated/70 px-5 py-2.5 text-lg font-black backdrop-blur-sm la-gradient-text"
          >
            QUALIFYING ZONE
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
