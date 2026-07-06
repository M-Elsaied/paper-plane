"use client";
import { useEffect } from "react";
import { animate, useMotionValue, useTransform, motion } from "motion/react";

/** Springy count-up number. Re-animates whenever `value` changes. */
export function CountUp({
  value,
  className,
  decimals = 0,
  prefix = "",
  suffix = "",
}: {
  value: number;
  className?: string;
  decimals?: number;
  prefix?: string;
  suffix?: string;
}) {
  const mv = useMotionValue(0);
  const rounded = useTransform(mv, (v) =>
    `${prefix}${v.toLocaleString("en-US", {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    })}${suffix}`,
  );

  useEffect(() => {
    const controls = animate(mv, value, {
      duration: 0.9,
      ease: [0.16, 1, 0.3, 1],
    });
    return controls.stop;
  }, [value, mv]);

  return <motion.span className={className}>{rounded}</motion.span>;
}
