import Link from "next/link";

export default function NotFound() {
  return (
    <main className="flex min-h-[70dvh] flex-col items-center justify-center px-4 text-center">
      <div className="tnum text-6xl font-black la-gradient-text">404</div>
      <p className="mt-2 text-ink-dim">That athlete or page isn&apos;t on the road to LA28.</p>
      <Link
        href="/"
        className="mt-6 rounded-xl la-gradient px-5 py-2.5 font-bold text-navy-950"
      >
        Back to the cockpit
      </Link>
    </main>
  );
}
