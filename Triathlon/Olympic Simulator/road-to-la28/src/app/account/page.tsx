"use client";
import { Suspense, useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { ChevronLeft, Cloud, CloudOff, Copy, Check, Bell, BellOff, Send, Link2 } from "lucide-react";
import {
  fetchAccount,
  claimBoard,
  getRecoveryLink,
  pullBoardToLocal,
  type Board,
} from "@/lib/local/account-client";
import {
  getPushState,
  enablePush,
  disablePush,
  sendTestPush,
  refreshPushFollows,
  type PushState,
} from "@/lib/local/push-client";
import { cn } from "@/lib/utils";

export default function AccountPage() {
  return (
    <main className="mx-auto px-4 pt-6 lg:max-w-2xl">
      <div className="mb-4 flex items-center gap-2">
        <Link
          href="/"
          className="flex h-9 w-9 items-center justify-center rounded-full border border-hairline bg-surface text-ink-dim"
        >
          <ChevronLeft size={18} />
        </Link>
        <h1 className="text-lg font-extrabold leading-tight">Your Board</h1>
      </div>
      <Suspense fallback={<div className="text-sm text-ink-faint">Loading…</div>}>
        <AccountBody />
      </Suspense>
    </main>
  );
}

function AccountBody() {
  const params = useSearchParams();
  const justLinked = params.get("linked") === "1";

  const [claimed, setClaimed] = useState<boolean | null>(null);
  const [busy, setBusy] = useState(false);
  const [recovery, setRecovery] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [pushState, setPushState] = useState<PushState>("default");
  const [testSent, setTestSent] = useState(false);

  const refresh = useCallback(async () => {
    const acc = await fetchAccount();
    setClaimed(acc.claimed);
    if (acc.claimed && justLinked && acc.board) await pullBoardToLocal(acc.board as Board);
    setPushState(await getPushState());
  }, [justLinked]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  async function onClaim() {
    setBusy(true);
    await claimBoard();
    await refresh();
    setBusy(false);
  }

  async function onCopyRecovery() {
    let url = recovery;
    if (!url) {
      url = await getRecoveryLink();
      setRecovery(url);
    }
    if (url) {
      await navigator.clipboard.writeText(url).catch(() => {});
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    }
  }

  async function onTogglePush() {
    setBusy(true);
    if (pushState === "on") await disablePush();
    else await enablePush();
    await refreshPushFollows();
    setPushState(await getPushState());
    setBusy(false);
  }

  async function onTest() {
    const ok = await sendTestPush();
    if (ok) {
      setTestSent(true);
      setTimeout(() => setTestSent(false), 2000);
    }
  }

  return (
    <div className="space-y-4">
      {justLinked && (
        <div className="rounded-xl bg-good/15 px-3 py-2 text-sm font-semibold text-good">
          This device is now linked to your board.
        </div>
      )}

      {/* Sync */}
      <section className="card p-4">
        <div className="mb-1 flex items-center gap-2">
          {claimed ? <Cloud size={18} className="text-good" /> : <CloudOff size={18} className="text-ink-faint" />}
          <h2 className="text-sm font-bold">Cross-device sync</h2>
        </div>
        {claimed ? (
          <>
            <p className="text-[13px] text-ink-dim">
              Your athlete and follows are backed up and synced. Open your recovery link on another
              device to see the same board there.
            </p>
            <button
              onClick={onCopyRecovery}
              className="mt-3 flex w-full items-center justify-center gap-2 rounded-xl border border-hairline bg-surface py-2.5 text-sm font-semibold"
            >
              {copied ? <Check size={15} className="text-good" /> : <Link2 size={15} />}
              {copied ? "Copied recovery link" : "Copy recovery link"}
            </button>
            <p className="mt-2 text-[11px] text-ink-faint">
              Keep this link private — anyone with it can load your board.
            </p>
          </>
        ) : (
          <>
            <p className="text-[13px] text-ink-dim">
              Right now your board lives only on this device. Claim it to back it up and sync across
              your phone, tablet, and laptop — no email, no password.
            </p>
            <button
              onClick={onClaim}
              disabled={busy}
              className="mt-3 w-full rounded-xl la-gradient py-2.5 text-sm font-bold text-navy-950 disabled:opacity-60"
            >
              {busy ? "Claiming…" : "Claim your board"}
            </button>
          </>
        )}
      </section>

      {/* Notifications */}
      <section className="card p-4">
        <div className="mb-1 flex items-center gap-2">
          {pushState === "on" ? <Bell size={18} className="text-good" /> : <BellOff size={18} className="text-ink-faint" />}
          <h2 className="text-sm font-bold">Race-day alerts</h2>
        </div>
        <p className="text-[13px] text-ink-dim">
          We&apos;ll only ping you when something <em>actually</em> happens to an athlete you follow —
          a new ranking, a rival banking points, crossing the line. Never for engagement.
        </p>

        {pushState === "unsupported" && (
          <p className="mt-3 text-[12px] text-ink-faint">
            This browser doesn&apos;t support notifications. On iPhone, add the app to your Home
            Screen first, then enable alerts.
          </p>
        )}
        {pushState === "denied" && (
          <p className="mt-3 text-[12px] text-bad">
            Notifications are blocked in your browser settings. Re-enable them there to turn on alerts.
          </p>
        )}
        {(pushState === "default" || pushState === "granted-off" || pushState === "on") && (
          <button
            onClick={onTogglePush}
            disabled={busy}
            className={cn(
              "mt-3 w-full rounded-xl py-2.5 text-sm font-bold disabled:opacity-60",
              pushState === "on" ? "border border-hairline bg-surface text-ink-dim" : "la-gradient text-navy-950",
            )}
          >
            {busy ? "…" : pushState === "on" ? "Turn off alerts" : "Turn on alerts"}
          </button>
        )}
        {pushState === "on" && (
          <button
            onClick={onTest}
            className="mt-2 flex w-full items-center justify-center gap-2 rounded-xl border border-hairline py-2 text-[13px] font-semibold text-ink-dim"
          >
            {testSent ? <Check size={14} className="text-good" /> : <Send size={14} />}
            {testSent ? "Sent — check your notifications" : "Send a test notification"}
          </button>
        )}
      </section>

      <p className="px-2 text-center text-[11px] text-ink-faint">
        No accounts, no email, no tracking. Your board is a private list of athletes.
      </p>
    </div>
  );
}
