import { describe, it, expect } from "vitest";
import { GET as accountGET, POST as accountPOST, PUT as accountPUT } from "@/app/api/account/route";
import { GET as recoveryGET } from "@/app/api/account/recovery/route";
import { GET as claimGET } from "@/app/claim/route";
import { verifyToken, signToken } from "@/lib/auth";
import { getDb } from "@/db/client";
import { accounts } from "@/db/schema";
import { cookieJar, resetCookies, setTestDb } from "../helpers/setup";

const board = { myAthlete: { athleteId: 86042, fullName: "Vasco Vilaca", noc: "POR", gender: "male" }, follows: [] };
const claimReq = (b: unknown) =>
  accountPOST(new Request("http://test/api/account", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(b) }));

describe("account bridge API", () => {
  it("claims a board, sets a session, and reads it back", async () => {
    const claimed = await claimReq({ board });
    expect((await claimed.json()).claimed).toBe(true);
    expect(cookieJar().store.has("rtla28_session")).toBe(true);

    const status = await (await accountGET()).json();
    expect(status.claimed).toBe(true);
    expect(status.board.myAthlete.athleteId).toBe(86042);

    expect((await getDb()!.select().from(accounts)).length).toBe(1);
  });

  it("does not create a second account on double-claim", async () => {
    await claimReq({ board });
    await claimReq({ board });
    expect((await getDb()!.select().from(accounts)).length).toBe(1);
  });

  it("PUT syncs the board only with a session", async () => {
    const noSession = await accountPUT(new Request("http://test/api/account", { method: "PUT", body: JSON.stringify({ board }) }));
    expect(noSession.status).toBe(401);

    await claimReq({ board });
    const noBody = await accountPUT(new Request("http://test/api/account", { method: "PUT", body: JSON.stringify({}) }));
    expect(noBody.status).toBe(400);

    const next = { myAthlete: null, follows: [{ athleteId: 1, fullName: "X", noc: "GBR", gender: "male" }] };
    const ok = await accountPUT(new Request("http://test/api/account", { method: "PUT", body: JSON.stringify({ board: next }) }));
    expect(ok.status).toBe(200);
    const status = await (await accountGET()).json();
    expect(status.board.follows.length).toBe(1);
  });

  it("recovery link round-trips onto a fresh device", async () => {
    await claimReq({ board });
    const [row] = await getDb()!.select().from(accounts);
    const rec = await (await recoveryGET(new Request("http://test/api/account/recovery"))).json();
    expect(rec.url).toContain("/claim?t=");
    const token = new URL(rec.url).searchParams.get("t")!;
    expect(verifyToken(token)).toBe(row.id);

    // simulate a fresh device: drop cookies, redeem the link
    resetCookies();
    const redeem = await claimGET(new Request(`http://test/claim?t=${encodeURIComponent(token)}`));
    expect(redeem.status).toBe(307);
    expect(redeem.headers.get("location")).toContain("/account?linked=1");
    // the fresh device now sees the same board
    expect((await (await accountGET()).json()).board.myAthlete.athleteId).toBe(86042);
  });

  it("rejects a forged recovery token", async () => {
    const redeem = await claimGET(new Request("http://test/claim?t=not-a-real-token"));
    expect(redeem.headers.get("location")).toContain("error=invalid");
  });

  it("signs a token for a nonexistent account that verifies but has no board", async () => {
    const token = signToken("ghost-account");
    const redeem = await claimGET(new Request(`http://test/claim?t=${encodeURIComponent(token)}`));
    expect(redeem.headers.get("location")).toContain("error=invalid");
  });

  it("returns 503 without a database", async () => {
    setTestDb(null);
    expect((await claimReq({ board })).status).toBe(503);
  });
});
