"use client";
/** A stable per-device id so anonymous users can make Pick-'Em predictions
 *  before claiming an account. */
const KEY = "rtla28:device";

export function getDeviceId(): string {
  try {
    let id = localStorage.getItem(KEY);
    if (!id) {
      id = crypto.randomUUID();
      localStorage.setItem(KEY, id);
    }
    return id;
  } catch {
    return "anon-device";
  }
}
