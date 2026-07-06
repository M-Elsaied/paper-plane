import { config } from "dotenv";
import { defineConfig } from "drizzle-kit";

// Neon integration writes DATABASE_URL to .env.local; fall back to .env.
config({ path: ".env.local" });
config();

export default defineConfig({
  schema: "./src/db/schema.ts",
  out: "./drizzle",
  dialect: "postgresql",
  dbCredentials: { url: process.env.DATABASE_URL ?? "" },
});
