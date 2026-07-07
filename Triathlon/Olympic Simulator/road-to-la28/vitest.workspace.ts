import { defineWorkspace } from "vitest/config";

export default defineWorkspace([
  {
    extends: "./vitest.config.ts",
    test: {
      name: "unit",
      include: [
        "tests/engine/**/*.test.ts",
        "tests/unit/**/*.test.ts",
        "tests/golden/**/*.test.ts",
      ],
    },
  },
  {
    extends: "./vitest.config.ts",
    test: {
      name: "integration",
      include: ["tests/integration/**/*.test.ts"],
      setupFiles: ["./tests/helpers/setup.ts"],
    },
  },
]);
