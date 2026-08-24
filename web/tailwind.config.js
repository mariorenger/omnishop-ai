/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#0a0c11",
        panel: "#0f121a",
        card: "#141824",
        card2: "#181d2b",
        line: "#232a3b",
        line2: "#2d364c",
        fg: "#e8ecf6",
        muted: "#8b95ad",
        accent: "#6d7cff",
        accent2: "#8b5cf6",
        ok: "#34d399",
        warn: "#fbbf24",
        bad: "#f87171",
      },
      fontFamily: {
        sans: ["Be Vietnam Pro", "Inter", "ui-sans-serif", "system-ui", "Segoe UI", "Roboto", "sans-serif"],
      },
      boxShadow: {
        soft: "0 1px 0 0 rgba(255,255,255,0.03) inset, 0 8px 30px rgba(0,0,0,0.35)",
        glow: "0 0 0 1px rgba(109,124,255,0.25), 0 8px 40px rgba(109,124,255,0.15)",
      },
      borderRadius: { xl: "14px", "2xl": "18px" },
    },
  },
  plugins: [],
};
