/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#090b10",
        panel: "#0e1119",
        card: "#151a26",
        card2: "#1b2130",
        line: "#2b3346",
        line2: "#3a4661",
        fg: "#eff2f9",
        muted: "#9aa6c0",
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
