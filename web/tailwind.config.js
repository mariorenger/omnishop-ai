/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#080a10",
        panel: "#0c0f18",
        card: "#141826",
        card2: "#1a1f30",
        line: "#2a3145",
        line2: "#3c4763",
        fg: "#f2f4fc",
        muted: "#a2adc6",
        accent: "#818cf8",
        accent2: "#c084fc",
        accent3: "#67e8f9",
        ok: "#34d399",
        warn: "#fbbf24",
        bad: "#fb7185",
      },
      fontFamily: {
        sans: ["Plus Jakarta Sans", "ui-sans-serif", "system-ui", "Segoe UI", "Roboto", "sans-serif"],
      },
      backgroundImage: {
        "pastel": "linear-gradient(120deg, #818cf8 0%, #a78bfa 42%, #67e8f9 100%)",
        "pastel-soft": "linear-gradient(120deg, rgba(129,140,248,0.16), rgba(167,139,250,0.12) 50%, rgba(103,232,249,0.10))",
      },
      boxShadow: {
        soft: "0 1px 0 0 rgba(255,255,255,0.04) inset, 0 10px 34px rgba(0,0,0,0.40)",
        glow: "0 0 0 1px rgba(129,140,248,0.28), 0 10px 46px rgba(129,140,248,0.20)",
      },
      borderRadius: { xl: "14px", "2xl": "18px" },
    },
  },
  plugins: [],
};
