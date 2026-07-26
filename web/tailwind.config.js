/** Tokens carried over from the original app's aesthetics spec.
 *  Terminal genre: dense, instrumented, amber when live, cool when settled. */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas: { inset: "#08080C", DEFAULT: "#0A0A0F", raised: "#111118", overlay: "#1A1A24" },
        edge:   { subtle: "#1E1E2A", DEFAULT: "#2A2A3A", strong: "#3A3A4E" },
        ink:    { DEFAULT: "#EEEEF0", secondary: "#9394A1", tertiary: "#55566A", inverse: "#0A0A0F" },
        amber:  { DEFAULT: "#FBBF24", dim: "#D97706", bright: "#FCD34D" },
        mint:   { DEFAULT: "#34D399", dim: "#059669", bright: "#6EE7B7" },
        rose:   { DEFAULT: "#FB7185", dim: "#E11D48" },
        indigo: { DEFAULT: "#818CF8", dim: "#6366F1" },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ['"JetBrains Mono"', "ui-monospace", "monospace"],
      },
      fontSize: {
        "2xs": ["11px", { lineHeight: "16px" }],
        xs:    ["12px", { lineHeight: "18px" }],
        sm:    ["13px", { lineHeight: "20px" }],
        base:  ["14px", { lineHeight: "22px" }],
        lg:    ["16px", { lineHeight: "24px" }],
        xl:    ["20px", { lineHeight: "28px" }],
        "2xl": ["28px", { lineHeight: "36px" }],
      },
      letterSpacing: { label: "0.15em" },
      keyframes: {
        "pulse-live": { "0%,100%": { opacity: "1" }, "50%": { opacity: "0.4" } },
        "tape-arrival": {
          "0%": { backgroundColor: "rgba(217,119,6,0.18)" },
          "100%": { backgroundColor: "transparent" },
        },
      },
      animation: {
        "pulse-live": "pulse-live 1.6s cubic-bezier(0.4,0,0.6,1) infinite",
        "tape-arrival": "tape-arrival 2000ms cubic-bezier(0.16,1,0.3,1) forwards",
      },
    },
  },
  plugins: [],
};
