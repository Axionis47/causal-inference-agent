/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          900: '#1a1a2e',
          700: '#334155',
          500: '#64748b',
          300: '#94a3b8',
          200: '#cbd5e1',
          100: '#e2e8f0',
          50: '#f8fafc',
        },
        accent: {
          DEFAULT: '#1e40af',
          hover: '#1e3a8a',
          light: '#dbeafe',
        },
        sig: {
          yes: '#15803d',
          maybe: '#b45309',
          no: '#b91c1c',
        },
      },
      fontFamily: {
        serif: ['"Source Serif 4"', 'Georgia', 'serif'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'monospace'],
      },
    },
  },
  plugins: [],
}
