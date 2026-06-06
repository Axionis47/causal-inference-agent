/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        // ─── Journal palette (legacy, retires per aesthetics.md §14) ───
        ink: {
          900: '#1a1a2e',
          700: '#334155',
          500: '#64748b',
          300: '#94a3b8',
          200: '#cbd5e1',
          100: '#e2e8f0',
          50: '#f8fafc',
          // Terminal scale — same token namespace per spec §3.3
          DEFAULT: '#EEEEF0',
          secondary: '#9394A1',
          tertiary: '#55566A',
          inverse: '#0A0A0F',
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

        // ─── Terminal palette (target per aesthetics.md §3) ───
        canvas: {
          inset: '#08080C',
          DEFAULT: '#0A0A0F',
          raised: '#111118',
          overlay: '#1A1A24',
        },
        edge: {
          subtle: '#1E1E2A',
          DEFAULT: '#2A2A3A',
          strong: '#3A3A4E',
        },
        amber: {
          DEFAULT: '#FBBF24',
          dim: '#D97706',
          bright: '#FCD34D',
        },
        mint: {
          DEFAULT: '#34D399',
          dim: '#059669',
          bright: '#6EE7B7',
        },
        rose: {
          DEFAULT: '#FB7185',
          dim: '#E11D48',
        },
        indigo: {
          DEFAULT: '#818CF8',
          dim: '#6366F1',
        },
        bone: {
          DEFAULT: '#D4C7A8',
        },
      },
      fontFamily: {
        serif: ['"Source Serif 4"', 'Georgia', 'serif'],
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', '"IBM Plex Mono"', 'ui-monospace', 'monospace'],
      },
      fontSize: {
        '2xs': ['11px', { lineHeight: '16px' }],
        xs: ['12px', { lineHeight: '18px' }],
        sm: ['13px', { lineHeight: '20px' }],
        base: ['14px', { lineHeight: '22px' }],
        lg: ['16px', { lineHeight: '24px' }],
        xl: ['20px', { lineHeight: '28px' }],
        '2xl': ['28px', { lineHeight: '36px' }],
      },
      letterSpacing: {
        'terminal-label': '0.15em',
      },
      keyframes: {
        'pulse-live': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.4' },
        },
        'tape-arrival': {
          '0%': { backgroundColor: 'rgba(217, 119, 6, 0.18)' },
          '100%': { backgroundColor: 'transparent' },
        },
      },
      animation: {
        'pulse-live': 'pulse-live 1.6s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'tape-arrival': 'tape-arrival 2000ms cubic-bezier(0.16, 1, 0.3, 1) forwards',
      },
    },
  },
  plugins: [],
}
