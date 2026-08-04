export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        display: ['Hanken Grotesk', 'system-ui', 'sans-serif'],
        body: ['IBM Plex Sans', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      colors: {
        // Semantic surface colors for light/dark theming
        surface: {
          DEFAULT: 'var(--surface)',
          inverse: 'var(--surface-inverse)',
        },
        elevated: {
          DEFAULT: 'var(--elevated)',
        },
        muted: {
          DEFAULT: 'var(--muted)',
        },
        border: 'var(--border)',
        'intent-primary': 'var(--intent-primary)',
        'intent-primary-contrast': 'var(--intent-primary-contrast)',
      },
      animation: {
        'fade-in': 'fadeIn 200ms ease-out',
        'slide-in': 'slideIn 200ms ease-out',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0', transform: 'translateY(4px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        slideIn: {
          '0%': { opacity: '0', transform: 'translateX(-8px)' },
          '100%': { opacity: '1', transform: 'translateX(0)' },
        },
      },
    },
  },
  plugins: [],
};
