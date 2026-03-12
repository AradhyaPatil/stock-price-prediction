/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        'bg-primary':   '#071521',
        'bg-secondary': '#0b1c2c',
        'bg-panel':     '#102539',
        'bg-card':      '#132a3f',
        'accent-blue':  '#6fa8dc',
        'accent-cyan':  '#62c6c2',
        'accent-green': '#4fbf8f',
        'accent-red':   '#d96a5d',
        'accent-gold':  '#caa76a',
        'text-primary': '#edf3f8',
        'text-muted':   '#8ea2b5',
        'text-soft':    '#b8c7d4',
        'line-subtle':  'rgba(222, 234, 245, 0.08)',
      },
      fontFamily: {
        sans: ['IBM Plex Sans', 'sans-serif'],
        display: ['Source Serif 4', 'serif'],
      },
      boxShadow: {
        panel: '0 14px 34px rgba(1, 11, 20, 0.35)',
      },
    },
  },
  plugins: [],
}
