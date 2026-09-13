/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#0b0f14",
          900: "#10161e",
          800: "#161d27",
          700: "#1d2632",
          600: "#2a3544",
        },
        paper: "#e8eef4",
        mute: "#8b98a5",
        fire: "#d4480b",
        uav: "#3d8bfd",
        ok: "#2f9e6c",
        warn: "#c9a227",
      },
      fontFamily: {
        sans: ["IBM Plex Sans", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["IBM Plex Mono", "ui-monospace", "SFMono-Regular", "monospace"],
      },
    },
  },
  plugins: [],
};
