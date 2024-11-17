import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class", // Enable dark mode support
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "var(--background)",
        foreground: "var(--foreground)",
        mainBg: "#161616", // For main background
        sidebarBg: "#202020", // For sidebar and component backgrounds
        chatIconColor: "#10A37F",
      },
    },
  },
  plugins: [],
};
export default config;
