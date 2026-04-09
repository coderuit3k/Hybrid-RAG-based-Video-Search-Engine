/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    darkMode: 'class',
    theme: {
        extend: {
            colors: {
                primary: {
                    DEFAULT: '#06b6d4', // cyan
                    dark: '#0891b2',
                },
                secondary: {
                    DEFAULT: '#a855f7', // purple
                    dark: '#9333ea',
                },
                accent: {
                    DEFAULT: '#ec4899', // pink
                    dark: '#db2777',
                },
                dark: {
                    bg: '#0f172a',
                    surface: '#1e293b',
                    border: '#334155',
                }
            },
            animation: {
                'glow': 'glow 2s ease-in-out infinite alternate',
                'float': 'float 3s ease-in-out infinite',
            },
            keyframes: {
                glow: {
                    '0%': { boxShadow: '0 0 5px rgba(6, 182, 212, 0.5), 0 0 10px rgba(6, 182, 212, 0.3)' },
                    '100%': { boxShadow: '0 0 10px rgba(6, 182, 212, 0.8), 0 0 20px rgba(6, 182, 212, 0.5)' },
                },
                float: {
                    '0%, 100%': { transform: 'translateY(0px)' },
                    '50%': { transform: 'translateY(-10px)' },
                }
            }
        },
    },
    plugins: [],
}