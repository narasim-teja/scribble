import { defineConfig } from "vite";

// Relative asset paths, so the built page works from any folder.
export default defineConfig({ base: "./", build: { target: "es2022" } });
