import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

const API_TARGET = process.env.API_TARGET ?? 'http://localhost:8000';

export default defineConfig({
	plugins: [sveltekit()],
	server: {
		proxy: {
			'/api': API_TARGET
		}
	}
});
