type Mode = 'light' | 'dark';

class Theme {
	current = $state<Mode>('light');

	init() {
		this.current = document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light';
	}

	toggle() {
		this.current = this.current === 'dark' ? 'light' : 'dark';
		document.documentElement.dataset.theme = this.current;
		try {
			localStorage.setItem('theme', this.current);
		} catch {
			/* stockage indisponible */
		}
	}
}

export const theme = new Theme();
