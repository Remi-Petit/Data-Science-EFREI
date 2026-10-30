<script lang="ts">
	let { value = 0, text, indeterminate = false }: { value?: number; text: string; indeterminate?: boolean } = $props();
</script>

<div class="progress">
	<div class="track">
		<div class="fill" class:indeterminate style={indeterminate ? '' : `width: ${value * 100}%`}></div>
	</div>
	<div class="caption">{text}</div>
</div>

<style>
	.track {
		height: 0.5rem;
		margin-bottom: 0.4rem;
		background: var(--bg-secondary);
		border-radius: 999px;
		overflow: hidden;
	}
	.fill {
		height: 100%;
		background: var(--gradient);
		border-radius: 999px;
		transition: width 0.25s;
	}
	.fill.indeterminate {
		width: 35%;
		animation: slide 1.1s ease-in-out infinite;
	}
	@keyframes slide {
		from {
			transform: translateX(-100%);
		}
		to {
			transform: translateX(300%);
		}
	}
</style>
