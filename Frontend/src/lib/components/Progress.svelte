<script lang="ts">
	let { value, text }: { value: number; text: string } = $props();
</script>

<div class="progress">
	<div class="track"><div class="fill" style="width: {value * 100}%"></div></div>
	<div class="caption">{text}</div>
</div>

<style>
	.progress {
		margin-bottom: 1rem;
	}
	.track {
		height: 0.5rem;
		background: rgba(151, 166, 195, 0.25);
		border-radius: 0.25rem;
		overflow: hidden;
	}
	.fill {
		height: 100%;
		background: var(--primary);
		transition: width 0.2s;
	}
</style>
