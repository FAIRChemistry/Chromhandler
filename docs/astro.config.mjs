// @ts-check
import { defineConfig } from 'astro/config';
import react from '@astrojs/react';
import starlight from '@astrojs/starlight';

// https://astro.build/config
export default defineConfig({
	integrations: [
		react(),
		starlight({
			title: 'ChromHandler',
			social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/FAIRChemistry/Chromhandler' }],
			sidebar: [
				{
					label: '📚 Guides',
					items: [
						{ label: '📖 Reading Chromatograms', slug: 'reading-chromatograms' },
						{ label: '🗂️ Data Organization', slug: 'reading-chromatograms/data-organization' },
						{ label: '🧪 Supported File Types', slug: 'reading-chromatograms/supported-file-types' },
						{ label: '⚡ How To Read Chromatograms', slug: 'reading-chromatograms/how-to-read-chromatograms' },
						{ label: '🧬 Define Molecules', slug: 'reading-chromatograms/define-molecules' },
						{ label: '🧫 Load Initial Conditions', slug: 'reading-chromatograms/load-initial-conditions' },
						{ label: '📊 Interactive Plot Demo', slug: 'plotly-demo' },
					],
				},
				{
					label: '📘 Reference',
					autogenerate: { directory: 'reference' },
				},
			],
		}),
	],
});
