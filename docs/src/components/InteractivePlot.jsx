import { useEffect, useState } from 'react';

const linkedPoints = [
	{
		sampleId: 'P0-0.0_min',
		reactionTime: 0.0,
		derivedConcentration: 0.05,
		sourceReplicates: [0.04, 0.05, 0.06, 0.05],
	},
	{
		sampleId: 'P1-2.0_min',
		reactionTime: 2.0,
		derivedConcentration: 0.14,
		sourceReplicates: [0.12, 0.15, 0.13, 0.16],
	},
	{
		sampleId: 'P2-4.0_min',
		reactionTime: 4.0,
		derivedConcentration: 0.29,
		sourceReplicates: [0.27, 0.3, 0.28, 0.31],
	},
	{
		sampleId: 'P3-6.0_min',
		reactionTime: 6.0,
		derivedConcentration: 0.43,
		sourceReplicates: [0.4, 0.45, 0.43, 0.44],
	},
	{
		sampleId: 'P4-8.0_min',
		reactionTime: 8.0,
		derivedConcentration: 0.59,
		sourceReplicates: [0.56, 0.61, 0.58, 0.61],
	},
	{
		sampleId: 'P5-10.0_min',
		reactionTime: 10.0,
		derivedConcentration: 0.71,
		sourceReplicates: [0.68, 0.73, 0.7, 0.73],
	},
];

export default function InteractivePlot() {
	const [Plot, setPlot] = useState(null);
	const [activeIndex, setActiveIndex] = useState(2);

	useEffect(() => {
		let isMounted = true;

		import('react-plotly.js').then((module) => {
			if (isMounted) {
				setPlot(() => module.default);
			}
		});

		return () => {
			isMounted = false;
		};
	}, []);

	if (!Plot) {
		return <p>Loading interactive plot...</p>;
	}

	const selectedPoint = linkedPoints[activeIndex];
	const topX = linkedPoints.map((point) => point.reactionTime);
	const topY = linkedPoints.map((point) => point.derivedConcentration);
	const topMarkerColors = linkedPoints.map((_, index) =>
		index === activeIndex ? '#c25b12' : '#2f7d32'
	);
	const topMarkerSizes = linkedPoints.map((_, index) => (index === activeIndex ? 14 : 10));
	const bottomX = selectedPoint.sourceReplicates.map((_, index) => index + 1);
	const bottomMeanLine = selectedPoint.sourceReplicates.map(
		() => selectedPoint.derivedConcentration
	);

	return (
		<div style={{ width: '100%' }}>
			<p style={{ marginBottom: '1rem' }}>
				Hover a point in Plot A to update Plot B and inspect the source measurements
				that produced the derived value.
			</p>
			<Plot
				data={[
					{
						x: topX,
						y: topY,
						type: 'scatter',
						mode: 'lines+markers',
						name: 'derived concentration',
						line: { color: '#6b8f71', width: 3 },
						marker: { size: topMarkerSizes, color: topMarkerColors },
						customdata: linkedPoints.map((point) => [
							point.sampleId,
							point.sourceReplicates.length,
						]),
						hovertemplate:
							'<b>%{customdata[0]}</b><br>reaction time: %{x} min<br>derived concentration: %{y:.2f} mM<br>source measurements: %{customdata[1]}<extra></extra>',
					},
				]}
				layout={{
					title: 'Plot A: derived concentrations',
					xaxis: { title: 'Reaction time (min)' },
					yaxis: { title: 'Concentration (mM)' },
					hovermode: 'closest',
					margin: { l: 60, r: 24, t: 70, b: 60 },
					paper_bgcolor: 'white',
					plot_bgcolor: 'white',
					autosize: true,
				}}
				config={{
					displayModeBar: true,
					responsive: true,
				}}
				style={{ width: '100%', height: '420px' }}
				useResizeHandler
				onHover={(event) => {
					const nextIndex = event?.points?.[0]?.pointIndex;
					if (typeof nextIndex === 'number') {
						setActiveIndex(nextIndex);
					}
				}}
			/>
			<div
				style={{
					margin: '0.75rem 0 1rem',
					padding: '0.85rem 1rem',
					border: '1px solid #d9dde3',
					borderRadius: '0.75rem',
					background: '#f7f8fa',
				}}
			>
				<strong>Selected derived point:</strong> {selectedPoint.sampleId} at{' '}
				{selectedPoint.reactionTime} min. Plot B shows the source replicate values whose
				mean becomes the point in Plot A.
			</div>
			<Plot
				data={[
					{
						x: bottomX,
						y: selectedPoint.sourceReplicates,
						type: 'scatter',
						mode: 'markers',
						name: 'source replicates',
						marker: { size: 12, color: '#1d4ed8' },
						hovertemplate:
							'replicate %{x}<br>estimated concentration: %{y:.2f} mM<extra></extra>',
					},
					{
						x: bottomX,
						y: bottomMeanLine,
						type: 'scatter',
						mode: 'lines',
						name: 'mean used in Plot A',
						line: { color: '#c25b12', width: 2, dash: 'dash' },
						hovertemplate:
							'mean used in Plot A: %{y:.2f} mM<extra></extra>',
					},
				]}
				layout={{
					title: `Plot B: source measurements for ${selectedPoint.sampleId}`,
					xaxis: { title: 'Replicate injection' },
					yaxis: { title: 'Estimated concentration (mM)' },
					hovermode: 'closest',
					legend: { orientation: 'h', y: 1.12 },
					margin: { l: 60, r: 24, t: 70, b: 60 },
					paper_bgcolor: 'white',
					plot_bgcolor: 'white',
					autosize: true,
				}}
				config={{
					displayModeBar: true,
					responsive: true,
				}}
				style={{ width: '100%', height: '380px' }}
				useResizeHandler
			/>
		</div>
	);
}
