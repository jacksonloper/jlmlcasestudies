// Drop-in replacement for `import Plot from 'react-plotly.js'`.
//
// Plotly is several MB, and parsing it was most of the site's build time. Here it is not
// bundled at all: Vite copies the prebuilt plotly.min.js into the build as a plain file, and
// the browser loads it with a <script> tag the first time a plot is shown. That also keeps
// it out of the main bundle, so pages without plots no longer download it.
import { useEffect, useState } from 'react';
import createPlotlyComponent from 'react-plotly.js/factory';
import plotlyUrl from 'plotly.js-dist-min/plotly.min.js?url';

let plotComponent;

function loadPlot() {
  plotComponent ??= new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = plotlyUrl;
    script.onload = () => resolve(createPlotlyComponent(window.Plotly));
    script.onerror = () => {
      plotComponent = undefined;
      reject(new Error('Could not load Plotly'));
    };
    document.head.appendChild(script);
  });
  return plotComponent;
}

export default function Plot(props) {
  const [Component, setComponent] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let live = true;
    loadPlot().then(
      (C) => live && setComponent(() => C),
      (e) => live && setError(e.message),
    );
    return () => {
      live = false;
    };
  }, []);

  if (Component) return <Component {...props} />;
  return (
    <div style={props.style} className={`${props.className || ''} flex items-center justify-center text-sm text-gray-500`}>
      {error || 'Loading plot…'}
    </div>
  );
}
