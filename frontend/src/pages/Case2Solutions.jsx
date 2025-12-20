import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { BlockMath, InlineMath } from 'react-katex';
import Plot from 'react-plotly.js';
import npyjs from 'npyjs';

export default function Case2Solutions() {
  const [plotData, setPlotData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function loadData() {
      try {
        const npy = new npyjs();

        // Load training data
        const trainResponse = await fetch('/case2/data/train.npy');
        const trainArrayBuffer = await trainResponse.arrayBuffer();
        const trainData = await npy.load(trainArrayBuffer);

        // Load test X data
        const testXResponse = await fetch('/case2/data/test_x.npy');
        const testXArrayBuffer = await testXResponse.arrayBuffer();
        const testXData = await npy.load(testXArrayBuffer);

        // Load optimal samples from rectified flow
        const optimalResponse = await fetch('/case2/data/optimal_samples.npy');
        const optimalArrayBuffer = await optimalResponse.arrayBuffer();
        const optimalData = await npy.load(optimalArrayBuffer);

        // Extract data
        const trainX = [];
        const trainY = [];
        for (let i = 0; i < trainData.shape[0]; i++) {
          trainX.push(trainData.data[i * 2]);
          trainY.push(trainData.data[i * 2 + 1]);
        }

        // Extract test X and optimal samples
        const testX = Array.from(testXData.data);
        const sample1 = [];
        const sample2 = [];
        for (let i = 0; i < optimalData.shape[0]; i++) {
          sample1.push(optimalData.data[i * 2]);
          sample2.push(optimalData.data[i * 2 + 1]);
        }

        // Create a range of x values for the true conditional expectation curve
        const xMin = Math.min(...trainX);
        const xMax = Math.max(...trainX);
        const xRange = [];
        const trueExpectation = [];
        const numPoints = Math.ceil((xMax - xMin) * 10);
        for (let i = 0; i <= numPoints; i++) {
          const x = xMin + (i * 0.1);
          xRange.push(x);
          trueExpectation.push(5 * Math.cos(x)); // E[y|x] = 5*cos(x)
        }

        // Prepare plot data
        const traces = [
          {
            x: trainX,
            y: trainY,
            mode: 'markers',
            type: 'scatter',
            name: 'Training Data',
            marker: {
              color: 'rgba(59, 130, 246, 0.5)',
              size: 5,
            },
          },
          {
            x: xRange,
            y: trueExpectation,
            mode: 'lines',
            type: 'scatter',
            name: 'True E[y|x] = 5cos(x)',
            line: {
              color: 'rgba(220, 38, 38, 1)',
              width: 3,
            },
          },
          {
            x: [...testX, ...testX],
            y: [...sample1, ...sample2],
            mode: 'markers',
            type: 'scatter',
            name: 'Reference Solution Samples',
            marker: {
              color: 'rgba(34, 197, 94, 0.6)',
              size: 6,
              symbol: 'circle',
            },
          },
        ];

        setPlotData(traces);
        setLoading(false);
      } catch (err) {
        setError(`Error loading data: ${err.message}`);
        setLoading(false);
      }
    }

    loadData();
  }, []);

  return (
    <div className="min-h-screen bg-white">
      <div className="max-w-6xl mx-auto px-6 py-16">
        <Link to="/case2" className="text-blue-600 hover:text-blue-800 mb-8 inline-block">
          ← Back to challenge
        </Link>

        <h1 className="text-4xl font-light text-gray-900 mb-8">
          Case Study 2: Solutions
        </h1>

        <section className="mb-12">
          <h2 className="text-2xl font-medium text-gray-900 mb-4">The True Data Generation Process</h2>
          <div className="prose max-w-none text-gray-700 space-y-4">
            <p>
              The data for this challenge was generated using the same process as Case Study 1:
            </p>
            
            <div className="bg-gray-50 p-6 rounded-lg my-6 overflow-x-auto">
              <h3 className="font-medium text-gray-900 mb-3">Data Generation:</h3>
              <div className="overflow-x-auto">
                <BlockMath math="x \sim N(4, 1)" />
              </div>
              <div className="my-2">
                <InlineMath math="y \mid x" /> is an equal parts mixture:
              </div>
              <div className="overflow-x-auto">
                <BlockMath math="y \mid x \sim \frac{1}{2} N(10\cos(x), 1) + \frac{1}{2} N(0, 1)" />
              </div>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-medium text-gray-900 mb-4">Reference Solution: Rectified Flow Matching</h2>
          <div className="prose max-w-none text-gray-700 space-y-4">
            <p>
              The reference solution uses <strong>rectified flow matching</strong>, a powerful technique
              for learning to generate samples from the conditional distribution without knowing its structure.
            </p>
            
            <div className="bg-gray-50 p-6 rounded-lg my-6">
              <h3 className="font-medium text-gray-900 mb-3">Algorithm:</h3>
              <ol className="list-decimal list-inside space-y-2">
                <li>Generate 10 random time values <InlineMath math="t \in [0,1]" /> per training datapoint</li>
                <li>For each <InlineMath math="t" />, generate random noise <InlineMath math="\epsilon \sim N(0,1)" /></li>
                <li>
                  Compute interpolated points: <InlineMath math="z_t = y \cdot t + (1-t) \cdot \epsilon" />
                </li>
                <li>
                  Train MLP to predict the velocity field <InlineMath math="v = y - \epsilon" /> from 
                  Fourier embeddings of <InlineMath math="(x, t, z_t)" />
                </li>
                <li>
                  Generate samples by solving ODE: start from <InlineMath math="z_0 \sim N(0,1)" /> and 
                  integrate <InlineMath math="dz/dt = v(x,t,z)" /> to <InlineMath math="t=1" />
                </li>
              </ol>
            </div>

            <p>
              This approach learns to transport samples from a simple distribution <InlineMath math="N(0,1)" />
              {' '}to the complex target conditional distribution <InlineMath math="p(y|x)" /> by following
              the learned velocity field.
            </p>

            <div className="bg-blue-50 p-4 rounded-lg my-4">
              <p className="text-sm">
                <strong>Performance:</strong> The improved rectified flow implementation achieves
                an energy score of ~2.0. For comparison, oracle access to the true mixture structure
                (ground truth) achieves ~0.5, representing the best possible performance.
              </p>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-medium text-gray-900 mb-4">Visualization</h2>
          <div className="bg-white border border-gray-200 rounded-lg p-4 sm:p-6">
            {loading && (
              <div className="text-center py-12">
                <div className="text-gray-600">Loading data...</div>
              </div>
            )}
            
            {error && (
              <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
                {error}
              </div>
            )}
            
            {plotData && (
              <Plot
                data={plotData}
                layout={{
                  title: {
                    text: 'Training Data Showing Mixture Distribution',
                    font: { size: window.innerWidth < 640 ? 14 : 16 }
                  },
                  xaxis: { title: 'x' },
                  yaxis: { title: 'y' },
                  hovermode: 'closest',
                  showlegend: true,
                  legend: {
                    x: window.innerWidth < 640 ? 0 : 0.02,
                    y: window.innerWidth < 640 ? -0.15 : 0.98,
                    orientation: window.innerWidth < 640 ? 'h' : 'v',
                    xanchor: 'left',
                    yanchor: window.innerWidth < 640 ? 'top' : 'top',
                    bgcolor: 'rgba(255, 255, 255, 0.8)',
                    bordercolor: 'rgba(0, 0, 0, 0.2)',
                    borderwidth: 1,
                  },
                  autosize: true,
                  margin: { 
                    l: window.innerWidth < 640 ? 40 : 50, 
                    r: window.innerWidth < 640 ? 10 : 20, 
                    t: window.innerWidth < 640 ? 40 : 50, 
                    b: window.innerWidth < 640 ? 80 : 50 
                  },
                }}
                style={{ width: '100%', height: window.innerWidth < 640 ? '400px' : '600px' }}
                config={{ responsive: true }}
                useResizeHandler={true}
              />
            )}
          </div>
          
          <div className="mt-6 prose max-w-none text-gray-700">
            <h3 className="text-lg font-medium text-gray-900 mb-2">Interpretation:</h3>
            <ul className="space-y-2">
              <li>
                <strong>Blue points</strong>: Training data showing the mixture distribution
              </li>
              <li>
                <strong>Red curve</strong>: Conditional expectation E[y|x] = 5cos(x)
              </li>
              <li>
                <strong>Green circles</strong>: Samples from rectified flow reference solution
              </li>
            </ul>
            <p className="mt-4">
              Notice how the data splits into two clusters: one following the cosine pattern
              (around the red curve) and another centered around y=0. The reference solution samples
              (green circles) demonstrate how rectified flow matching learns to capture
              this bimodal distribution, with samples spread across both modes.
            </p>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-medium text-gray-900 mb-4">Understanding the Energy Score</h2>
          <div className="bg-blue-50 p-6 rounded-lg">
            <div className="prose max-w-none text-gray-700">
              <p className="mb-4">
                The 2-sample energy score balances two objectives:
              </p>
              <ol className="space-y-3">
                <li>
                  <strong>Accuracy</strong>: Both samples should be close to the true value
                  (minimizing <InlineMath math="|Y - X_1|" /> and <InlineMath math="|Y - X_2|" />)
                </li>
                <li>
                  <strong>Diversity</strong>: The samples should be different from each other
                  (maximizing <InlineMath math="|X_1 - X_2|" />), which encourages exploration
                  of the distribution
                </li>
              </ol>
              <p className="mt-4">
                A good strategy often involves sampling from different modes or regions of
                the conditional distribution, rather than returning two nearly identical
                predictions.
              </p>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-medium text-gray-900 mb-4">Comparison with Case Study 1</h2>
          <div className="prose max-w-none text-gray-700">
            <div className="bg-gray-50 p-6 rounded-lg">
              <table className="min-w-full">
                <thead>
                  <tr className="border-b border-gray-300">
                    <th className="text-left py-2 pr-4">Aspect</th>
                    <th className="text-left py-2 pr-4">Case Study 1</th>
                    <th className="text-left py-2">Case Study 2</th>
                  </tr>
                </thead>
                <tbody className="text-sm">
                  <tr className="border-b border-gray-200">
                    <td className="py-2 pr-4 font-medium">Prediction</td>
                    <td className="py-2 pr-4">Single point</td>
                    <td className="py-2">Two samples</td>
                  </tr>
                  <tr className="border-b border-gray-200">
                    <td className="py-2 pr-4 font-medium">Output shape</td>
                    <td className="py-2 pr-4">100×1</td>
                    <td className="py-2">100×2</td>
                  </tr>
                  <tr className="border-b border-gray-200">
                    <td className="py-2 pr-4 font-medium">Metric</td>
                    <td className="py-2 pr-4">RMSE</td>
                    <td className="py-2">Energy Score</td>
                  </tr>
                  <tr className="border-b border-gray-200">
                    <td className="py-2 pr-4 font-medium">Goal</td>
                    <td className="py-2 pr-4">Minimize prediction error</td>
                    <td className="py-2">Represent distribution</td>
                  </tr>
                  <tr>
                    <td className="py-2 pr-4 font-medium">Optimal strategy</td>
                    <td className="py-2 pr-4">Predict E[y|x]</td>
                    <td className="py-2">Sample from distribution</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
