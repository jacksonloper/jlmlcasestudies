import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { BlockMath, InlineMath } from 'react-katex';
import Plot from 'react-plotly.js';
import npyjs from 'npyjs';

export default function Case1Solutions() {
  const [plotData, setPlotData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function loadData() {
      try {
        const npy = new npyjs();

        // Load training data
        const trainResponse = await fetch('/case1/data/train.npy');
        const trainArrayBuffer = await trainResponse.arrayBuffer();
        const trainData = await npy.load(trainArrayBuffer);

        // Load test data
        const testXResponse = await fetch('/case1/data/test_x.npy');
        const testXArrayBuffer = await testXResponse.arrayBuffer();
        const testXData = await npy.load(testXArrayBuffer);

        // Load MLP predictions
        const mlpResponse = await fetch('/case1/data/mlp_test_yhat.npy');
        const mlpArrayBuffer = await mlpResponse.arrayBuffer();
        const mlpData = await npy.load(mlpArrayBuffer);

        // Extract data
        const trainX = [];
        const trainY = [];
        for (let i = 0; i < trainData.shape[0]; i++) {
          trainX.push(trainData.data[i * 2]);
          trainY.push(trainData.data[i * 2 + 1]);
        }

        const testX = Array.from(testXData.data);
        const mlpPredictions = Array.from(mlpData.data);

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
            x: testX,
            y: mlpPredictions,
            mode: 'markers',
            type: 'scatter',
            name: 'MLP Predictions (test set)',
            marker: {
              color: 'rgba(34, 197, 94, 1)',
              size: 8,
              symbol: 'x',
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
        <Link to="/case1" className="text-blue-600 hover:text-blue-800 mb-8 inline-block">
          ← Back to challenge
        </Link>

        <h1 className="text-4xl font-light text-gray-900 mb-8">
          Case Study 1: Solutions
        </h1>

        <section className="mb-12">
          <h2 className="text-2xl font-medium text-gray-900 mb-4">The True Data Generation Process</h2>
          <div className="prose max-w-none text-gray-700 space-y-4">
            <p>
              The data for this challenge was generated using the following process:
            </p>
            
            <div className="bg-gray-50 p-6 rounded-lg my-6">
              <h3 className="font-medium text-gray-900 mb-3">Data Generation:</h3>
              <BlockMath math="x \sim N(4, 1)" />
              <div className="my-2">
                <InlineMath math="y \mid x" /> is an equal parts mixture:
              </div>
              <BlockMath math="y \mid x \sim \frac{1}{2} N(10\cos(x), 1) + \frac{1}{2} N(0, 1)" />
              <div className="mt-4">
                <p>This means the optimal prediction (minimizing MSE) is the conditional expectation:</p>
              </div>
              <BlockMath math="E[y \mid x] = \frac{1}{2} \cdot 10\cos(x) + \frac{1}{2} \cdot 0 = 5\cos(x)" />
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-medium text-gray-900 mb-4">Visualization</h2>
          <div className="bg-white border border-gray-200 rounded-lg p-6">
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
                  title: 'Training Data with True and Learned Conditional Expectations',
                  xaxis: { title: 'x' },
                  yaxis: { title: 'y' },
                  hovermode: 'closest',
                  showlegend: true,
                  legend: {
                    x: 0.02,
                    y: 0.98,
                    bgcolor: 'rgba(255, 255, 255, 0.8)',
                    bordercolor: 'rgba(0, 0, 0, 0.2)',
                    borderwidth: 1,
                  },
                  autosize: true,
                }}
                style={{ width: '100%', height: '600px' }}
                config={{ responsive: true }}
              />
            )}
          </div>
          
          <div className="mt-6 prose max-w-none text-gray-700">
            <h3 className="text-lg font-medium text-gray-900 mb-2">Interpretation:</h3>
            <p className="mb-4">
              <strong>Note:</strong> The scatter plot shows the <strong>training data</strong> (900 points),
              while the performance metrics below are evaluated on the <strong>test set</strong> (100 points).
            </p>
            <ul className="space-y-2">
              <li>
                <strong>Blue points</strong>: Training data showing the mixture distribution
              </li>
              <li>
                <strong>Red curve</strong>: True conditional expectation E[y|x] = 5cos(x) (optimal predictor)
              </li>
              <li>
                <strong>Green X markers</strong>: Predictions from tiny MLP on test set
              </li>
            </ul>
            <p className="mt-4">
              The <a 
                href="https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html"
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-600 hover:text-blue-800 underline"
              >
                MLP model
              </a> (16 hidden units) does an excellent job of learning the conditional expectation from the data.
              The mixture includes both a structured component (depending on x) and a component independent of x.
              {/* Note: These RMSE values are from the baseline model in case1/scripts/train_mlp.py */}
            </p>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-medium text-gray-900 mb-4">Baseline Performance (Test Set)</h2>
          <div className="bg-blue-50 p-6 rounded-lg">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h3 className="font-medium text-gray-900 mb-2">
                  Tiny <a 
                    href="https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-600 hover:text-blue-800 underline"
                  >
                    MLP
                  </a> (16 hidden units)
                </h3>
                <p className="text-3xl font-bold text-blue-700">RMSE ≈ 3.66</p>
                <p className="text-sm text-gray-600 mt-2">
                  Trained on 900 samples, evaluated on 100 test samples
                </p>
              </div>
              <div>
                <h3 className="font-medium text-gray-900 mb-2">Optimal Predictor</h3>
                <p className="text-3xl font-bold text-green-700">RMSE ≈ 3.63</p>
                <p className="text-sm text-gray-600 mt-2">
                  Using E[y|x] = 5cos(x) (theoretical best on test set)
                </p>
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
