import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { BlockMath, InlineMath } from 'react-katex';
import Plot from 'react-plotly.js';
import npyjs from 'npyjs';

export default function Case2Solutions() {
  const [plotData, setPlotData] = useState(null);
  const [trainingHistory, setTrainingHistory] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedSolution, setSelectedSolution] = useState('training');

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

        // Load test Y data
        const testYResponse = await fetch('/case2/data/test_y.npy');
        const testYArrayBuffer = await testYResponse.arrayBuffer();
        const testYData = await npy.load(testYArrayBuffer);

        // Load optimal samples from rectified flow
        const optimalResponse = await fetch('/case2/data/optimal_samples.npy');
        const optimalArrayBuffer = await optimalResponse.arrayBuffer();
        const optimalData = await npy.load(optimalArrayBuffer);
        
        // Load reference solution training history from CSVs
        try {
          const refTrainingLossResponse = await fetch('/case2/data/reference_training_loss.csv');
          const refTrainingLossText = await refTrainingLossResponse.text();
          
          if (refTrainingLossText && refTrainingLossText.trim().length > 0 && !refTrainingLossText.includes('<!DOCTYPE')) {
            const refTrainingLossLines = refTrainingLossText.trim().split('\n').slice(1); // Skip header
            
            const refSteps = [];
            const refTrainMse = [];
            const refTestMse = [];
            let refLastTime = 0;
            
            for (const line of refTrainingLossLines) {
              const parts = line.split(',');
              // Format: step, train_loss, test_mse, time_seconds
              if (parts.length >= 4) {
                const step = parseInt(parts[0]);
                const loss = parseFloat(parts[1]);
                const testMse = parseFloat(parts[2]);
                const time = parseFloat(parts[3]);
                if (!isNaN(step) && !isNaN(loss)) {
                  refSteps.push(step);
                  refTrainMse.push(loss);
                  if (!isNaN(testMse)) refTestMse.push(testMse);
                  if (!isNaN(time)) refLastTime = time;
                }
              } else if (parts.length >= 3) {
                // Backwards compatibility: step, train_loss, time_seconds
                const step = parseInt(parts[0]);
                const loss = parseFloat(parts[1]);
                const time = parseFloat(parts[2]);
                if (!isNaN(step) && !isNaN(loss)) {
                  refSteps.push(step);
                  refTrainMse.push(loss);
                  if (!isNaN(time)) refLastTime = time;
                }
              }
            }
            
            // Load reference energy score CSV
            const refEnergyScoreResponse = await fetch('/case2/data/reference_energy_score.csv');
            const refEnergyScoreText = await refEnergyScoreResponse.text();
            const refEnergyScoreLines = refEnergyScoreText.trim().split('\n').slice(1); // Skip header
            
            const refEnergySteps = [];
            const refEnergyScores = [];
            let refFinalEnergyScore = 0;
            
            for (const line of refEnergyScoreLines) {
              const parts = line.split(',');
              if (parts.length >= 2) {
                const step = parseInt(parts[0]);
                const score = parseFloat(parts[1]);
                if (!isNaN(step) && !isNaN(score)) {
                  refEnergySteps.push(step);
                  refEnergyScores.push(score);
                  refFinalEnergyScore = score; // Last one is final
                }
              }
            }
            
            // Build history object from CSVs
            const refHistory = {
              epochs: refSteps,
              train_mse: refTrainMse,
              test_mse: refTestMse.length > 0 ? refTestMse : null,
              val_energy_scores: refEnergyScores,
              energy_epochs: refEnergySteps,
              training_time: refLastTime,
              hardware: 'T4 GPU (Modal)',
              final_energy_score: refFinalEnergyScore,
              architecture: 'raw_features_only',
              hidden_layers: [256, 128, 128, 64],
              training_data: '900 samples (finite)'
            };
            
            setTrainingHistory(refHistory);
          }
        } catch (err) {
          console.warn('Reference training history CSVs not available:', err);
        }
        
        // Load reference scatter samples from CSV (if available)
        let referenceScatterData = null;
        try {
          const refScatterResponse = await fetch('/case2/data/reference_scatter_samples.csv');
          const refScatterText = await refScatterResponse.text();
          
          if (refScatterText && refScatterText.trim().length > 0 && !refScatterText.includes('<!DOCTYPE')) {
            const refScatterLines = refScatterText.trim().split('\n').slice(1); // Skip header
            
            const refScatterX = [];
            const refScatterYSampled = [];
            
            for (const line of refScatterLines) {
              if (line && line.trim().length > 0) {
                const parts = line.split(',');
                if (parts.length >= 3) {
                  const x = parseFloat(parts[0]);
                  const ySampled = parseFloat(parts[2]);
                  if (!isNaN(x) && !isNaN(ySampled)) {
                    refScatterX.push(x);
                    refScatterYSampled.push(ySampled);
                  }
                }
              }
            }
            
            if (refScatterX.length > 0) {
              referenceScatterData = { x: refScatterX, y: refScatterYSampled };
            }
          }
        } catch (err) {
          console.warn('Reference scatter samples CSV not available:', err);
        }

        // Extract data
        const trainX = [];
        const trainY = [];
        for (let i = 0; i < trainData.shape[0]; i++) {
          trainX.push(trainData.data[i * 2]);
          trainY.push(trainData.data[i * 2 + 1]);
        }

        // Extract test X and optimal samples
        const testX = Array.from(testXData.data);
        const testY = Array.from(testYData.data);
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

        // Calculate fixed axis ranges based on all data
        const allX = [...trainX, ...testX, ...testX];
        const allY = [...trainY, ...testY, ...sample1, ...sample2];
        if (referenceScatterData) {
          allX.push(...referenceScatterData.x);
          allY.push(...referenceScatterData.y);
        }
        const axisXMin = Math.min(...allX);
        const axisXMax = Math.max(...allX);
        const axisYMin = Math.min(...allY, ...trueExpectation);
        const axisYMax = Math.max(...allY, ...trueExpectation);
        
        // Add padding to axis ranges
        const xPadding = (axisXMax - axisXMin) * 0.05;
        const yPadding = (axisYMax - axisYMin) * 0.05;
        
        // Store all plot data components with same marker style for all scatters
        const allPlotData = {
          training: {
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
          test: {
            x: testX,
            y: testY,
            mode: 'markers',
            type: 'scatter',
            name: 'Test Data',
            marker: {
              color: 'rgba(251, 146, 60, 0.5)',
              size: 5,
            },
          },
          trueExpectation: {
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
          // Use reference scatter CSV if available, otherwise fall back to npy samples
          reference: referenceScatterData ? {
            x: referenceScatterData.x,
            y: referenceScatterData.y,
            mode: 'markers',
            type: 'scatter',
            name: 'Reference Solution',
            marker: {
              color: 'rgba(34, 197, 94, 0.6)',
              size: 6,
            },
          } : {
            x: [...testX, ...testX],
            y: [...sample1, ...sample2],
            mode: 'markers',
            type: 'scatter',
            name: 'Reference Solution',
            marker: {
              color: 'rgba(34, 197, 94, 0.6)',
              size: 6,
            },
          },
          axisRanges: {
            x: [axisXMin - xPadding, axisXMax + xPadding],
            y: [axisYMin - yPadding, axisYMax + yPadding],
          },
        };

        setPlotData(allPlotData);
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
          <h2 className="text-2xl font-medium text-gray-900 mb-4">Reference Solution</h2>
          <div className="prose max-w-none text-gray-700 space-y-4">
            <p>
              We present a solution using <strong>rectified flow matching</strong>. See below for more details on how it works.
            </p>
            
            <div className="my-6">
              <div className="bg-green-50 p-6 rounded-lg border border-green-200">
                <h3 className="font-medium text-gray-900 mb-3">Reference Solution</h3>
                <ul className="list-disc list-inside space-y-2 text-sm">
                  <li><strong>Training Data:</strong> 900 finite samples from dataset</li>
                  <li><strong>Energy Score:</strong> {trainingHistory?.final_energy_score ? trainingHistory.final_energy_score.toFixed(4) : '~1.8'}</li>
                  <li><strong>Training Time:</strong> {trainingHistory?.training_time ? `${trainingHistory.training_time.toFixed(1)}s` : '~50s'}</li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-medium text-gray-900 mb-4">Rectified Flow Matching</h2>
          <div className="prose max-w-none text-gray-700 space-y-4">
            <p>
              The reference solution uses <strong>rectified flow matching</strong>, a powerful technique
              for learning to generate samples from the conditional distribution without knowing its structure.
            </p>
            
            <div className="bg-gray-50 p-6 rounded-lg my-6">
              <h3 className="font-medium text-gray-900 mb-3">Algorithm:</h3>
              <ol className="list-decimal list-inside space-y-2">
                <li>For each training step, sample time values <InlineMath math="t \sim \text{Uniform}(0, 1)" /> and noise <InlineMath math="\epsilon \sim N(0,1)" /></li>
                <li>
                  Compute interpolated points: <InlineMath math="z_t = y \cdot t + (1-t) \cdot \epsilon" />
                </li>
                <li>
                  Train MLP to predict the velocity field <InlineMath math="v = y - \epsilon" /> from 
                  inputs <InlineMath math="(x, t, z_t)" />
                </li>
                <li>
                  Generate samples by solving ODE: start from <InlineMath math="z_0 \sim N(0,1)" /> and 
                  integrate <InlineMath math="dz/dt = v(x,t,z)" /> to <InlineMath math="t=1" /> using diffrax
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
                <strong>Performance:</strong> The rectified flow implementation achieves
                an energy score of {trainingHistory?.final_energy_score ? `${trainingHistory.final_energy_score.toFixed(4)}` : '~1.8'}.
                For comparison, random sampling from the true mixture distribution
                (ground truth) achieves ~1.9.
                {trainingHistory?.training_time && (
                  <>
                    <br /><br />
                    <strong>Training Time:</strong> {trainingHistory.training_time.toFixed(2)} seconds on {trainingHistory.hardware}
                  </>
                )}
              </p>
            </div>
          </div>
        </section>

        {trainingHistory && (
          <section className="mb-12">
            <h2 className="text-2xl font-medium text-gray-900 mb-4">Training Progress</h2>
            <div className="bg-white border border-gray-200 rounded-lg p-4 sm:p-6">
              {/* MSE Loss Plot */}
              <div className="mb-8">
                <h3 className="text-lg font-medium text-gray-900 mb-3">Mean Squared Error (MSE) Loss</h3>
                <Plot
                  data={[
                    {
                      x: trainingHistory.epochs,
                      y: trainingHistory.train_mse,
                      mode: 'lines+markers',
                      name: 'Train MSE',
                      line: { color: 'rgba(34, 197, 94, 1)' },
                      marker: { size: 6 },
                    },
                  ]}
                  layout={{
                    title: {
                      text: 'Training MSE per Step',
                      font: { size: window.innerWidth < 640 ? 14 : 16 }
                    },
                    xaxis: { title: 'Step' },
                    yaxis: { title: 'MSE Loss' },
                    hovermode: 'closest',
                    showlegend: true,
                    legend: {
                      x: window.innerWidth < 640 ? 0 : 0.02,
                      y: window.innerWidth < 640 ? -0.2 : 0.98,
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
                      b: window.innerWidth < 640 ? 90 : 50 
                    },
                  }}
                  style={{ width: '100%', height: window.innerWidth < 640 ? '300px' : '400px' }}
                  config={{ responsive: true }}
                  useResizeHandler={true}
                />
              </div>
              
              {/* Energy Score Plot */}
              {trainingHistory.val_energy_scores && trainingHistory.val_energy_scores.length > 0 && (
                <div className="mb-6">
                  <h3 className="text-lg font-medium text-gray-900 mb-3">Energy Score on Fixed 100 Test Points</h3>
                  <Plot
                    data={[
                      {
                        x: trainingHistory.energy_epochs || trainingHistory.epochs,  // Use energy_epochs if available (CSV format)
                        y: trainingHistory.val_energy_scores,
                        mode: 'lines+markers',
                        name: 'Energy Score',
                        marker: { size: 8, color: 'rgba(34, 197, 94, 1)' },
                        line: { color: 'rgba(34, 197, 94, 1)' },
                      },
                    ]}
                    layout={{
                      title: {
                        text: 'Energy Score on Fixed Test Set (Lower is Better)',
                        font: { size: window.innerWidth < 640 ? 14 : 16 }
                      },
                      xaxis: { title: 'Step' },
                      yaxis: { title: 'Energy Score' },
                      hovermode: 'closest',
                      showlegend: false,
                      autosize: true,
                      margin: { 
                        l: window.innerWidth < 640 ? 40 : 50, 
                        r: window.innerWidth < 640 ? 10 : 20, 
                        t: window.innerWidth < 640 ? 40 : 50, 
                        b: window.innerWidth < 640 ? 50 : 50 
                      },
                    }}
                    style={{ width: '100%', height: window.innerWidth < 640 ? '250px' : '300px' }}
                    config={{ responsive: true }}
                    useResizeHandler={true}
                  />
                </div>
              )}
              
              {/* Test MSE Plot */}
              {trainingHistory.test_mse && trainingHistory.test_mse.length > 0 && (
                <div className="mb-6">
                  <h3 className="text-lg font-medium text-gray-900 mb-3">Test MSE Over Time</h3>
                  <Plot
                    data={[
                      {
                        x: trainingHistory.epochs,
                        y: trainingHistory.test_mse,
                        mode: 'lines+markers',
                        name: 'Test MSE',
                        marker: { size: 6, color: 'rgba(220, 38, 38, 1)' },
                        line: { color: 'rgba(220, 38, 38, 1)' },
                      },
                    ]}
                    layout={{
                      title: {
                        text: 'Test MSE per Step (on Fixed Test Flow Batch)',
                        font: { size: window.innerWidth < 640 ? 14 : 16 }
                      },
                      xaxis: { title: 'Step' },
                      yaxis: { title: 'Test MSE' },
                      hovermode: 'closest',
                      showlegend: false,
                      autosize: true,
                      margin: { 
                        l: window.innerWidth < 640 ? 40 : 50, 
                        r: window.innerWidth < 640 ? 10 : 20, 
                        t: window.innerWidth < 640 ? 40 : 50, 
                        b: window.innerWidth < 640 ? 50 : 50 
                      },
                    }}
                    style={{ width: '100%', height: window.innerWidth < 640 ? '250px' : '300px' }}
                    config={{ responsive: true }}
                    useResizeHandler={true}
                  />
                </div>
              )}
              
              <div className="mt-4 prose max-w-none text-gray-700 text-sm">
                <p>
                  <strong>Training Details:</strong> JAX-based training with diffrax ODE integration on T4 GPU. Uses minibatched AdamW optimization with gradient clipping for stability. Learning rate is halved halfway through training.
                </p>
                {trainingHistory?.final_energy_score && (
                  <p className="mt-2">
                    <strong>Final Energy Score:</strong> {trainingHistory.final_energy_score.toFixed(4)} 
                    {' '}(computed on fixed 100 test points)
                  </p>
                )}
                {trainingHistory?.training_time && (
                  <p className="mt-2">
                    <strong>Training Time:</strong> {trainingHistory.training_time.toFixed(2)} seconds
                    {trainingHistory.hardware && ` on ${trainingHistory.hardware}`}
                  </p>
                )}
              </div>
            </div>
          </section>
        )}

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
              <>
                {/* Radio button controls */}
                <div className="mb-6">
                  <label className="block text-sm font-medium text-gray-900 mb-3">
                    Select Data to Display:
                  </label>
                  <div className="space-y-2">
                    <label className="flex items-center">
                      <input
                        type="radio"
                        name="solution"
                        value="training"
                        checked={selectedSolution === 'training'}
                        onChange={(e) => setSelectedSolution(e.target.value)}
                        className="mr-2 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300"
                      />
                      <span className="text-gray-700">Training Data</span>
                    </label>
                    <label className="flex items-center">
                      <input
                        type="radio"
                        name="solution"
                        value="test"
                        checked={selectedSolution === 'test'}
                        onChange={(e) => setSelectedSolution(e.target.value)}
                        className="mr-2 h-4 w-4 text-orange-600 focus:ring-orange-500 border-gray-300"
                      />
                      <span className="text-gray-700">Test Data</span>
                    </label>
                    <label className="flex items-center">
                      <input
                        type="radio"
                        name="solution"
                        value="reference"
                        checked={selectedSolution === 'reference'}
                        onChange={(e) => setSelectedSolution(e.target.value)}
                        className="mr-2 h-4 w-4 text-green-600 focus:ring-green-500 border-gray-300"
                      />
                      <span className="text-gray-700">Reference Solution</span>
                    </label>
                  </div>
                </div>
                
                <Plot
                  data={[plotData[selectedSolution]]}
                  layout={{
                    title: {
                      text: 'Training Data Showing Mixture Distribution',
                      font: { size: window.innerWidth < 640 ? 14 : 16 }
                    },
                    xaxis: { 
                      title: 'x',
                      range: plotData.axisRanges.x,
                    },
                    yaxis: { 
                      title: 'y',
                      range: plotData.axisRanges.y,
                    },
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
                  config={{ 
                    responsive: true,
                    displayModeBar: false,
                    staticPlot: true,
                  }}
                  useResizeHandler={true}
                />
              </>
            )}
          </div>
          
          <div className="mt-6 prose max-w-none text-gray-700">
            <h3 className="text-lg font-medium text-gray-900 mb-2">Interpretation:</h3>
            {selectedSolution === 'training' && (
              <>
                <p>
                  <strong>Blue points</strong>: Training data showing the mixture distribution
                </p>
                <p className="mt-4">
                  Notice how the data splits into two clusters: one following the cosine pattern
                  and another centered around y=0. This bimodal distribution reflects the mixture
                  of two normal distributions in the data generation process. The training data is
                  denser (900 samples) compared to the test data.
                </p>
              </>
            )}
            {selectedSolution === 'test' && (
              <>
                <p>
                  <strong>Orange points</strong>: Test data showing the mixture distribution
                </p>
                <p className="mt-4">
                  The test data follows the same bimodal distribution as the training data but 
                  is less dense (100 samples). This sparser sampling makes it easier to see the 
                  individual data points and the two-mode structure of the mixture distribution.
                </p>
              </>
            )}
            {selectedSolution === 'reference' && (
              <>
                <p>
                  <strong>Green points</strong>: Samples from the reference solution trained on finite data
                </p>
                <p className="mt-4">
                  The reference solution demonstrates how rectified flow matching learns to capture
                  the bimodal distribution, with samples spread across both modes of the mixture.
                </p>
              </>
            )}
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-medium text-gray-900 mb-4">Understanding the Energy Score</h2>
          <div className="bg-blue-50 p-6 rounded-lg">
            <div className="prose max-w-none text-gray-700">
              <p className="mb-4">
                The energy score (with 900 samples per held-out point) provides a Monte Carlo estimate
                that balances two objectives:
              </p>
              <ol className="space-y-3">
                <li>
                  <strong>Accuracy</strong>: Samples should be close to the true value
                  (minimizing <InlineMath math="E[|Y - X_j|]" /> averaged over 900 samples)
                </li>
                <li>
                  <strong>Diversity</strong>: Samples should cover the distribution
                  (maximizing <InlineMath math="E[|X_j - X_{j'}|]" /> between different samples)
                </li>
              </ol>
              <p className="mt-4">
                A good strategy involves sampling from different modes or regions of
                the conditional distribution, capturing its full structure rather than
                returning nearly identical predictions.
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
                    <td className="py-2">900 samples</td>
                  </tr>
                  <tr className="border-b border-gray-200">
                    <td className="py-2 pr-4 font-medium">Output shape</td>
                    <td className="py-2 pr-4">100×1</td>
                    <td className="py-2">100×900</td>
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
