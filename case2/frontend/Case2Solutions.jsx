import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { BlockMath, InlineMath } from 'react-katex';
import Plot from 'react-plotly.js';
import npyjs from 'npyjs';

export default function Case2Solutions() {
  const [plotData, setPlotData] = useState(null);
  const [trainingHistory, setTrainingHistory] = useState(null);
  const [infiniteDataHistory, setInfiniteDataHistory] = useState(null);
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

        // Load optimal samples from rectified flow
        const optimalResponse = await fetch('/case2/data/optimal_samples.npy');
        const optimalArrayBuffer = await optimalResponse.arrayBuffer();
        const optimalData = await npy.load(optimalArrayBuffer);
        
        // Load infinite data samples
        let infiniteDataData = null;
        try {
          const infiniteDataResponse = await fetch('/case2/data/infinitedata_samples.npy');
          const infiniteDataArrayBuffer = await infiniteDataResponse.arrayBuffer();
          infiniteDataData = await npy.load(infiniteDataArrayBuffer);
        } catch (err) {
          console.warn('Infinite data samples not available:', err);
        }
        
        // Load training history
        try {
          const historyResponse = await fetch('/case2/data/reference_training_history.json');
          const history = await historyResponse.json();
          setTrainingHistory(history);
        } catch (err) {
          console.warn('Training history not available:', err);
        }
        
        // Load infinite data training history
        try {
          const infHistoryResponse = await fetch('/case2/data/infinitedata_training_history.json');
          const infHistory = await infHistoryResponse.json();
          setInfiniteDataHistory(infHistory);
        } catch (err) {
          console.warn('Infinite data training history not available:', err);
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
        const sample1 = [];
        const sample2 = [];
        for (let i = 0; i < optimalData.shape[0]; i++) {
          sample1.push(optimalData.data[i * 2]);
          sample2.push(optimalData.data[i * 2 + 1]);
        }
        
        // Extract infinite data samples if available
        const infiniteSample1 = [];
        const infiniteSample2 = [];
        if (infiniteDataData) {
          for (let i = 0; i < infiniteDataData.shape[0]; i++) {
            infiniteSample1.push(infiniteDataData.data[i * 2]);
            infiniteSample2.push(infiniteDataData.data[i * 2 + 1]);
          }
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
        const allY = [...trainY, ...sample1, ...sample2];
        if (infiniteDataData) {
          allY.push(...infiniteSample1, ...infiniteSample2);
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
          reference: {
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
        
        // Add infinite data samples if available
        if (infiniteDataData) {
          allPlotData.infinite = {
            x: [...testX, ...testX],
            y: [...infiniteSample1, ...infiniteSample2],
            mode: 'markers',
            type: 'scatter',
            name: 'Infinite Data Solution',
            marker: {
              color: 'rgba(168, 85, 247, 0.6)',
              size: 6,
            },
          };
        }

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
          <h2 className="text-2xl font-medium text-gray-900 mb-4">Solutions Overview</h2>
          <div className="prose max-w-none text-gray-700 space-y-4">
            <p>
              We present two solutions using <strong>rectified flow matching</strong> with the same
              architecture (256, 128, 128, 64 hidden layers) and the same raw features, but different training data:
            </p>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 my-6">
              <div className="bg-green-50 p-6 rounded-lg border border-green-200">
                <h3 className="font-medium text-gray-900 mb-3">Reference Solution</h3>
                <ul className="list-disc list-inside space-y-2 text-sm">
                  <li><strong>Features:</strong> Raw features only</li>
                  <li><strong>Training Data:</strong> 900 finite samples from dataset</li>
                  <li><strong>Energy Score:</strong> {trainingHistory?.final_energy_score ? trainingHistory.final_energy_score.toFixed(4) : '~1.8'}</li>
                  <li><strong>Training Time:</strong> {trainingHistory?.training_time ? `${trainingHistory.training_time.toFixed(1)}s` : '~50s'}</li>
                </ul>
              </div>
              
              {infiniteDataHistory && (
                <div className="bg-purple-50 p-6 rounded-lg border border-purple-200">
                  <h3 className="font-medium text-gray-900 mb-3">Infinite Data Solution</h3>
                  <ul className="list-disc list-inside space-y-2 text-sm">
                    <li><strong>Features:</strong> Raw features only</li>
                    <li><strong>Training Data:</strong> Fresh samples each epoch (infinite)</li>
                    <li><strong>Energy Score:</strong> {infiniteDataHistory.final_energy_score.toFixed(4)}</li>
                    <li><strong>Training Time:</strong> {infiniteDataHistory.training_time.toFixed(1)}s</li>
                  </ul>
                </div>
              )}
            </div>
            
            <p>
              Both solutions use identical architectures and features. The key difference is the training data:
              the reference solution uses a fixed dataset while the infinite data solution generates fresh samples
              from the true distribution each epoch.
            </p>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-medium text-gray-900 mb-4">Reference Solution: Rectified Flow Matching with Finite Data</h2>
          <div className="prose max-w-none text-gray-700 space-y-4">
            <p>
              The reference solution uses <strong>rectified flow matching</strong>, a powerful technique
              for learning to generate samples from the conditional distribution without knowing its structure.
            </p>
            
            <div className="bg-gray-50 p-6 rounded-lg my-6">
              <h3 className="font-medium text-gray-900 mb-3">Algorithm:</h3>
              <ol className="list-decimal list-inside space-y-2">
                <li>For each training epoch, generate fresh random time values and noise for each sample</li>
                <li>Use exactly 3 t values per sample: t=0 (beginning), t=1 (ending), t=random (middle)</li>
                <li>For each <InlineMath math="t" />, generate random noise <InlineMath math="\epsilon \sim N(0,1)" /></li>
                <li>
                  Compute interpolated points: <InlineMath math="z_t = y \cdot t + (1-t) \cdot \epsilon" />
                </li>
                <li>
                  Train MLP using partial_fit to predict the velocity field <InlineMath math="v = y - \epsilon" /> from 
                  raw features of <InlineMath math="(x, t, z_t)" />
                </li>
                <li>
                  Generate samples by solving ODE: start from <InlineMath math="z_0 \sim N(0,1)" /> and 
                  integrate <InlineMath math="dz/dt = v(x,t,z)" /> to <InlineMath math="t=1" /> using scipy solve_ivp
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
                For comparison, oracle access to the true mixture structure
                (ground truth) achieves ~0.5, representing the best possible performance.
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

        {infiniteDataHistory && (
          <section className="mb-12">
            <h2 className="text-2xl font-medium text-gray-900 mb-4">Infinite Data Solution: Rectified Flow with Synthetic Data</h2>
            <div className="prose max-w-none text-gray-700 space-y-4">
              <p>
                The infinite data solution uses the same rectified flow matching algorithm but with a different approach to training:
              </p>
              
              <div className="bg-gray-50 p-6 rounded-lg my-6">
                <h3 className="font-medium text-gray-900 mb-3">Key Differences:</h3>
                <ul className="list-disc list-inside space-y-2">
                  <li>
                    <strong>Training Data:</strong> Generates fresh samples from the true generative model each epoch
                    (x ~ N(4,1), y|x ~ mixture) rather than using the fixed 900 training samples
                  </li>
                  <li>
                    <strong>Features:</strong> Uses the same raw features (x, t, z_t) as the reference solution
                  </li>
                  <li>
                    <strong>Architecture:</strong> Same (256, 128, 128, 64) hidden layers as reference solution
                  </li>
                  <li>
                    <strong>Benefit:</strong> Infinite fresh data prevents overfitting to a fixed training set
                  </li>
                </ul>
              </div>

              <div className="bg-purple-50 p-4 rounded-lg my-4">
                <p className="text-sm">
                  <strong>Performance:</strong> The infinite data approach achieves
                  an energy score of {infiniteDataHistory.final_energy_score.toFixed(4)},
                  similar to the reference solution.
                  <br /><br />
                  <strong>Training Time:</strong> {infiniteDataHistory.training_time.toFixed(2)} seconds on {infiniteDataHistory.hardware}
                  <br />
                  <strong>Architecture:</strong> {infiniteDataHistory.hidden_layers.join(', ')} hidden layers
                </p>
              </div>
              
              <p className="text-sm text-gray-600 italic">
                Note: This approach is only possible when the true data generation process is known,
                making it a useful comparison for understanding the impact of training data quantity.
              </p>
            </div>
          </section>
        )}

        {trainingHistory && (
          <section className="mb-12">
            <h2 className="text-2xl font-medium text-gray-900 mb-4">Reference Solution Training Progress</h2>
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
                      line: { color: 'rgba(59, 130, 246, 1)' },
                      marker: { size: 6 },
                    },
                    {
                      x: trainingHistory.epochs,
                      y: trainingHistory.val_mse,
                      mode: 'lines+markers',
                      name: 'Validation MSE',
                      line: { color: 'rgba(220, 38, 38, 1)' },
                      marker: { size: 6 },
                    },
                  ]}
                  layout={{
                    title: {
                      text: 'Training and Validation MSE per Epoch',
                      font: { size: window.innerWidth < 640 ? 14 : 16 }
                    },
                    xaxis: { title: 'Epoch' },
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
                  <h3 className="text-lg font-medium text-gray-900 mb-3">Energy Score (CRPS) on Validation Set</h3>
                  <Plot
                    data={[
                      {
                        x: trainingHistory.epochs,  // Use actual epoch values
                        y: trainingHistory.val_energy_scores,
                        mode: 'lines+markers',
                        name: 'Validation Energy Score',
                        marker: { size: 8, color: 'rgba(16, 185, 129, 1)' },
                        line: { color: 'rgba(16, 185, 129, 1)' },
                      },
                    ]}
                    layout={{
                      title: {
                        text: 'Validation Energy Score (Lower is Better)',
                        font: { size: window.innerWidth < 640 ? 14 : 16 }
                      },
                      xaxis: { title: 'Epoch' },
                      yaxis: { title: 'Energy Score (CRPS)' },
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
                  <strong>Training Details:</strong> Uses partial_fit with fresh random t and ε samples each epoch.
                  Each training sample generates 3 t values: t=0 (beginning), t=1 (ending), and t=random (middle).
                  Model trained with early stopping based on validation MSE.
                </p>
                {trainingHistory?.final_energy_score && (
                  <p className="mt-2">
                    <strong>Final Energy Score:</strong> {trainingHistory.final_energy_score.toFixed(4)} 
                    {' '}(computed on test set after training)
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

        {infiniteDataHistory && (
          <section className="mb-12">
            <h2 className="text-2xl font-medium text-gray-900 mb-4">Infinite Data Solution Training Progress</h2>
            <div className="bg-white border border-gray-200 rounded-lg p-4 sm:p-6">
              {/* MSE Loss Plot */}
              <div className="mb-8">
                <h3 className="text-lg font-medium text-gray-900 mb-3">Mean Squared Error (MSE) Loss</h3>
                <Plot
                  data={[
                    {
                      x: infiniteDataHistory.epochs,
                      y: infiniteDataHistory.train_mse,
                      mode: 'lines+markers',
                      name: 'Train MSE',
                      line: { color: 'rgba(168, 85, 247, 1)' },
                      marker: { size: 6 },
                    },
                    {
                      x: infiniteDataHistory.epochs,
                      y: infiniteDataHistory.val_mse,
                      mode: 'lines+markers',
                      name: 'Validation MSE',
                      line: { color: 'rgba(220, 38, 38, 1)' },
                      marker: { size: 6 },
                    },
                  ]}
                  layout={{
                    title: {
                      text: 'Training and Validation MSE per Epoch',
                      font: { size: window.innerWidth < 640 ? 14 : 16 }
                    },
                    xaxis: { title: 'Epoch' },
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
              {infiniteDataHistory.val_energy_scores && infiniteDataHistory.val_energy_scores.length > 0 && (
                <div className="mb-6">
                  <h3 className="text-lg font-medium text-gray-900 mb-3">Energy Score (CRPS) on Validation Set</h3>
                  <Plot
                    data={[
                      {
                        x: infiniteDataHistory.epochs,
                        y: infiniteDataHistory.val_energy_scores,
                        mode: 'lines+markers',
                        name: 'Validation Energy Score',
                        marker: { size: 8, color: 'rgba(168, 85, 247, 1)' },
                        line: { color: 'rgba(168, 85, 247, 1)' },
                      },
                    ]}
                    layout={{
                      title: {
                        text: 'Validation Energy Score (Lower is Better)',
                        font: { size: window.innerWidth < 640 ? 14 : 16 }
                      },
                      xaxis: { title: 'Epoch' },
                      yaxis: { title: 'Energy Score (CRPS)' },
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
                  <strong>Training Details:</strong> Uses partial_fit with fresh random t and ε samples each epoch.
                  Each training sample generates 3 t values: t=0 (beginning), t=1 (ending), and t=random (middle).
                  Fresh training data is generated from the true generative model each epoch.
                  Model trained with early stopping based on validation MSE.
                </p>
                {infiniteDataHistory?.final_energy_score && (
                  <p className="mt-2">
                    <strong>Final Energy Score:</strong> {infiniteDataHistory.final_energy_score.toFixed(4)} 
                    {' '}(computed on test set after training)
                  </p>
                )}
                {infiniteDataHistory?.training_time && (
                  <p className="mt-2">
                    <strong>Training Time:</strong> {infiniteDataHistory.training_time.toFixed(2)} seconds
                    {infiniteDataHistory.hardware && ` on ${infiniteDataHistory.hardware}`}
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
                        value="reference"
                        checked={selectedSolution === 'reference'}
                        onChange={(e) => setSelectedSolution(e.target.value)}
                        className="mr-2 h-4 w-4 text-green-600 focus:ring-green-500 border-gray-300"
                      />
                      <span className="text-gray-700">Reference Solution</span>
                    </label>
                    {plotData.infinite && (
                      <label className="flex items-center">
                        <input
                          type="radio"
                          name="solution"
                          value="infinite"
                          checked={selectedSolution === 'infinite'}
                          onChange={(e) => setSelectedSolution(e.target.value)}
                          className="mr-2 h-4 w-4 text-purple-600 focus:ring-purple-500 border-gray-300"
                        />
                        <span className="text-gray-700">Infinite Data Solution</span>
                      </label>
                    )}
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
                  of two normal distributions in the data generation process.
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
            {selectedSolution === 'infinite' && infiniteDataHistory && (
              <>
                <p>
                  <strong>Purple points</strong>: Samples from the infinite data solution
                </p>
                <p className="mt-4">
                  The infinite data solution demonstrates how rectified flow matching with unlimited
                  training samples learns to capture the bimodal distribution, with samples spread
                  across both modes of the mixture.
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
