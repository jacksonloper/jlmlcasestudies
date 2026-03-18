import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { BlockMath, InlineMath } from 'react-katex';
import Plot from 'react-plotly.js';
import npyjs from 'npyjs';

export default function Case5Solutions() {
  const [plotData, setPlotData] = useState(null);
  const [trainingHistory, setTrainingHistory] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedView, setSelectedView] = useState('training');

  useEffect(() => {
    async function loadData() {
      try {
        const npy = new npyjs();

        // Load training data
        const trainXResponse = await fetch(`${import.meta.env.BASE_URL}case5/data/train_x.npy`);
        const trainXBuffer = await trainXResponse.arrayBuffer();
        const trainXData = await npy.load(trainXBuffer);

        const trainYResponse = await fetch(`${import.meta.env.BASE_URL}case5/data/train_y.npy`);
        const trainYBuffer = await trainYResponse.arrayBuffer();
        const trainYData = await npy.load(trainYBuffer);

        // Load test data
        const testXResponse = await fetch(`${import.meta.env.BASE_URL}case5/data/test_x.npy`);
        const testXBuffer = await testXResponse.arrayBuffer();
        const testXData = await npy.load(testXBuffer);

        const testYResponse = await fetch(`${import.meta.env.BASE_URL}case5/data/test_y.npy`);
        const testYBuffer = await testYResponse.arrayBuffer();
        const testYData = await npy.load(testYBuffer);

        // Load true log-likelihoods
        const trueLLResponse = await fetch(`${import.meta.env.BASE_URL}case5/data/test_true_loglik.npy`);
        const trueLLBuffer = await trueLLResponse.arrayBuffer();
        const trueLLData = await npy.load(trueLLBuffer);

        // Extract arrays
        const trainX1 = [];
        const trainX2 = [];
        for (let i = 0; i < trainXData.shape[0]; i++) {
          trainX1.push(trainXData.data[i * 2]);
          trainX2.push(trainXData.data[i * 2 + 1]);
        }
        const trainY = Array.from(trainYData.data);

        const testX1 = [];
        const testX2 = [];
        for (let i = 0; i < testXData.shape[0]; i++) {
          testX1.push(testXData.data[i * 2]);
          testX2.push(testXData.data[i * 2 + 1]);
        }
        const testY = Array.from(testYData.data);
        const trueLogLik = Array.from(trueLLData.data);

        // Load reference solution CSVs
        let refHistory = null;
        try {
          const refLossResponse = await fetch(`${import.meta.env.BASE_URL}case5/data/reference_training_loss.csv`);
          const refLossText = await refLossResponse.text();

          if (refLossText && refLossText.trim().length > 0 && !refLossText.includes('<!DOCTYPE')) {
            const refLossLines = refLossText.trim().split('\n').slice(1);

            const refSteps = [];
            const refTrainLoss = [];
            const refTestMse = [];
            let refLastTime = 0;

            for (const line of refLossLines) {
              const parts = line.split(',');
              if (parts.length >= 4) {
                const step = parseInt(parts[0]);
                const loss = parseFloat(parts[1]);
                const testMse = parseFloat(parts[2]);
                const time = parseFloat(parts[3]);
                if (!isNaN(step) && !isNaN(loss)) {
                  refSteps.push(step);
                  refTrainLoss.push(loss);
                  if (!isNaN(testMse)) refTestMse.push(testMse);
                  if (!isNaN(time)) refLastTime = time;
                }
              }
            }

            // Load log-likelihood MSE CSV (may include loglik_mean column)
            let refLoglikMseSteps = [];
            let refLoglikMseValues = [];
            let refLoglikMeanValues = [];
            try {
              const refLoglikResponse = await fetch(`${import.meta.env.BASE_URL}case5/data/reference_loglik_mse.csv`);
              const refLoglikText = await refLoglikResponse.text();
              if (refLoglikText && refLoglikText.trim().length > 0 && !refLoglikText.includes('<!DOCTYPE')) {
                const refLoglikLines = refLoglikText.trim().split('\n').slice(1);
                for (const line of refLoglikLines) {
                  const parts = line.split(',');
                  if (parts.length >= 2) {
                    const step = parseInt(parts[0]);
                    const mse = parseFloat(parts[1]);
                    if (!isNaN(step) && !isNaN(mse)) {
                      refLoglikMseSteps.push(step);
                      refLoglikMseValues.push(mse);
                      // Parse optional loglik_mean column (index 2)
                      if (parts.length >= 3) {
                        const meanLL = parseFloat(parts[2]);
                        if (!isNaN(meanLL)) refLoglikMeanValues.push(meanLL);
                      }
                    }
                  }
                }
              }
            } catch (err) {
              console.warn('Log-likelihood MSE CSV not available:', err);
            }

            refHistory = {
              steps: refSteps,
              train_loss: refTrainLoss,
              test_mse: refTestMse.length > 0 ? refTestMse : null,
              loglik_mse_steps: refLoglikMseSteps,
              loglik_mse_values: refLoglikMseValues,
              loglik_mean_values: refLoglikMeanValues.length > 0 ? refLoglikMeanValues : null,
              training_time: refLastTime,
              hardware: 'T4 GPU (Modal)',
              architecture: '(256, 128, 128, 64) MLP',
            };

            setTrainingHistory(refHistory);
          }
        } catch (err) {
          console.warn('Reference training history not available:', err);
        }

        // Load scatter plot data (true vs estimated log-likelihoods)
        let scatterData = null;
        try {
          const scatterResponse = await fetch(`${import.meta.env.BASE_URL}case5/data/reference_loglik_scatter.csv`);
          const scatterText = await scatterResponse.text();

          if (scatterText && scatterText.trim().length > 0 && !scatterText.includes('<!DOCTYPE')) {
            const scatterLines = scatterText.trim().split('\n').slice(1);
            const trueLLs = [];
            const estLLs = [];

            for (const line of scatterLines) {
              if (line && line.trim().length > 0) {
                const parts = line.split(',');
                if (parts.length >= 5) {
                  const trueLL = parseFloat(parts[3]);
                  const estLL = parseFloat(parts[4]);
                  if (!isNaN(trueLL) && !isNaN(estLL)) {
                    trueLLs.push(trueLL);
                    estLLs.push(estLL);
                  }
                }
              }
            }

            if (trueLLs.length > 0) {
              scatterData = { true_ll: trueLLs, est_ll: estLLs };
            }
          }
        } catch (err) {
          console.warn('Reference scatter data not available:', err);
        }

        // Load generated samples CSV
        let genSamplesData = null;
        try {
          const genResponse = await fetch(`${import.meta.env.BASE_URL}case5/data/reference_generated_samples.csv`);
          const genText = await genResponse.text();

          if (genText && genText.trim().length > 0 && !genText.includes('<!DOCTYPE')) {
            const genLines = genText.trim().split('\n').slice(1);
            const genX1 = [];
            const genX2 = [];
            const genY = [];

            for (const line of genLines) {
              if (line && line.trim().length > 0) {
                const parts = line.split(',');
                if (parts.length >= 3) {
                  const x1 = parseFloat(parts[0]);
                  const x2 = parseFloat(parts[1]);
                  const y = parseFloat(parts[2]);
                  if (!isNaN(x1) && !isNaN(x2) && !isNaN(y)) {
                    genX1.push(x1);
                    genX2.push(x2);
                    genY.push(y);
                  }
                }
              }
            }

            if (genX1.length > 0) {
              genSamplesData = { x1: genX1, x2: genX2, y: genY };
            }
          }
        } catch (err) {
          console.warn('Generated samples data not available:', err);
        }

        setPlotData({
          trainX1, trainX2, trainY,
          testX1, testX2, testY,
          trueLogLik,
          scatterData,
          genSamplesData,
        });

        setLoading(false);
      } catch (err) {
        setError(err.message);
        setLoading(false);
      }
    }

    loadData();
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-white flex items-center justify-center">
        <p className="text-gray-600">Loading data...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-white flex items-center justify-center">
        <p className="text-red-600">Error: {error}</p>
      </div>
    );
  }

  // Build plot for each view
  const renderPlot = () => {
    if (!plotData) return null;

    // Compute shared color range across training, test, and generated data for Y-colored plots
    const allYValues = [...plotData.trainY, ...plotData.testY];
    if (plotData.genSamplesData) {
      allYValues.push(...plotData.genSamplesData.y);
    }
    const yMin = Math.min(...allYValues);
    const yMax = Math.max(...allYValues);
    const sharedColorscale = 'RdBu';

    if (selectedView === 'training') {
      return (
        <Plot
          data={[{
            x: plotData.trainX1,
            y: plotData.trainX2,
            mode: 'markers',
            type: 'scatter',
            marker: {
              color: plotData.trainY,
              colorscale: sharedColorscale,
              cmin: yMin,
              cmax: yMax,
              size: 4,
              opacity: 0.6,
              colorbar: { title: 'Y' },
            },
            name: 'Training data',
          }]}
          layout={{
            title: 'Training Data (5000 points)',
            xaxis: { title: 'X1' },
            yaxis: { title: 'X2' },
            width: 700,
            height: 500,
          }}
        />
      );
    }

    if (selectedView === 'test') {
      return (
        <Plot
          data={[{
            x: plotData.testX1,
            y: plotData.testX2,
            mode: 'markers',
            type: 'scatter',
            marker: {
              color: plotData.testY,
              colorscale: sharedColorscale,
              cmin: yMin,
              cmax: yMax,
              size: 6,
              colorbar: { title: 'Y' },
            },
            name: 'Test data',
          }]}
          layout={{
            title: 'Test Data (500 points)',
            xaxis: { title: 'X1' },
            yaxis: { title: 'X2' },
            width: 700,
            height: 500,
          }}
        />
      );
    }

    if (selectedView === 'generated_samples' && plotData.genSamplesData) {
      return (
        <Plot
          data={[{
            x: plotData.genSamplesData.x1,
            y: plotData.genSamplesData.x2,
            mode: 'markers',
            type: 'scatter',
            marker: {
              color: plotData.genSamplesData.y,
              colorscale: sharedColorscale,
              cmin: yMin,
              cmax: yMax,
              size: 5,
              opacity: 0.6,
              colorbar: { title: 'Y' },
            },
            name: 'Generated samples',
          }]}
          layout={{
            title: `Generated Samples from Flow Model (${plotData.genSamplesData.x1.length} points)`,
            xaxis: { title: 'X1' },
            yaxis: { title: 'X2' },
            width: 700,
            height: 500,
          }}
        />
      );
    }

    if (selectedView === 'loglik_scatter' && plotData.scatterData) {
      const minVal = Math.min(
        Math.min(...plotData.scatterData.true_ll),
        Math.min(...plotData.scatterData.est_ll)
      );
      const maxVal = Math.max(
        Math.max(...plotData.scatterData.true_ll),
        Math.max(...plotData.scatterData.est_ll)
      );

      // Compute MSE
      let mse = 0;
      for (let i = 0; i < plotData.scatterData.true_ll.length; i++) {
        const diff = plotData.scatterData.true_ll[i] - plotData.scatterData.est_ll[i];
        mse += diff * diff;
      }
      mse /= plotData.scatterData.true_ll.length;

      return (
        <Plot
          data={[
            {
              x: plotData.scatterData.true_ll,
              y: plotData.scatterData.est_ll,
              mode: 'markers',
              type: 'scatter',
              marker: { color: 'steelblue', size: 5, opacity: 0.6 },
              name: 'Test points',
            },
            {
              x: [minVal, maxVal],
              y: [minVal, maxVal],
              mode: 'lines',
              type: 'scatter',
              line: { color: 'red', dash: 'dash', width: 2 },
              name: 'y = x (perfect)',
            },
          ]}
          layout={{
            title: `True vs Estimated Log-Likelihoods (MSE = ${mse.toFixed(4)})`,
            xaxis: { title: 'True log p(y|x)' },
            yaxis: { title: 'Estimated log p(y|x)' },
            width: 700,
            height: 600,
            showlegend: true,
          }}
        />
      );
    }

    if (selectedView === 'training_loss' && trainingHistory) {
      const traces = [{
        x: trainingHistory.steps,
        y: trainingHistory.train_loss,
        mode: 'lines',
        name: 'Train Loss',
        line: { color: 'blue' },
      }];

      if (trainingHistory.test_mse) {
        traces.push({
          x: trainingHistory.steps,
          y: trainingHistory.test_mse,
          mode: 'lines',
          name: 'Test MSE',
          line: { color: 'orange' },
        });
      }

      return (
        <Plot
          data={traces}
          layout={{
            title: 'Training Loss (Flow Matching MSE)',
            xaxis: { title: 'Step' },
            yaxis: { title: 'Loss', type: 'log' },
            width: 700,
            height: 500,
          }}
        />
      );
    }

    if (selectedView === 'loglik_mse_curve' && trainingHistory && trainingHistory.loglik_mse_values.length > 0) {
      return (
        <Plot
          data={[{
            x: trainingHistory.loglik_mse_steps,
            y: trainingHistory.loglik_mse_values,
            mode: 'lines+markers',
            name: 'Log-lik MSE',
            line: { color: 'green' },
            marker: { size: 4 },
          }]}
          layout={{
            title: 'Log-Likelihood MSE Over Training',
            xaxis: { title: 'Step' },
            yaxis: { title: 'MSE', type: 'log' },
            width: 700,
            height: 500,
          }}
        />
      );
    }

    if (selectedView === 'avg_loglik' && trainingHistory && trainingHistory.loglik_mean_values && trainingHistory.loglik_mean_values.length > 0) {
      const traces = [{
        x: trainingHistory.loglik_mse_steps,
        y: trainingHistory.loglik_mean_values,
        mode: 'lines+markers',
        name: 'Mean Est. Log-Likelihood',
        line: { color: 'purple' },
        marker: { size: 4 },
      }];

      // Add ground truth mean log-likelihood as a horizontal reference line
      if (plotData && plotData.trueLogLik && plotData.trueLogLik.length > 0) {
        const trueMeanLL = plotData.trueLogLik.reduce((a, b) => a + b, 0) / plotData.trueLogLik.length;
        const steps = trainingHistory.loglik_mse_steps;
        traces.push({
          x: [steps[0], steps[steps.length - 1]],
          y: [trueMeanLL, trueMeanLL],
          mode: 'lines',
          name: `True Mean Log-Lik (${trueMeanLL.toFixed(3)})`,
          line: { color: 'red', dash: 'dash', width: 2 },
        });
      }

      return (
        <Plot
          data={traces}
          layout={{
            title: 'Average Estimated Log-Likelihood on Test Data',
            xaxis: { title: 'Step' },
            yaxis: { title: 'Mean log p(y|x)' },
            width: 700,
            height: 500,
            showlegend: true,
            annotations: [{
              text: 'Higher = model assigns more probability to test data',
              showarrow: false,
              xref: 'paper',
              yref: 'paper',
              x: 0.5,
              y: -0.15,
              font: { size: 12, color: 'gray' },
            }],
          }}
        />
      );
    }

    return <p className="text-gray-500 italic">Data not available for this view. Run the Modal training script to generate results.</p>;
  };

  const viewOptions = [
    { key: 'training', label: 'Training Data' },
    { key: 'test', label: 'Test Data' },
    { key: 'generated_samples', label: 'Generated Samples' },
    { key: 'loglik_scatter', label: 'Log-Likelihood Scatter' },
    { key: 'training_loss', label: 'Training Loss' },
    { key: 'loglik_mse_curve', label: 'Log-Lik MSE Curve' },
    { key: 'avg_loglik', label: 'Avg Log-Likelihood' },
  ];

  return (
    <div className="min-h-screen bg-white">
      <div className="max-w-4xl mx-auto px-6 py-16">
        <Link to="/case5" className="text-blue-600 hover:text-blue-800 mb-8 inline-block">
          ← Back to Case Study 5
        </Link>

        <h1 className="text-4xl font-light text-gray-900 mb-8">
          Case Study 5: Solutions
        </h1>

        {/* True Distribution */}
        <section className="mb-12">
          <h2 className="text-2xl font-medium text-gray-900 mb-4">The True Distribution</h2>
          <div className="prose max-w-none text-gray-700 space-y-4">
            <p>
              The data is generated from the following model:
            </p>
            <BlockMath math="X_1, X_2 \stackrel{\text{iid}}{\sim} \frac{1}{2}\mathcal{N}(-2, 1) + \frac{1}{2}\mathcal{N}(2, 1)" />
            <BlockMath math="Y \mid X_1, X_2 \sim \frac{1}{2}\mathcal{N}(X_1, 1) + \frac{1}{2}\mathcal{N}(X_2, 1)" />

            <p>
              The true log-likelihood of a test point <InlineMath math="(x_1, x_2, y)" /> is:
            </p>
            <BlockMath math="\log p(y \mid x_1, x_2) = \log\left(\frac{1}{2}\phi(y - x_1) + \frac{1}{2}\phi(y - x_2)\right)" />
            <p>
              where <InlineMath math="\phi" /> is the standard normal density.
            </p>
          </div>
        </section>

        {/* Reference Solution */}
        <section className="mb-12">
          <h2 className="text-2xl font-medium text-gray-900 mb-4">Reference Solution: Flow Matching</h2>
          <div className="prose max-w-none text-gray-700 space-y-4">
            <p>
              The reference solution uses <strong>rectified flow matching</strong> to learn the conditional
              distribution <InlineMath math="p(y \mid x_1, x_2)" />, then computes log-likelihoods using
              the <strong>continuous normalizing flow</strong> (CNF) change of variables formula.
            </p>

            <p>
              A neural network <InlineMath math="v_\theta(x_1, x_2, t, z)" /> is trained to approximate
              the velocity field of a flow that transforms noise <InlineMath math="z_0 \sim \mathcal{N}(0,1)" /> into
              conditional samples <InlineMath math="z_1 \sim p(y \mid x_1, x_2)" />.
            </p>

            <h3 className="text-lg font-medium text-gray-900 mt-6">Log-Likelihood via Augmented ODE</h3>
            <p>
              To compute <InlineMath math="\log p(y \mid x_1, x_2)" />, we integrate the augmented ODE
              backwards from <InlineMath math="t=1" /> (data) to <InlineMath math="t=0" /> (noise):
            </p>
            <BlockMath math="\frac{dz}{dt} = v_\theta(x, t, z), \qquad \frac{d\ell}{dt} = -\frac{\partial v_\theta}{\partial z}(x, t, z)" />
            <p>
              Starting from <InlineMath math="z_1 = y" /> and <InlineMath math="\ell_1 = 0" />, we obtain
              the noise point <InlineMath math="z_0" /> and the accumulated log-density change <InlineMath math="\ell_0" />.
              The log-likelihood is then:
            </p>
            <BlockMath math="\log p(y \mid x) = \log \mathcal{N}(z_0; 0, 1) - \ell_0" />
            <p>
              Since <InlineMath math="z" /> is 1-dimensional, the divergence <InlineMath math="\partial v / \partial z" /> is
              computed exactly via automatic differentiation (no trace estimation needed). We use a high-order
              adaptive solver (Dormand-Prince / dopri5) for accuracy.
            </p>
          </div>
        </section>

        {/* Visualization */}
        <section className="mb-12">
          <h2 className="text-2xl font-medium text-gray-900 mb-4">Visualizations</h2>

          <div className="flex flex-wrap gap-2 mb-6">
            {viewOptions.map(opt => (
              <button
                key={opt.key}
                onClick={() => setSelectedView(opt.key)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  selectedView === opt.key
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>

          <div className="bg-gray-50 p-4 rounded-lg flex justify-center">
            {renderPlot()}
          </div>
        </section>

        {/* Training Details */}
        {trainingHistory && (
          <section className="mb-12">
            <h2 className="text-2xl font-medium text-gray-900 mb-4">Training Details</h2>
            <div className="bg-gray-50 p-6 rounded-lg">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div><span className="font-medium">Hardware:</span> {trainingHistory.hardware}</div>
                <div><span className="font-medium">Architecture:</span> {trainingHistory.architecture}</div>
                <div><span className="font-medium">Training time:</span> {(trainingHistory.training_time / 60).toFixed(1)} minutes</div>
                <div><span className="font-medium">Total steps:</span> {trainingHistory.steps[trainingHistory.steps.length - 1]}</div>
              </div>
            </div>
          </section>
        )}
      </div>
    </div>
  );
}
