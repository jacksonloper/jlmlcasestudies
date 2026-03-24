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
                const valFlowLoss = parseFloat(parts[2]);
                const time = parseFloat(parts[3]);
                if (!isNaN(step) && !isNaN(loss)) {
                  refSteps.push(step);
                  refTrainLoss.push(loss);
                  if (!isNaN(valFlowLoss)) refTestMse.push(valFlowLoss);
                  if (!isNaN(time)) refLastTime = time;
                }
              }
            }

            // Load log-likelihood MSE CSV (may include loglik_mean column)
            let refLoglikMseSteps = [];
            let refLoglikMseValues = [];
            let refLoglikMeanValues = [];
            let refBestStep = null;
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
                      // Parse optional is_best column (index 4)
                      if (parts.length >= 5) {
                        const isBest = parseInt(parts[4]);
                        if (isBest === 1) refBestStep = step;
                      }
                    }
                  }
                }
                // If no is_best column, find the step with minimum MSE
                if (refBestStep === null && refLoglikMseValues.length > 0) {
                  const minIdx = refLoglikMseValues.indexOf(Math.min(...refLoglikMseValues));
                  refBestStep = refLoglikMseSteps[minIdx];
                }
              }
            } catch (err) {
              console.warn('Log-likelihood MSE CSV not available:', err);
            }

            refHistory = {
              steps: refSteps,
              train_loss: refTrainLoss,
              val_flow_loss: refTestMse.length > 0 ? refTestMse : null,
              loglik_mse_steps: refLoglikMseSteps,
              loglik_mse_values: refLoglikMseValues,
              loglik_mean_values: refLoglikMeanValues.length > 0 ? refLoglikMeanValues : null,
              best_step: refBestStep,
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

  // Captions for each plot view
  const plotCaptions = {
    training: 'Each dot is one training point. Color indicates the Y value. The covariates X1 and X2 each follow a bimodal mixture distribution.',
    test: 'Held-out test points used only for final evaluation (not for model selection). Color indicates Y.',
    generated_samples: 'Samples drawn from the learned flow model by integrating the ODE forward from noise to data. These should resemble the training distribution if the model has learned well.',
    loglik_scatter: 'Each dot is one test point. The x-axis is the true log-likelihood (known from the data-generating process), and the y-axis is the model\'s estimate. Points near the diagonal indicate accurate estimates.',
    training_loss: 'The flow matching loss measures how well the neural network predicts the velocity field that transforms noise into data. Each epoch is one full pass over all training data with fresh random noise and interpolation times. The red dashed line marks the epoch of the model we selected.',
    loglik_mse_curve: 'At each epoch, we compute log-likelihood estimates for every validation point and compare them to the true log-likelihoods. This plot shows the mean squared error of those estimates over training. The model checkpoint with the lowest validation error (red dashed line) is selected as our final model.',
    avg_loglik: 'The average of the model\'s estimated log-likelihoods across all validation points, plotted over training. The red dashed horizontal line shows the true average log-likelihood. A well-calibrated model should approach the true value. The vertical red dashed line marks the selected model.',
  };

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
              colorbar: { title: 'Y value' },
            },
            name: 'Training data',
          }]}
          layout={{
            title: 'Training Data (5000 points)',
            xaxis: { title: 'X₁ (covariate 1)' },
            yaxis: { title: 'X₂ (covariate 2)' },
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
              colorbar: { title: 'Y value' },
            },
            name: 'Test data',
          }]}
          layout={{
            title: 'Test Data (500 points)',
            xaxis: { title: 'X₁ (covariate 1)' },
            yaxis: { title: 'X₂ (covariate 2)' },
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
              colorbar: { title: 'Y value' },
            },
            name: 'Generated samples',
          }]}
          layout={{
            title: `Samples Generated by the Flow Model (${plotData.genSamplesData.x1.length} points)`,
            xaxis: { title: 'X₁ (covariate 1)' },
            yaxis: { title: 'X₂ (covariate 2)' },
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
              name: 'Perfect estimates (y = x)',
            },
          ]}
          layout={{
            title: `True vs Estimated Log-Likelihoods (test set, MSE = ${mse.toFixed(4)})`,
            xaxis: { title: 'True log p(y | x₁, x₂)' },
            yaxis: { title: 'Estimated log p(y | x₁, x₂)' },
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
        name: 'Training set',
        line: { color: 'blue' },
      }];

      if (trainingHistory.val_flow_loss) {
        traces.push({
          x: trainingHistory.steps,
          y: trainingHistory.val_flow_loss,
          mode: 'lines',
          name: 'Validation set',
          line: { color: 'orange' },
        });
      }

      const shapes = [];
      const annotations = [];
      if (trainingHistory.best_step) {
        shapes.push({
          type: 'line',
          x0: trainingHistory.best_step,
          x1: trainingHistory.best_step,
          yref: 'paper',
          y0: 0,
          y1: 1,
          line: { color: 'red', width: 2, dash: 'dash' },
        });
        annotations.push({
          x: trainingHistory.best_step,
          yref: 'paper',
          y: 1.05,
          text: `Selected model (epoch ${trainingHistory.best_step})`,
          showarrow: false,
          font: { size: 11, color: 'red' },
        });
      }

      return (
        <Plot
          data={traces}
          layout={{
            title: 'Flow Matching Loss During Training',
            xaxis: { title: 'Epoch (each epoch = one full pass over training data)' },
            yaxis: { title: 'Flow matching loss (MSE of predicted velocity)' },
            width: 700,
            height: 500,
            shapes,
            annotations,
          }}
        />
      );
    }

    if (selectedView === 'loglik_mse_curve' && trainingHistory && trainingHistory.loglik_mse_values.length > 0) {
      const shapes = [];
      const annotations = [];
      if (trainingHistory.best_step) {
        shapes.push({
          type: 'line',
          x0: trainingHistory.best_step,
          x1: trainingHistory.best_step,
          yref: 'paper',
          y0: 0,
          y1: 1,
          line: { color: 'red', width: 2, dash: 'dash' },
        });
        annotations.push({
          x: trainingHistory.best_step,
          yref: 'paper',
          y: 1.05,
          text: `Selected model (epoch ${trainingHistory.best_step})`,
          showarrow: false,
          font: { size: 11, color: 'red' },
        });
      }

      return (
        <Plot
          data={[{
            x: trainingHistory.loglik_mse_steps,
            y: trainingHistory.loglik_mse_values,
            mode: 'lines+markers',
            name: 'MSE of log-likelihood estimates',
            line: { color: 'green' },
            marker: { size: 6 },
          }]}
          layout={{
            title: 'How Accurate Are the Log-Likelihood Estimates? (validation set)',
            xaxis: { title: 'Epoch (each epoch = one full pass over training data)' },
            yaxis: { title: 'Mean squared error: (estimated − true log-lik)²' },
            width: 700,
            height: 500,
            shapes,
            annotations,
          }}
        />
      );
    }

    if (selectedView === 'avg_loglik' && trainingHistory && trainingHistory.loglik_mean_values && trainingHistory.loglik_mean_values.length > 0) {
      const traces = [{
        x: trainingHistory.loglik_mse_steps,
        y: trainingHistory.loglik_mean_values,
        mode: 'lines+markers',
        name: 'Model estimate (validation avg)',
        line: { color: 'purple' },
        marker: { size: 6 },
      }];

      // Add ground truth mean log-likelihood as a horizontal reference line
      if (plotData && plotData.trueLogLik && plotData.trueLogLik.length > 0) {
        const trueMeanLL = plotData.trueLogLik.reduce((a, b) => a + b, 0) / plotData.trueLogLik.length;
        const steps = trainingHistory.loglik_mse_steps;
        traces.push({
          x: [steps[0], steps[steps.length - 1]],
          y: [trueMeanLL, trueMeanLL],
          mode: 'lines',
          name: `True average log-likelihood (${trueMeanLL.toFixed(3)})`,
          line: { color: 'red', dash: 'dash', width: 2 },
        });
      }

      const shapes = [];
      const plotAnnotations = [];
      if (trainingHistory.best_step) {
        shapes.push({
          type: 'line',
          x0: trainingHistory.best_step,
          x1: trainingHistory.best_step,
          yref: 'paper',
          y0: 0,
          y1: 1,
          line: { color: 'red', width: 2, dash: 'dash' },
        });
        plotAnnotations.push({
          x: trainingHistory.best_step,
          yref: 'paper',
          y: 1.05,
          text: `Selected model (epoch ${trainingHistory.best_step})`,
          showarrow: false,
          font: { size: 11, color: 'red' },
        });
      }

      return (
        <Plot
          data={traces}
          layout={{
            title: 'Average Estimated Log-Likelihood Over Training (validation set)',
            xaxis: { title: 'Epoch (each epoch = one full pass over training data)' },
            yaxis: { title: 'Average estimated log p(y | x₁, x₂)' },
            width: 700,
            height: 500,
            showlegend: true,
            shapes,
            annotations: plotAnnotations,
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
    { key: 'loglik_scatter', label: 'Log-Lik Scatter' },
    { key: 'training_loss', label: 'Flow Matching Loss' },
    { key: 'loglik_mse_curve', label: 'Log-Lik Accuracy' },
    { key: 'avg_loglik', label: 'Avg Estimated Log-Lik' },
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

          <div className="bg-gray-50 p-4 rounded-lg flex flex-col items-center">
            {renderPlot()}
            {plotCaptions[selectedView] && (
              <p className="text-sm text-gray-600 mt-3 max-w-xl text-center italic">
                {plotCaptions[selectedView]}
              </p>
            )}
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
                <div><span className="font-medium">Total epochs:</span> {trainingHistory.steps[trainingHistory.steps.length - 1]}</div>
                {trainingHistory.best_step && (
                  <div><span className="font-medium">Selected model epoch:</span> {trainingHistory.best_step} (lowest log-likelihood error on validation set)</div>
                )}
              </div>
            </div>
          </section>
        )}
      </div>
    </div>
  );
}
