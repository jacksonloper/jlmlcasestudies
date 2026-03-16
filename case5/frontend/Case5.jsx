import { useState } from 'react';
import { Link } from 'react-router-dom';
import { InlineMath, BlockMath } from 'react-katex';
import npyjs from 'npyjs';

export default function Case5() {
  const [predictionFile, setPredictionFile] = useState(null);
  const [score, setScore] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setPredictionFile(file);
    setScore(null);
    setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!predictionFile) {
      setError('Please select a file');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const npy = new npyjs();

      // Read the uploaded file (should be a 500-element vector of log-likelihoods)
      const arrayBuffer = await predictionFile.arrayBuffer();
      const predictions = await npy.load(arrayBuffer);

      // Fetch the true log-likelihoods
      const response = await fetch(`${import.meta.env.BASE_URL}case5/data/test_true_loglik.npy`);
      const trueArrayBuffer = await response.arrayBuffer();
      const trueData = await npy.load(trueArrayBuffer);

      const predictedLogLik = predictions.data;
      const trueLogLik = trueData.data;

      // Validate shape: must be 1D with 500 elements
      if (predictions.shape.length !== 1) {
        throw new Error(`Expected a 1D vector, got ${predictions.shape.length}D array`);
      }

      if (predictedLogLik.length !== trueLogLik.length) {
        throw new Error(`Expected ${trueLogLik.length} values, got ${predictedLogLik.length}`);
      }

      // Check for NaN/Inf values
      let hasInvalid = false;
      for (let i = 0; i < predictedLogLik.length; i++) {
        if (!isFinite(predictedLogLik[i])) {
          hasInvalid = true;
          break;
        }
      }
      if (hasInvalid) {
        throw new Error('Predictions contain NaN or Infinity values');
      }

      // Calculate MSE of log-likelihoods
      let sumSquaredError = 0;
      for (let i = 0; i < predictedLogLik.length; i++) {
        const diff = predictedLogLik[i] - trueLogLik[i];
        sumSquaredError += diff * diff;
      }
      const mse = sumSquaredError / predictedLogLik.length;

      setScore(mse);
    } catch (err) {
      setError(`Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-white">
      <div className="max-w-4xl mx-auto px-6 py-16">
        <Link to="/" className="text-blue-600 hover:text-blue-800 mb-8 inline-block">
          ← Back to all case studies
        </Link>

        <h1 className="text-4xl font-light text-gray-900 mb-8">
          Case Study 5: Likelihood Estimation
        </h1>

        <section className="mb-12">
          <h2 className="text-2xl font-medium text-gray-900 mb-4">The Task</h2>
          <div className="prose max-w-none text-gray-700 space-y-4">
            <p>
              Given training data from a conditional distribution, estimate the <strong>log-likelihood</strong> of
              held-out test points.
            </p>

            <p>
              The data has inputs <InlineMath math="X = (X_1, X_2) \in \mathbb{R}^2" /> and
              output <InlineMath math="Y \in \mathbb{R}" />. The training set contains 5000
              examples <InlineMath math="(X_1, X_2, Y)" />, and the test set contains 500 examples.
            </p>

            <p>
              For each test point, you must estimate:
            </p>

            <BlockMath math="\log p(Y \mid X_1, X_2)" />

            <p>
              Your estimates will be evaluated using <strong>Mean Squared Error (MSE)</strong> against
              the true log-likelihoods:
            </p>

            <BlockMath math="\text{MSE} = \frac{1}{n}\sum_{i=1}^{n} \left(\hat{\ell}_i - \ell_i\right)^2" />

            <p>
              where <InlineMath math="\hat{\ell}_i" /> is your estimated log-likelihood
              and <InlineMath math="\ell_i" /> is the true log-likelihood for test point <InlineMath math="i" />.
            </p>

            <p className="font-medium text-blue-700">
              🎯 Goal: Try to achieve an MSE less than 0.1!
            </p>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-medium text-gray-900 mb-4">Download Data</h2>
          <div className="space-y-3">
            <a
              href={`${import.meta.env.BASE_URL}case5/data/train_x.npy`}
              download="train_x.npy"
              className="inline-block bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors"
            >
              Download Training X (5000×2 matrix)
            </a>
            <p className="text-sm text-gray-600">
              Contains 5000 rows with [X1, X2] pairs stored as float32
            </p>
          </div>

          <div className="space-y-3 mt-6">
            <a
              href={`${import.meta.env.BASE_URL}case5/data/train_y.npy`}
              download="train_y.npy"
              className="inline-block bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors"
            >
              Download Training Y (5000 vector)
            </a>
            <p className="text-sm text-gray-600">
              Contains 5000 Y values stored as float32
            </p>
          </div>

          <div className="space-y-3 mt-6">
            <a
              href={`${import.meta.env.BASE_URL}case5/data/test_x.npy`}
              download="test_x.npy"
              className="inline-block bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors"
            >
              Download Test X (500×2 matrix)
            </a>
            <p className="text-sm text-gray-600">
              Contains 500 rows with [X1, X2] pairs stored as float32
            </p>
          </div>

          <div className="space-y-3 mt-6">
            <a
              href={`${import.meta.env.BASE_URL}case5/data/test_y.npy`}
              download="test_y.npy"
              className="inline-block bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors"
            >
              Download Test Y (500 vector)
            </a>
            <p className="text-sm text-gray-600">
              Contains 500 Y values stored as float32
            </p>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-medium text-gray-900 mb-4">Submit Your Predictions</h2>
          <div className="bg-gray-50 p-6 rounded-lg">
            <p className="text-gray-700 mb-4">
              Upload your predictions as a .npy file containing a 500-element vector of
              estimated log-likelihoods (one per test point).
            </p>

            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <input
                  type="file"
                  accept=".npy"
                  onChange={handleFileChange}
                  className="block w-full text-sm text-gray-600
                    file:mr-4 file:py-2 file:px-4
                    file:rounded-lg file:border-0
                    file:text-sm file:font-medium
                    file:bg-blue-50 file:text-blue-700
                    hover:file:bg-blue-100"
                />
              </div>

              <button
                type="submit"
                disabled={!predictionFile || loading}
                className="bg-green-600 text-white px-6 py-3 rounded-lg hover:bg-green-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed"
              >
                {loading ? 'Calculating...' : 'Calculate Score'}
              </button>
            </form>

            {error && (
              <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg text-red-700">
                {error}
              </div>
            )}

            {score !== null && (
              <div className={`mt-4 p-4 border rounded-lg ${score < 0.1 ? 'bg-green-50 border-green-200' : 'bg-yellow-50 border-yellow-200'}`}>
                <h3 className={`font-medium mb-2 ${score < 0.1 ? 'text-green-900' : 'text-yellow-900'}`}>Your Score:</h3>
                <p className={`text-2xl font-bold ${score < 0.1 ? 'text-green-700' : 'text-yellow-700'}`}>
                  Log-Likelihood MSE = {score.toFixed(4)}
                </p>
                <p className={`text-sm mt-2 ${score < 0.1 ? 'text-green-700' : 'text-yellow-700'}`}>
                  {score < 0.1
                    ? '🎉 Great job! You beat the target of 0.1!'
                    : '🎯 Try to get below 0.1! Lower is better.'}
                </p>
              </div>
            )}
          </div>
        </section>

        <section className="mb-12 border-t pt-8">
          <div className="bg-gray-50 p-6 rounded-lg">
            <h2 className="text-xl font-medium text-gray-900 mb-3">
              Want to see the solution?
            </h2>
            <p className="text-gray-700 mb-4">
              View the solutions page to see the true data generation process, the reference
              solution using flow matching, and a comparison of true vs estimated log-likelihoods.
            </p>
            <Link
              to="/case5/solutions"
              className="inline-block bg-gray-700 text-white px-6 py-3 rounded-lg hover:bg-gray-800 transition-colors"
            >
              View Solutions →
            </Link>
          </div>
        </section>
      </div>
    </div>
  );
}
