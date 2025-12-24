import { useState } from 'react';
import { Link } from 'react-router-dom';
import { InlineMath, BlockMath } from 'react-katex';
import npyjs from 'npyjs';

/**
 * Compute mean pairwise absolute difference using O(n log n) algorithm.
 * For sorted samples x_{(0)} <= ... <= x_{(n-1)}, each x_{(k)} appears as
 * the larger element in k pairs and as the smaller element in (n-1-k) pairs.
 * Coefficient for x_{(k)} = k - (n-1-k) = 2k - n + 1.
 * The formula is shift-invariant because weights sum to zero.
 */
function meanPairwiseAbsDiff(samples) {
  const n = samples.length;
  if (n < 2) return 0;
  
  // Sort samples in-place (samples array is freshly created for each test point)
  samples.sort((a, b) => a - b);
  
  // Compute weighted sum: sum_k (2k - n + 1) * x_{(k)}
  let weightedSum = 0;
  for (let k = 0; k < n; k++) {
    const weight = 2 * k - n + 1;
    weightedSum += weight * samples[k];
  }
  
  // Divide by number of pairs to get mean
  const nPairs = (n * (n - 1)) / 2;
  return weightedSum / nPairs;
}

/**
 * Compute energy score for a single test point.
 * ES = E[|Y - X_j|] - 0.5 * E[|X_j - X_{j'}|]
 * where X_j are samples and Y is the true value.
 */
function computeEnergyScoreForPoint(samples, trueValue) {
  const n = samples.length;
  
  // Term 1: mean |Y - X_j|
  let sumDist = 0;
  for (let j = 0; j < n; j++) {
    sumDist += Math.abs(trueValue - samples[j]);
  }
  const term1 = sumDist / n;
  
  // Term 2: mean |X_j - X_{j'}| using O(n log n) algorithm
  const term2 = meanPairwiseAbsDiff(samples);
  
  return term1 - 0.5 * term2;
}

export default function Case2() {
  const [predictionFile, setPredictionFile] = useState(null);
  const [score, setScore] = useState(null);
  const [numSamples, setNumSamples] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setPredictionFile(file);
    setScore(null);
    setNumSamples(null);
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
      
      // Read the uploaded file (should be 100×n matrix where n is between 5 and 100)
      const arrayBuffer = await predictionFile.arrayBuffer();
      const predictions = await npy.load(arrayBuffer);
      
      // Fetch the true test_y values
      const response = await fetch('/case2/data/test_y.npy');
      const trueArrayBuffer = await response.arrayBuffer();
      const trueData = await npy.load(trueArrayBuffer);

      const predictedSamples = predictions.data;
      const trueY = trueData.data;

      // Validate shape: must be 2D with 100 rows and 5-100 columns
      if (predictions.shape.length !== 2) {
        throw new Error(`Expected a 2D matrix, got ${predictions.shape.length}D array`);
      }
      
      const nTestPoints = predictions.shape[0];
      const nSamplesPerPoint = predictions.shape[1];
      
      if (nTestPoints !== trueY.length) {
        throw new Error(`Expected ${trueY.length} test points, got ${nTestPoints}`);
      }
      
      if (nSamplesPerPoint < 5 || nSamplesPerPoint > 100) {
        throw new Error(`Expected between 5 and 100 samples per test point, got ${nSamplesPerPoint}`);
      }

      // Calculate energy score using efficient O(n log n) algorithm
      let totalEnergyScore = 0;
      
      for (let i = 0; i < nTestPoints; i++) {
        // Extract samples for this test point
        const samples = [];
        for (let j = 0; j < nSamplesPerPoint; j++) {
          samples.push(predictedSamples[i * nSamplesPerPoint + j]);
        }
        
        // Compute energy score for this point
        totalEnergyScore += computeEnergyScoreForPoint(samples, trueY[i]);
      }

      const avgEnergyScore = totalEnergyScore / nTestPoints;

      setScore(avgEnergyScore);
      setNumSamples(nSamplesPerPoint);
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
          Case Study 2: Distribution Sampling
        </h1>

        <section className="mb-12">
          <h2 className="text-2xl font-medium text-gray-900 mb-4">The Task</h2>
          <div className="prose max-w-none text-gray-700 space-y-4">
            <p>
              This case study uses the same dataset as Case Study 1, but instead of predicting
              a single value, you must produce <strong>multiple samples</strong> from the conditional
              distribution of <InlineMath math="y" /> given <InlineMath math="x" />.
            </p>

            <p>
              The training set contains 900 examples, and the test set contains 100 examples.
              For each test point, you should provide between <strong>5 and 100 samples</strong> from the distribution.
            </p>

            <p>
              Your predictions will be evaluated using the <strong>Energy Score</strong>,
              which measures how well your samples represent the true distribution:
            </p>

            <BlockMath math="\text{ES} = \mathbb{E}[|Y - X_j|] - \frac{1}{2}\mathbb{E}[|X_j - X_{j'}|]" />

            <p>
              where <InlineMath math="X_j" /> are your samples (j = 1, ..., n),
              and <InlineMath math="Y" /> is the true value. The first term measures accuracy
              (how close samples are to the truth), and the second term rewards diversity
              (how spread out your samples are). Lower scores are better.
            </p>

            <p className="font-medium text-blue-700">
              🎯 Goal: Try to achieve an energy score less than 2.1!
            </p>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-medium text-gray-900 mb-4">Download Data</h2>
          <div className="space-y-3">
            <a
              href="/case2/data/train.npy"
              download="train.npy"
              className="inline-block bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors"
            >
              Download Training Data (900×2 matrix)
            </a>
            <p className="text-sm text-gray-600">
              Contains 900 rows with [x, y] pairs stored as float16
            </p>
          </div>

          <div className="space-y-3 mt-6">
            <a
              href="/case2/data/test_x.npy"
              download="test_x.npy"
              className="inline-block bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors"
            >
              Download Test X (100 vector)
            </a>
            <p className="text-sm text-gray-600">
              Contains 100 x values stored as float16
            </p>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-medium text-gray-900 mb-4">Submit Your Predictions</h2>
          <div className="bg-gray-50 p-6 rounded-lg">
            <p className="text-gray-700 mb-4">
              Upload your predictions as a .npy file containing a 100×n matrix where each row
              contains between 5 and 100 samples for the corresponding test point.
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
              <div className={`mt-4 p-4 border rounded-lg ${score < 2.1 ? 'bg-green-50 border-green-200' : 'bg-yellow-50 border-yellow-200'}`}>
                <h3 className={`font-medium mb-2 ${score < 2.1 ? 'text-green-900' : 'text-yellow-900'}`}>Your Score:</h3>
                <p className={`text-2xl font-bold ${score < 2.1 ? 'text-green-700' : 'text-yellow-700'}`}>
                  Energy Score = {score.toFixed(4)}
                </p>
                <p className={`text-sm mt-2 ${score < 2.1 ? 'text-green-700' : 'text-yellow-700'}`}>
                  {numSamples && `(using ${numSamples} samples per test point)`}
                </p>
                <p className={`text-sm mt-1 ${score < 2.1 ? 'text-green-700' : 'text-yellow-700'}`}>
                  {score < 2.1 
                    ? '🎉 Great job! You beat the target of 2.1!' 
                    : '🎯 Try to get below 2.1! Lower is better.'}
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
              View the solutions page to see the true data generation process and understand
              the optimal sampling strategy.
            </p>
            <Link
              to="/case2/solutions"
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
