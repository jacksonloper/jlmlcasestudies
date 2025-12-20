import { useState } from 'react';
import { Link } from 'react-router-dom';
import { InlineMath, BlockMath } from 'react-katex';
import npyjs from 'npyjs';

export default function Case2() {
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
      
      // Read the uploaded file (should be 100x2 matrix)
      const arrayBuffer = await predictionFile.arrayBuffer();
      const predictions = await npy.load(arrayBuffer);
      
      // Fetch the true test_y values
      const response = await fetch('/case2/data/test_y.npy');
      const trueArrayBuffer = await response.arrayBuffer();
      const trueData = await npy.load(trueArrayBuffer);

      const predictedSamples = predictions.data;
      const trueY = trueData.data;

      // Validate shape
      if (predictions.shape.length !== 2 || predictions.shape[0] !== trueY.length || predictions.shape[1] !== 2) {
        throw new Error(`Expected a ${trueY.length}×2 matrix (two samples per test point), got shape [${predictions.shape.join(', ')}]`);
      }

      // Calculate 2-sample energy score
      // ES = E[|Y - X1|] + E[|Y - X2|] - 0.5 * (E[|X1 - X2|])
      // where X1 and X2 are the two samples, Y is the true value
      
      let sumDist1 = 0; // sum of |Y - X1|
      let sumDist2 = 0; // sum of |Y - X2|
      let sumDistSamples = 0; // sum of |X1 - X2|
      
      for (let i = 0; i < trueY.length; i++) {
        const sample1 = predictedSamples[i * 2];
        const sample2 = predictedSamples[i * 2 + 1];
        const truth = trueY[i];
        
        sumDist1 += Math.abs(truth - sample1);
        sumDist2 += Math.abs(truth - sample2);
        sumDistSamples += Math.abs(sample1 - sample2);
      }

      const energyScore = (sumDist1 + sumDist2) / (2 * trueY.length) - 0.5 * sumDistSamples / trueY.length;

      setScore(energyScore);
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
              a single value, you must produce <strong>two samples</strong> from the conditional
              distribution of <InlineMath math="y" /> given <InlineMath math="x" />.
            </p>

            <p>
              The training set contains 900 examples, and the test set contains 100 examples.
              For each test point, you should provide two samples from the distribution.
            </p>

            <p>
              Your predictions will be evaluated using the <strong>2-sample Energy Score</strong>,
              which measures how well your samples represent the true distribution. The energy score is:
            </p>

            <BlockMath math="\text{ES} = \frac{1}{2}\mathbb{E}[|Y - X_1|] + \frac{1}{2}\mathbb{E}[|Y - X_2|] - \frac{1}{2}\mathbb{E}[|X_1 - X_2|]" />

            <p>
              where <InlineMath math="X_1" /> and <InlineMath math="X_2" /> are your two samples,
              and <InlineMath math="Y" /> is the true value. Lower scores are better.
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
              Upload your predictions as a .npy file containing a 100×2 matrix where each row
              contains two samples for the corresponding test point.
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
              <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg">
                <h3 className="font-medium text-green-900 mb-2">Your Score:</h3>
                <p className="text-2xl font-bold text-green-700">
                  Energy Score = {score.toFixed(4)}
                </p>
                <p className="text-sm text-green-700 mt-2">
                  Lower is better! The energy score measures how well your samples represent the true distribution.
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
