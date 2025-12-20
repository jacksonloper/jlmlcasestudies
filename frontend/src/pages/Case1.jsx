import { useState } from 'react';
import { Link } from 'react-router-dom';
import { InlineMath, BlockMath } from 'react-katex';
import npyjs from 'npyjs';

export default function Case1() {
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
      
      // Read the uploaded file
      const arrayBuffer = await predictionFile.arrayBuffer();
      const predictions = await npy.load(arrayBuffer);
      
      // Fetch the true test_y values
      const response = await fetch('/case1/data/test_y.npy');
      const trueArrayBuffer = await response.arrayBuffer();
      const trueData = await npy.load(trueArrayBuffer);

      const predictedY = predictions.data;
      const trueY = trueData.data;

      // Calculate MSE
      if (predictedY.length !== trueY.length) {
        throw new Error(`Expected ${trueY.length} predictions, got ${predictedY.length}`);
      }

      const mse = predictedY.reduce((sum, pred, i) => {
        return sum + Math.pow(pred - trueY[i], 2);
      }, 0) / predictedY.length;

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
          Case Study 1: Conditional Distribution Prediction
        </h1>

        <section className="mb-12">
          <h2 className="text-2xl font-medium text-gray-900 mb-4">The Task</h2>
          <div className="prose max-w-none text-gray-700 space-y-4">
            <p>
              In this challenge, you are given training data where you can observe both{' '}
              <InlineMath math="x" /> and <InlineMath math="y" />. Your goal is to predict{' '}
              <InlineMath math="y" /> for the test set where only <InlineMath math="x" /> is given.
            </p>
            
            <div className="bg-gray-50 p-6 rounded-lg my-6">
              <h3 className="font-medium text-gray-900 mb-3">Data Generation Process:</h3>
              <BlockMath math="x \sim N(4, 1)" />
              <div className="my-2">
                <InlineMath math="y \mid x" /> is an equal parts mixture:
              </div>
              <BlockMath math="y \mid x \sim \frac{1}{2} N(x^2, 1) + \frac{1}{2} N(0, 1)" />
            </div>

            <p>
              The training set contains 900 examples, and the test set contains 100 examples.
              Your predictions will be evaluated using Mean Squared Error (MSE).
            </p>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-medium text-gray-900 mb-4">Download Data</h2>
          <div className="space-y-3">
            <a
              href="/case1/data/train.npy"
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
              href="/case1/data/test_x.npy"
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
              Upload your predictions as a .npy file containing 100 predicted y values (float16, float32, or float64).
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
                  MSE = {score.toFixed(4)}
                </p>
                <p className="text-sm text-green-700 mt-2">
                  Lower is better! The MSE measures the average squared difference between your predictions and the true values.
                </p>
              </div>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}
