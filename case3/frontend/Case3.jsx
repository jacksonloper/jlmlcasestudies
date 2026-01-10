import { useState } from 'react';
import { Link } from 'react-router-dom';
import { InlineMath } from 'react-katex';
import npyjs from 'npyjs';

export default function Case3() {
  const [predictionFile, setPredictionFile] = useState(null);
  const [accuracy, setAccuracy] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setPredictionFile(file);
    setAccuracy(null);
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
      
      // Read the uploaded file (should be 1D array of integer predictions)
      const arrayBuffer = await predictionFile.arrayBuffer();
      const predictions = await npy.load(arrayBuffer);
      
      // Fetch the true test_y labels
      const response = await fetch(`${import.meta.env.BASE_URL}case3/data/test_y.npy`);
      const trueArrayBuffer = await response.arrayBuffer();
      const trueData = await npy.load(trueArrayBuffer);

      const predictedY = predictions.data;
      const trueY = trueData.data;

      // Validate shape: must be 1D with n_test elements
      if (predictions.shape.length !== 1) {
        throw new Error(`Expected a 1D array of predictions, got ${predictions.shape.length}D array`);
      }
      
      const nTestPoints = predictions.shape[0];
      
      if (nTestPoints !== trueY.length) {
        throw new Error(`Expected ${trueY.length} predictions, got ${nTestPoints}`);
      }

      // Calculate accuracy
      let correctPredictions = 0;
      
      for (let i = 0; i < nTestPoints; i++) {
        const predicted = Math.round(predictedY[i]);
        if (predicted === trueY[i]) {
          correctPredictions++;
        }
      }

      const acc = correctPredictions / nTestPoints;
      setAccuracy(acc);
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
          Case Study 3: Modular Arithmetic
        </h1>

        <section className="mb-12">
          <h2 className="text-2xl font-medium text-gray-900 mb-4">The Task</h2>
          <div className="prose max-w-none text-gray-700 space-y-4">
            <p>
              Learn to perform <strong>modular addition mod 97</strong>. Given two numbers{' '}
              <InlineMath math="a, b \in \{0, 1, \ldots, 96\}" />, predict{' '}
              <InlineMath math="(a + b) \mod 97" />.
            </p>

            <p>
              The input is encoded as a <strong>one-hot vector of length 194</strong> (97 × 2),
              where exactly two bits are set: one for <InlineMath math="a" /> (positions 0-96) 
              and one for <InlineMath math="b" /> (positions 97-193).
            </p>

            <p>
              The training and test sets each contain approximately half of all{' '}
              <InlineMath math="97^2 = 9409" /> possible pairs, split randomly.
            </p>

            <p>
              Your predictions will be evaluated using <strong>accuracy</strong>:
              the percentage of test examples where your predicted class matches
              the true answer.
            </p>

            <p className="font-medium text-blue-700">
              🎯 Goal: Try to achieve at least 95% accuracy!
            </p>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-medium text-gray-900 mb-4">Download Data</h2>
          <div className="space-y-3">
            <a
              href={`${import.meta.env.BASE_URL}case3/data/train_x.npy`}
              download="train_x.npy"
              className="inline-block bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors mr-4"
            >
              Download Training X
            </a>
            <a
              href={`${import.meta.env.BASE_URL}case3/data/train_y.npy`}
              download="train_y.npy"
              className="inline-block bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors"
            >
              Download Training Y
            </a>
            <p className="text-sm text-gray-600">
              Training X: ~4705 × 194 one-hot encoded inputs (float32)
              <br />
              Training Y: ~4705 labels in &#123;0, ..., 96&#125; (int32)
            </p>
          </div>

          <div className="space-y-3 mt-6">
            <a
              href={`${import.meta.env.BASE_URL}case3/data/test_x.npy`}
              download="test_x.npy"
              className="inline-block bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors"
            >
              Download Test X
            </a>
            <p className="text-sm text-gray-600">
              Test X: ~4704 × 194 one-hot encoded inputs (float32)
            </p>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-medium text-gray-900 mb-4">Submit Your Predictions</h2>
          <div className="bg-gray-50 p-6 rounded-lg">
            <p className="text-gray-700 mb-4">
              Upload your predictions as a .npy file containing a{' '}
              <strong>1D array of ~4704 integers</strong> (one prediction per test point,
              each value in &#123;0, ..., 96&#125; representing your best guess for{' '}
              <InlineMath math="(a + b) \mod 97" />).
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

            {accuracy !== null && (
              <div className={`mt-4 p-4 border rounded-lg ${accuracy >= 0.95 ? 'bg-green-50 border-green-200' : 'bg-yellow-50 border-yellow-200'}`}>
                <h3 className={`font-medium mb-2 ${accuracy >= 0.95 ? 'text-green-900' : 'text-yellow-900'}`}>Your Score:</h3>
                <p className={`text-2xl font-bold ${accuracy >= 0.95 ? 'text-green-700' : 'text-yellow-700'}`}>
                  Accuracy = {(accuracy * 100).toFixed(2)}%
                </p>
                <p className={`text-sm mt-2 ${accuracy >= 0.95 ? 'text-green-700' : 'text-yellow-700'}`}>
                  {accuracy >= 0.95 
                    ? '🎉 Great job! You achieved at least 95% accuracy!' 
                    : '🎯 Try to get at least 95% accuracy!'}
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
              View the solutions page to learn more about the modular arithmetic task
              and approaches to solve it.
            </p>
            <Link
              to="/case3/solutions"
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
