import { useState } from 'react';
import { Link } from 'react-router-dom';
import { InlineMath, BlockMath } from 'react-katex';
import npyjs from 'npyjs';

/**
 * Compute softmax of logits
 */
function softmax(logits) {
  const maxLogit = Math.max(...logits);
  const expLogits = logits.map(l => Math.exp(l - maxLogit));
  const sumExp = expLogits.reduce((a, b) => a + b, 0);
  return expLogits.map(e => e / sumExp);
}

/**
 * Compute cross-entropy loss for a single prediction
 * CE = -log(p[true_class])
 */
function crossEntropyLoss(logits, trueClass) {
  const probs = softmax(logits);
  // Clip probability to avoid log(0)
  const prob = Math.max(probs[trueClass], 1e-10);
  return -Math.log(prob);
}

export default function Case3() {
  const [predictionFile, setPredictionFile] = useState(null);
  const [score, setScore] = useState(null);
  const [accuracy, setAccuracy] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setPredictionFile(file);
    setScore(null);
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
      
      // Read the uploaded file (should be n_test × 97 matrix of logits)
      const arrayBuffer = await predictionFile.arrayBuffer();
      const predictions = await npy.load(arrayBuffer);
      
      // Fetch the true test_y labels
      const response = await fetch(`${import.meta.env.BASE_URL}case3/data/test_y.npy`);
      const trueArrayBuffer = await response.arrayBuffer();
      const trueData = await npy.load(trueArrayBuffer);

      const predictedLogits = predictions.data;
      const trueY = trueData.data;

      // Validate shape: must be 2D with n_test rows and 97 columns
      if (predictions.shape.length !== 2) {
        throw new Error(`Expected a 2D matrix, got ${predictions.shape.length}D array`);
      }
      
      const nTestPoints = predictions.shape[0];
      const nClasses = predictions.shape[1];
      
      if (nTestPoints !== trueY.length) {
        throw new Error(`Expected ${trueY.length} test points, got ${nTestPoints}`);
      }
      
      if (nClasses !== 97) {
        throw new Error(`Expected 97 logits per test point (mod 97), got ${nClasses}`);
      }

      // Calculate cross-entropy loss
      let totalLoss = 0;
      let correctPredictions = 0;
      
      for (let i = 0; i < nTestPoints; i++) {
        // Extract logits for this test point
        const logits = [];
        for (let j = 0; j < nClasses; j++) {
          logits.push(predictedLogits[i * nClasses + j]);
        }
        
        const trueClass = trueY[i];
        totalLoss += crossEntropyLoss(logits, trueClass);
        
        // Check if prediction is correct (argmax of logits)
        const predictedClass = logits.indexOf(Math.max(...logits));
        if (predictedClass === trueClass) {
          correctPredictions++;
        }
      }

      const avgLoss = totalLoss / nTestPoints;
      const acc = correctPredictions / nTestPoints;

      setScore(avgLoss);
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
              Your predictions will be evaluated using <strong>cross-entropy loss</strong>:
            </p>

            <BlockMath math="\text{CE} = -\frac{1}{N}\sum_{i=1}^{N} \log p_i(y_i)" />

            <p>
              where <InlineMath math="p_i(y_i)" /> is the predicted probability of the correct 
              class <InlineMath math="y_i" /> after applying softmax to your 97 logits.
              Lower scores are better.
            </p>

            <p className="font-medium text-blue-700">
              🎯 Goal: Try to achieve a cross-entropy loss less than 0.5!
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
              <strong>~4704 × 97 matrix of logits</strong> (one row per test point,
              97 logits per row representing scores for each class).
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
              <div className={`mt-4 p-4 border rounded-lg ${score < 0.5 ? 'bg-green-50 border-green-200' : 'bg-yellow-50 border-yellow-200'}`}>
                <h3 className={`font-medium mb-2 ${score < 0.5 ? 'text-green-900' : 'text-yellow-900'}`}>Your Score:</h3>
                <p className={`text-2xl font-bold ${score < 0.5 ? 'text-green-700' : 'text-yellow-700'}`}>
                  Cross-Entropy Loss = {score.toFixed(4)}
                </p>
                {accuracy !== null && (
                  <p className={`text-lg mt-2 ${score < 0.5 ? 'text-green-700' : 'text-yellow-700'}`}>
                    Accuracy = {(accuracy * 100).toFixed(2)}%
                  </p>
                )}
                <p className={`text-sm mt-2 ${score < 0.5 ? 'text-green-700' : 'text-yellow-700'}`}>
                  {score < 0.5 
                    ? '🎉 Great job! You beat the target of 0.5!' 
                    : '🎯 Try to get below 0.5! Lower is better.'}
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
