import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { BlockMath, InlineMath } from 'react-katex';
import Plot from 'react-plotly.js';

export default function Case4Solutions() {
  const [trainingHistory, setTrainingHistory] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      // Helper function to parse CSV
      const parseCSV = (text) => {
        if (!text || text.trim().length === 0 || text.includes('<!DOCTYPE')) {
          return null;
        }
        const lines = text.trim().split('\n').slice(1); // Skip header
        
        const epochs = [];
        const trainLoss = [];
        const testLoss = [];
        const trainAccuracy = [];
        const testAccuracy = [];
        const weightNorm = [];
        
        for (const line of lines) {
          const parts = line.split(',');
          if (parts.length >= 5) {
            const epoch = parseInt(parts[0]);
            const tl = parseFloat(parts[1]);
            const tel = parseFloat(parts[2]);
            const ta = parseFloat(parts[3]);
            const tea = parseFloat(parts[4]);
            const wn = parts.length >= 6 ? parseFloat(parts[5]) : null;
            
            if (!isNaN(epoch)) {
              epochs.push(epoch);
              trainLoss.push(tl);
              testLoss.push(tel);
              trainAccuracy.push(ta);
              testAccuracy.push(tea);
              if (wn !== null && !isNaN(wn)) {
                weightNorm.push(wn);
              }
            }
          }
        }
        
        return { epochs, trainLoss, testLoss, trainAccuracy, testAccuracy, weightNorm };
      };

      try {
        // Load training history with weight decay
        const response = await fetch(`${import.meta.env.BASE_URL}case4/data/reference_training_loss_wd.csv`);
        const text = await response.text();
        const history = parseCSV(text);
        if (history) setTrainingHistory(history);
      } catch (err) {
        console.warn('Training history not available:', err);
      }
      setLoading(false);
    }
    
    loadData();
  }, []);

  return (
    <div className="min-h-screen bg-white">
      <div className="max-w-6xl mx-auto px-6 py-16">
        <Link to="/case4" className="text-blue-600 hover:text-blue-800 mb-8 inline-block">
          ← Back to challenge
        </Link>

        <h1 className="text-4xl font-light text-gray-900 mb-8">
          Case Study 4: Solutions
        </h1>

        <section className="mb-12">
          <h2 className="text-2xl font-medium text-gray-900 mb-4">The Task</h2>
          <div className="prose max-w-none text-gray-700 space-y-4">
            <p>
              This case study explores learning <strong>modular division mod 97</strong>.
              Given two numbers <InlineMath math="a" /> (0 to 96) and <InlineMath math="b" /> (1 to 96, non-zero),
              the model must predict <InlineMath math="(a / b) \mod 97" />.
            </p>
            
            <div className="bg-gray-50 p-6 rounded-lg my-6 overflow-x-auto">
              <h3 className="font-medium text-gray-900 mb-3">Problem Setup:</h3>
              <div className="overflow-x-auto space-y-2">
                <p><strong>Input:</strong> One-hot encoding of <InlineMath math="(a, b)" /> with 194 binary features</p>
                <p><strong>Output:</strong> Classification into 97 classes (the result of <InlineMath math="(a / b) \mod 97" />)</p>
                <p><strong>Total examples:</strong> <InlineMath math="97 \times 96 = 9312" /> valid pairs (b ≠ 0)</p>
                <p><strong>Train/Test split:</strong> Half and half (~4657 train, ~4656 test)</p>
              </div>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-medium text-gray-900 mb-4">Why This Problem is Interesting</h2>
          <div className="prose max-w-none text-gray-700 space-y-4">
            <p>
              Modular division is more challenging than modular addition because it requires
              understanding multiplicative inverses in modular arithmetic:
            </p>
            
            <BlockMath math="(a / b) \mod 97 = a \cdot b^{-1} \mod 97" />
            
            <p>
              Since 97 is prime, every non-zero element has a unique multiplicative inverse.
              By Fermat&apos;s Little Theorem:
            </p>
            
            <BlockMath math="b^{-1} \equiv b^{95} \pmod{97}" />

            <div className="bg-blue-50 p-6 rounded-lg my-6">
              <h3 className="font-medium text-gray-900 mb-3">Key Challenges:</h3>
              <ul className="list-disc list-inside space-y-2">
                <li>The model must learn the concept of multiplicative inverses</li>
                <li>The relationship between inputs and outputs is more complex than addition</li>
                <li>Simple pattern matching won&apos;t work—the model needs to discover the underlying group structure</li>
                <li>Division by different values of b requires different &quot;inversion&quot; patterns</li>
              </ul>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-medium text-gray-900 mb-4">Our Approach: MLP + Attention</h2>
          <div className="prose max-w-none text-gray-700 space-y-4">
            <div className="bg-gray-50 p-6 rounded-lg my-6">
              <h3 className="font-medium text-gray-900 mb-3">Architecture:</h3>
              <p className="mb-3">
                We use a hybrid architecture combining an MLP with multi-head attention:
              </p>
              <ul className="list-disc list-inside space-y-2">
                <li><strong>Input layer:</strong> 194 features (one-hot encoding)</li>
                <li><strong>MLP layer:</strong> 194 → 256 with ReLU activation</li>
                <li><strong>Attention layer:</strong> Reshape 256 to (4 tokens × 64 dims), apply multi-head self-attention</li>
                <li><strong>Output layer:</strong> 256 → 97 logits with softmax for classification</li>
                <li><strong>Loss function:</strong> Cross-entropy</li>
              </ul>
            </div>

            <div className="bg-purple-50 p-6 rounded-lg my-6">
              <h3 className="font-medium text-gray-900 mb-3">Why Attention Helps:</h3>
              <ul className="list-disc list-inside space-y-2">
                <li>The attention mechanism can learn to relate different parts of the hidden representation</li>
                <li>Self-attention enables the model to capture complex dependencies between the numerator and denominator</li>
                <li>The multi-head design allows learning multiple types of relationships simultaneously</li>
              </ul>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-medium text-gray-900 mb-4">The Role of Regularization</h2>
          <div className="prose max-w-none text-gray-700 space-y-4">
            <p>
              Weight decay regularization plays a crucial role in helping the network find
              a solution that generalizes well:
            </p>
            
            <div className="bg-purple-50 p-6 rounded-lg my-6">
              <ul className="list-disc list-inside space-y-2">
                <li>Without regularization, the network may memorize training data without generalizing</li>
                <li>Weight decay encourages simpler, more regularized solutions</li>
                <li>With proper regularization, the network learns patterns that transfer to unseen data</li>
                <li>The regularized solution captures the underlying mathematical structure
                of modular division rather than just memorizing input-output pairs</li>
              </ul>
            </div>
          </div>
        </section>

        {trainingHistory && (
          <section className="mb-12">
            <h2 className="text-2xl font-medium text-gray-900 mb-4">Training Results</h2>
            <div className="prose max-w-none text-gray-700 space-y-4 mb-6">
              <p>
                The plots below show real training of the MLP+Attention network (194→256→Attn→97)
                using AdamW optimizer with weight decay.
              </p>
            </div>

            <div className="bg-white border border-green-200 rounded-lg p-4">
              <h3 className="text-lg font-medium text-gray-900 mb-2">Training with Weight Decay</h3>
              <p className="text-sm text-gray-600 mb-4">Network learns to generalize with proper regularization</p>
              
              <Plot
                data={[
                  {
                    x: trainingHistory.epochs,
                    y: trainingHistory.trainAccuracy.map(a => a * 100),
                    mode: 'lines+markers',
                    name: 'Train Accuracy',
                    line: { color: 'rgba(59, 130, 246, 1)', width: 2 },
                    marker: { size: 8 },
                  },
                  {
                    x: trainingHistory.epochs,
                    y: trainingHistory.testAccuracy.map(a => a * 100),
                    mode: 'lines+markers',
                    name: 'Test Accuracy',
                    line: { color: 'rgba(34, 197, 94, 1)', width: 2 },
                    marker: { size: 8 },
                  },
                ]}
                layout={{
                  title: { text: 'Train vs Test Accuracy', font: { size: 16 } },
                  xaxis: { title: 'Epoch' },
                  yaxis: { title: 'Accuracy (%)', range: [0, 105] },
                  hovermode: 'closest',
                  showlegend: true,
                  legend: { x: 0.02, y: 0.5, bgcolor: 'rgba(255,255,255,0.8)' },
                  autosize: true,
                  margin: { l: 50, r: 20, t: 50, b: 50 },
                }}
                style={{ width: '100%', height: '400px' }}
                config={{ responsive: true }}
                useResizeHandler={true}
              />
              
              {trainingHistory.weightNorm && trainingHistory.weightNorm.length > 0 && (
                <Plot
                  data={[
                    {
                      x: trainingHistory.epochs,
                      y: trainingHistory.weightNorm,
                      mode: 'lines+markers',
                      name: 'Weight Norm',
                      line: { color: 'rgba(168, 85, 247, 1)', width: 2 },
                      marker: { size: 8 },
                    },
                  ]}
                  layout={{
                    title: { text: 'Total Weight Norm Over Time', font: { size: 16 } },
                    xaxis: { title: 'Epoch' },
                    yaxis: { title: 'L2 Norm of Weights' },
                    hovermode: 'closest',
                    showlegend: true,
                    legend: { x: 0.02, y: 0.98, bgcolor: 'rgba(255,255,255,0.8)' },
                    autosize: true,
                    margin: { l: 50, r: 20, t: 50, b: 50 },
                  }}
                  style={{ width: '100%', height: '400px' }}
                  config={{ responsive: true }}
                  useResizeHandler={true}
                />
              )}
              
              <div className="mt-4 bg-green-50 p-4 rounded-lg">
                <p className="text-sm text-gray-700">
                  <strong>Result:</strong> With weight decay, both train and test accuracy reach high values. 
                  The attention mechanism helps the network learn the complex structure of modular division.
                </p>
              </div>
            </div>
              
            <div className="mt-6 prose max-w-none text-gray-700 text-sm bg-blue-50 p-4 rounded-lg">
              <p><strong>Key observations:</strong></p>
              <ul className="list-disc list-inside space-y-1 mt-2">
                <li>Weight decay regularization helps the network learn generalizable patterns</li>
                <li>The MLP+Attention architecture can learn the complex structure of modular division</li>
                <li>The network achieves good accuracy on unseen test data</li>
              </ul>
            </div>
          </section>
        )}

        {loading && (
          <section className="mb-12">
            <div className="text-center py-12">
              <div className="text-gray-600">Loading training history...</div>
            </div>
          </section>
        )}
      </div>
    </div>
  );
}
