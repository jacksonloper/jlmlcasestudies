import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { BlockMath, InlineMath } from 'react-katex';
import Plot from 'react-plotly.js';

export default function Case3Solutions() {
  const [trainingHistory, setTrainingHistory] = useState(null);
  const [trainingHistoryWD, setTrainingHistoryWD] = useState(null);
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
        
        for (const line of lines) {
          const parts = line.split(',');
          if (parts.length >= 5) {
            const epoch = parseInt(parts[0]);
            const tl = parseFloat(parts[1]);
            const tel = parseFloat(parts[2]);
            const ta = parseFloat(parts[3]);
            const tea = parseFloat(parts[4]);
            
            if (!isNaN(epoch)) {
              epochs.push(epoch);
              trainLoss.push(tl);
              testLoss.push(tel);
              trainAccuracy.push(ta);
              testAccuracy.push(tea);
            }
          }
        }
        
        return { epochs, trainLoss, testLoss, trainAccuracy, testAccuracy };
      };

      try {
        // Load both training histories in parallel
        const [responseNoWD, responseWD] = await Promise.all([
          fetch(`${import.meta.env.BASE_URL}case3/data/reference_training_loss.csv`),
          fetch(`${import.meta.env.BASE_URL}case3/data/reference_training_loss_wd.csv`)
        ]);
        
        const [textNoWD, textWD] = await Promise.all([
          responseNoWD.text(),
          responseWD.text()
        ]);
        
        const historyNoWD = parseCSV(textNoWD);
        const historyWD = parseCSV(textWD);
        
        if (historyNoWD) setTrainingHistory(historyNoWD);
        if (historyWD) setTrainingHistoryWD(historyWD);
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
        <Link to="/case3" className="text-blue-600 hover:text-blue-800 mb-8 inline-block">
          ← Back to challenge
        </Link>

        <h1 className="text-4xl font-light text-gray-900 mb-8">
          Case Study 3: Solutions
        </h1>

        <section className="mb-12">
          <h2 className="text-2xl font-medium text-gray-900 mb-4">The Task</h2>
          <div className="prose max-w-none text-gray-700 space-y-4">
            <p>
              This case study explores learning <strong>modular arithmetic mod 97</strong>.
              Given two numbers <InlineMath math="a" /> and <InlineMath math="b" /> (each between 0 and 96),
              the model must predict <InlineMath math="(a + b) \mod 97" />.
            </p>
            
            <div className="bg-gray-50 p-6 rounded-lg my-6 overflow-x-auto">
              <h3 className="font-medium text-gray-900 mb-3">Problem Setup:</h3>
              <div className="overflow-x-auto space-y-2">
                <p><strong>Input:</strong> One-hot encoding of <InlineMath math="(a, b)" /> with 194 binary features</p>
                <p><strong>Output:</strong> Classification into 97 classes (the result of <InlineMath math="(a + b) \mod 97" />)</p>
                <p><strong>Total examples:</strong> <InlineMath math="97^2 = 9409" /> possible pairs</p>
                <p><strong>Train/Test split:</strong> Half and half (~4705 train, ~4704 test)</p>
              </div>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-medium text-gray-900 mb-4">Why This Problem is Interesting</h2>
          <div className="prose max-w-none text-gray-700 space-y-4">
            <p>
              Modular arithmetic is a classic test case for neural network generalization.
              The key insight is that the operation has a beautiful structure:
            </p>
            
            <BlockMath math="(a + b) \mod 97" />
            
            <p>
              This can be understood through the lens of cyclic groups. The integers mod 97
              form a cyclic group under addition, and the operation &quot;wraps around&quot; at 97.
            </p>

            <div className="bg-blue-50 p-6 rounded-lg my-6">
              <h3 className="font-medium text-gray-900 mb-3">Key Challenges:</h3>
              <ul className="list-disc list-inside space-y-2">
                <li>The model must learn the cyclic structure of modular arithmetic</li>
                <li>Simple linear models cannot capture this pattern</li>
                <li>The one-hot encoding provides no positional information about the numbers</li>
                <li>Generalization to unseen pairs requires understanding the underlying structure</li>
              </ul>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-medium text-gray-900 mb-4">Approaches</h2>
          <div className="prose max-w-none text-gray-700 space-y-4">
            <div className="bg-gray-50 p-6 rounded-lg my-6">
              <h3 className="font-medium text-gray-900 mb-3">Neural Network Approach:</h3>
              <p className="mb-3">
                A multi-layer perceptron (MLP) can learn to perform modular arithmetic by
                discovering the underlying structure. Key architectural choices:
              </p>
              <ul className="list-disc list-inside space-y-2">
                <li>Input layer: 194 features (one-hot encoding)</li>
                <li>Hidden layers: Experiment with different widths and depths</li>
                <li>Output layer: 97 logits with softmax for classification</li>
                <li>Loss function: Cross-entropy</li>
              </ul>
            </div>

            <div className="bg-green-50 p-6 rounded-lg my-6">
              <h3 className="font-medium text-gray-900 mb-3">Fourier Feature Approach:</h3>
              <p className="mb-3">
                Modular arithmetic has a natural representation in terms of complex exponentials.
                If we embed each number <InlineMath math="k" /> as:
              </p>
              <BlockMath math="e^{2\pi i k / 97}" />
              <p className="mt-3">
                Then addition mod 97 becomes multiplication of complex numbers!
                This suggests that networks with periodic activation functions
                or explicit Fourier features may perform well.
              </p>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-medium text-gray-900 mb-4">Evaluation</h2>
          <div className="prose max-w-none text-gray-700 space-y-4">
            <p>
              The model is evaluated using <strong>cross-entropy loss</strong> on the test set:
            </p>
            
            <BlockMath math="\text{CE} = -\frac{1}{N}\sum_{i=1}^{N} \log p_i(y_i)" />
            
            <p>
              where <InlineMath math="p_i(y_i)" /> is the predicted probability for the correct class
              after applying softmax to the 97 output logits.
            </p>

            <div className="bg-yellow-50 p-6 rounded-lg my-6">
              <h3 className="font-medium text-gray-900 mb-3">Benchmarks:</h3>
              <table className="min-w-full">
                <thead>
                  <tr className="border-b border-gray-300">
                    <th className="text-left py-2 pr-4">Model</th>
                    <th className="text-left py-2 pr-4">Cross-Entropy</th>
                    <th className="text-left py-2">Notes</th>
                  </tr>
                </thead>
                <tbody className="text-sm">
                  <tr className="border-b border-gray-200">
                    <td className="py-2 pr-4">Random guessing</td>
                    <td className="py-2 pr-4">~4.57</td>
                    <td className="py-2"><InlineMath math="-\log(1/97)" /></td>
                  </tr>
                  <tr className="border-b border-gray-200">
                    <td className="py-2 pr-4">Simple MLP</td>
                    <td className="py-2 pr-4">~1.0</td>
                    <td className="py-2">Basic architecture</td>
                  </tr>
                  <tr className="border-b border-gray-200">
                    <td className="py-2 pr-4">Optimal</td>
                    <td className="py-2 pr-4">~0.0</td>
                    <td className="py-2">Perfect predictions</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-medium text-gray-900 mb-4">Connection to Grokking</h2>
          <div className="prose max-w-none text-gray-700 space-y-4">
            <p>
              This task is related to the phenomenon of <strong>grokking</strong> discovered
              by Power et al. (2022). They observed that neural networks trained on modular
              arithmetic tasks can exhibit delayed generalization:
            </p>
            
            <div className="bg-purple-50 p-6 rounded-lg my-6">
              <ul className="list-disc list-inside space-y-2">
                <li>Training loss decreases rapidly (memorization)</li>
                <li>Test loss remains high for many epochs</li>
                <li>With <strong>weight decay regularization</strong>, test loss suddenly drops (generalization)</li>
              </ul>
              <p className="mt-3 text-sm">
                This &quot;grokking&quot; behavior suggests the network is learning the underlying
                algorithmic structure rather than just memorizing the training data.
                <strong>Note:</strong> Weight decay is critical for grokking to occur.
              </p>
            </div>
          </div>
        </section>

        {(trainingHistory || trainingHistoryWD) && (
          <section className="mb-12">
            <h2 className="text-2xl font-medium text-gray-900 mb-4">Training Progress: Comparing Approaches</h2>
            <div className="prose max-w-none text-gray-700 space-y-4 mb-6">
              <p>
                The plots below compare two training runs of the same neural network architecture 
                (194→128→128→97 with ReLU, Adam optimizer):
              </p>
              <ul className="list-disc list-inside space-y-1">
                <li><strong>Without weight decay:</strong> Shows memorization (100% train accuracy, ~0% test accuracy)</li>
                <li><strong>With weight decay:</strong> Shows grokking (delayed generalization to test set)</li>
              </ul>
            </div>

            {/* Side by side comparison */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              
              {/* No Weight Decay Section */}
              {trainingHistory && (
                <div className="bg-white border border-gray-200 rounded-lg p-4">
                  <h3 className="text-lg font-medium text-gray-900 mb-2">Without Weight Decay (Memorization)</h3>
                  <p className="text-sm text-gray-600 mb-4">Network memorizes training data but cannot generalize</p>
                  
                  <Plot
                    data={[
                      {
                        x: trainingHistory.epochs,
                        y: trainingHistory.trainAccuracy.map(a => a * 100),
                        mode: 'lines',
                        name: 'Train Acc',
                        line: { color: 'rgba(59, 130, 246, 1)', width: 2 },
                      },
                      {
                        x: trainingHistory.epochs,
                        y: trainingHistory.testAccuracy.map(a => a * 100),
                        mode: 'lines',
                        name: 'Test Acc',
                        line: { color: 'rgba(239, 68, 68, 1)', width: 2 },
                      },
                    ]}
                    layout={{
                      title: { text: 'Accuracy (No Weight Decay)', font: { size: 14 } },
                      xaxis: { title: 'Epoch' },
                      yaxis: { title: 'Accuracy (%)', range: [0, 105] },
                      hovermode: 'closest',
                      showlegend: true,
                      legend: { x: 0.02, y: 0.98, bgcolor: 'rgba(255,255,255,0.8)' },
                      autosize: true,
                      margin: { l: 50, r: 20, t: 40, b: 50 },
                    }}
                    style={{ width: '100%', height: '300px' }}
                    config={{ responsive: true }}
                    useResizeHandler={true}
                  />
                  
                  <div className="mt-3 text-sm text-gray-600">
                    <p><strong>Result:</strong> Train accuracy hits 100% quickly, test accuracy stays near 0%.</p>
                  </div>
                </div>
              )}

              {/* With Weight Decay Section */}
              {trainingHistoryWD && (
                <div className="bg-white border border-green-200 rounded-lg p-4">
                  <h3 className="text-lg font-medium text-gray-900 mb-2">With Weight Decay (Grokking)</h3>
                  <p className="text-sm text-gray-600 mb-4">Network eventually learns to generalize</p>
                  
                  <Plot
                    data={[
                      {
                        x: trainingHistoryWD.epochs,
                        y: trainingHistoryWD.trainAccuracy.map(a => a * 100),
                        mode: 'lines',
                        name: 'Train Acc',
                        line: { color: 'rgba(59, 130, 246, 1)', width: 2 },
                      },
                      {
                        x: trainingHistoryWD.epochs,
                        y: trainingHistoryWD.testAccuracy.map(a => a * 100),
                        mode: 'lines',
                        name: 'Test Acc',
                        line: { color: 'rgba(34, 197, 94, 1)', width: 2 },
                      },
                    ]}
                    layout={{
                      title: { text: 'Accuracy (With Weight Decay)', font: { size: 14 } },
                      xaxis: { title: 'Epoch' },
                      yaxis: { title: 'Accuracy (%)', range: [0, 105] },
                      hovermode: 'closest',
                      showlegend: true,
                      legend: { x: 0.02, y: 0.98, bgcolor: 'rgba(255,255,255,0.8)' },
                      autosize: true,
                      margin: { l: 50, r: 20, t: 40, b: 50 },
                    }}
                    style={{ width: '100%', height: '300px' }}
                    config={{ responsive: true }}
                    useResizeHandler={true}
                  />
                  
                  <div className="mt-3 text-sm text-gray-600">
                    <p><strong>Result:</strong> Test accuracy eventually catches up to train accuracy (grokking).</p>
                  </div>
                </div>
              )}
            </div>

            {/* Loss comparison */}
            {trainingHistory && trainingHistoryWD && (
              <div className="mt-6 bg-white border border-gray-200 rounded-lg p-4">
                <h3 className="text-lg font-medium text-gray-900 mb-3">Loss Comparison</h3>
                <Plot
                  data={[
                    {
                      x: trainingHistory.epochs,
                      y: trainingHistory.testLoss,
                      mode: 'lines',
                      name: 'Test Loss (No WD)',
                      line: { color: 'rgba(239, 68, 68, 1)', width: 2 },
                    },
                    {
                      x: trainingHistoryWD.epochs,
                      y: trainingHistoryWD.testLoss,
                      mode: 'lines',
                      name: 'Test Loss (With WD)',
                      line: { color: 'rgba(34, 197, 94, 1)', width: 2 },
                    },
                  ]}
                  layout={{
                    title: { text: 'Test Loss: Without vs With Weight Decay', font: { size: 16 } },
                    xaxis: { title: 'Epoch' },
                    yaxis: { title: 'Cross-Entropy Loss', type: 'log' },
                    hovermode: 'closest',
                    showlegend: true,
                    legend: { x: 0.7, y: 0.98, bgcolor: 'rgba(255,255,255,0.8)' },
                    autosize: true,
                    margin: { l: 60, r: 20, t: 50, b: 50 },
                  }}
                  style={{ width: '100%', height: '400px' }}
                  config={{ responsive: true }}
                  useResizeHandler={true}
                />
              </div>
            )}
              
            <div className="mt-6 prose max-w-none text-gray-700 text-sm bg-blue-50 p-4 rounded-lg">
              <p><strong>Key observations:</strong></p>
              <ul className="list-disc list-inside space-y-1 mt-2">
                <li><strong>Without weight decay (red):</strong> Test loss increases continuously - pure memorization</li>
                <li><strong>With weight decay (green):</strong> Test loss eventually drops - the network &quot;groks&quot; the pattern</li>
                <li>Weight decay acts as a regularizer that prevents memorization and encourages learning generalizable patterns</li>
                <li>The &quot;grokking&quot; phenomenon shows that generalization can happen suddenly after many epochs of apparent overfitting</li>
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
