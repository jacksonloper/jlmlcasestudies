import { Link } from 'react-router-dom';
import { BlockMath, InlineMath } from 'react-katex';

export default function Case3Solutions() {
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
                <li>Suddenly, test loss drops dramatically (generalization)</li>
              </ul>
              <p className="mt-3 text-sm">
                This &quot;grokking&quot; behavior suggests the network is learning the underlying
                algorithmic structure rather than just memorizing the training data.
              </p>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
