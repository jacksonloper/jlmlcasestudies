import { Link } from 'react-router-dom';

export default function Landing() {
  const caseStudies = [
    {
      id: 'case1',
      title: 'Case Study 1: Point Prediction',
      description: 'Predict a single y value given x (evaluated with RMSE)',
    },
    {
      id: 'case2',
      title: 'Case Study 2: Distribution Sampling',
      description: 'Generate two samples from the conditional distribution (evaluated with Energy Score)',
    },
    {
      id: 'case3',
      title: 'Case Study 3: Modular Arithmetic',
      description: 'Learn modular addition mod 97 with one-hot encodings (evaluated with Accuracy)',
    },
    {
      id: 'case4',
      title: 'Case Study 4: Modular Division',
      description: 'Learn modular division mod 97 with one-hot encodings (evaluated with Accuracy)',
    },
    {
      id: 'case5',
      title: 'Case Study 5: Multispectral Imaging',
      description: 'Explore predicting multispectral data from RGB smartphone images (Beyond RGB dataset)',
    },
  ];

  return (
    <div className="min-h-screen bg-white">
      <div className="max-w-4xl mx-auto px-6 py-16">
        <header className="mb-16">
          <h1 className="text-4xl font-light text-gray-900 mb-2">
            ML Case Studies
          </h1>
          <p className="text-gray-600">
            A collection of machine learning challenges and experiments
          </p>
        </header>

        <div className="space-y-4">
          {caseStudies.map((study) => (
            <Link
              key={study.id}
              to={`/${study.id}`}
              className="block border border-gray-200 rounded-lg p-6 hover:border-gray-400 transition-colors"
            >
              <h2 className="text-xl font-medium text-gray-900 mb-2">
                {study.title}
              </h2>
              <p className="text-gray-600">{study.description}</p>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
}
