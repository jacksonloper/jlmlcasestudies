import { useState } from 'react';
import { Link } from 'react-router-dom';

export default function Case5() {
  const [showMIS, setShowMIS] = useState(false);

  return (
    <div className="min-h-screen bg-white">
      <div className="max-w-5xl mx-auto px-6 py-16">
        <Link to="/" className="text-blue-600 hover:text-blue-800 mb-8 inline-block">
          ← Back to all case studies
        </Link>

        <h1 className="text-4xl font-light text-gray-900 mb-8">
          Case Study 5: Multispectral Imaging
        </h1>

        <section className="mb-12">
          <h2 className="text-2xl font-medium text-gray-900 mb-4">The Challenge</h2>
          <div className="prose max-w-none text-gray-700 space-y-4">
            <p>
              <strong>Can we predict multispectral information from a standard RGB smartphone camera?</strong>
            </p>
            <p>
              This example showcases data from the{' '}
              <a 
                href="https://zenodo.org/records/16848482" 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-blue-600 hover:text-blue-800"
              >
                Beyond RGB dataset
              </a>, which provides paired images from mobile phones (OPPO Find X5 Pro) 
              and a professional multispectral imaging system (MIS) capturing 31 spectral bands 
              from 400-700nm at 10nm intervals.
            </p>
            <p>
              The images below show the same scene captured by both devices under controlled 
              lighting conditions with a white target (WT) for calibration.
            </p>
            <p className="font-medium text-blue-700">
              🎯 The key question: How accurately can one reconstruct the full spectral 
              information from just the RGB channels of a smartphone camera?
            </p>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-medium text-gray-900 mb-4">Example Images</h2>
          <p className="text-gray-600 mb-4">
            Sample: <code className="bg-gray-100 px-2 py-1 rounded">13_mix26/WT</code> (mixed lighting condition)
          </p>
          
          <div className="space-y-8">
            {/* Image Toggle */}
            <div className="flex space-x-4 mb-4">
              <button
                onClick={() => setShowMIS(false)}
                className={`px-6 py-3 rounded-lg font-medium transition-colors ${
                  !showMIS 
                    ? 'bg-blue-600 text-white' 
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                OPPO RGB
              </button>
              <button
                onClick={() => setShowMIS(true)}
                className={`px-6 py-3 rounded-lg font-medium transition-colors ${
                  showMIS 
                    ? 'bg-green-600 text-white' 
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
              >
                MIS (Multispectral)
              </button>
            </div>

            {/* Image Display */}
            <div className="border border-gray-200 rounded-lg overflow-hidden">
              {!showMIS ? (
                <div>
                  <img 
                    src={`${import.meta.env.BASE_URL}case5/data/oppo_rgb.png`}
                    alt="OPPO Find X5 Pro RGB image"
                    className="w-full"
                  />
                  <div className="bg-gray-50 p-4">
                    <h3 className="font-medium text-gray-900">OPPO Find X5 Pro (RGB)</h3>
                    <p className="text-sm text-gray-600">
                      Standard RGB smartphone camera image (3 channels: Red, Green, Blue)
                    </p>
                    <p className="text-xs text-gray-500 mt-1">
                      Original resolution: 4096 × 3072 pixels
                    </p>
                  </div>
                </div>
              ) : (
                <div>
                  <img 
                    src={`${import.meta.env.BASE_URL}case5/data/mis_band_00.png`}
                    alt="MIS Multispectral mosaic image"
                    className="w-full"
                  />
                  <div className="bg-gray-50 p-4">
                    <h3 className="font-medium text-gray-900">Multispectral Imaging System (MIS)</h3>
                    <p className="text-sm text-gray-600">
                      Grayscale visualization of the multispectral mosaic image. 
                      The MIS captures 31 spectral bands (400-700nm) in a spatial mosaic pattern.
                    </p>
                    <p className="text-xs text-gray-500 mt-1">
                      Original resolution: 2584 × 1936 pixels
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-medium text-gray-900 mb-4">About the Dataset</h2>
          <div className="prose max-w-none text-gray-700 space-y-4">
            <p>
              The <strong>Beyond RGB</strong> dataset contains paired images captured simultaneously 
              by mobile phone cameras and a multispectral imaging system under various 
              lighting conditions. This enables research into spectral reconstruction 
              from RGB images.
            </p>
            
            <h3 className="text-lg font-medium text-gray-900 mt-6">Data Structure</h3>
            <ul className="list-disc list-inside space-y-2">
              <li><strong>OPPO:</strong> RGB images from OPPO Find X5 Pro (4096×3072×3, float32)</li>
              <li><strong>Samsung:</strong> RGB images from Samsung Galaxy S21+ (4032×3024×3, float32)</li>
              <li><strong>MIS:</strong> Multispectral mosaic images (2584×1936, float32) containing 31 bands</li>
              <li><strong>WT (White Target):</strong> Calibrated images with color checker</li>
              <li><strong>NT (No Target):</strong> Scene images without color checker</li>
            </ul>

            <h3 className="text-lg font-medium text-gray-900 mt-6">Spectral Bands</h3>
            <p>
              The MIS captures 31 narrow spectral bands from 400nm (violet) to 700nm (red) 
              at 10nm intervals. This provides detailed spectral information that goes 
              far beyond what standard RGB cameras can capture.
            </p>
          </div>
        </section>

        <section className="mb-12">
          <h2 className="text-2xl font-medium text-gray-900 mb-4">Potential Applications</h2>
          <div className="prose max-w-none text-gray-700">
            <ul className="list-disc list-inside space-y-2">
              <li>Spectral reconstruction from RGB images</li>
              <li>Color constancy and white balance algorithms</li>
              <li>Material classification and identification</li>
              <li>Agricultural monitoring (vegetation indices)</li>
              <li>Medical imaging (skin analysis, wound assessment)</li>
              <li>Art and document analysis</li>
            </ul>
          </div>
        </section>

        <section className="mb-12 bg-gray-50 p-6 rounded-lg">
          <h2 className="text-xl font-medium text-gray-900 mb-4">Citation</h2>
          <div className="font-mono text-sm text-gray-700 bg-white p-4 rounded border overflow-x-auto">
            <pre>{`@InProceedings{Glatt_2024_WACV,
    author    = {Glatt, Ortal and Ater, Yotam and Kim, Woo-Shik 
                 and Werman, Shira and Berby, Oded and Zini, Yael 
                 and Zelinger, Shay and Lee, Sangyoon and Choi, Heejin 
                 and Soloveichik, Evgeny},
    title     = {Beyond RGB: A Real World Dataset for Multispectral 
                 Imaging in Mobile Devices},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference 
                 on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2024},
    pages     = {4344-4354}
}`}</pre>
          </div>
          <p className="text-sm text-gray-600 mt-4">
            Full dataset available at:{' '}
            <a 
              href="https://zenodo.org/records/16848482" 
              target="_blank" 
              rel="noopener noreferrer"
              className="text-blue-600 hover:text-blue-800"
            >
              https://zenodo.org/records/16848482
            </a>
          </p>
        </section>
      </div>
    </div>
  );
}
