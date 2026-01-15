import { useState } from 'react';
import { Link } from 'react-router-dom';

// MIS sensor has 16 spectral channels (4x4 mosaic pattern)
// Wavelengths span approximately 385-723nm
const NUM_BANDS = 16;
const WAVELENGTHS = [
  385, 407, 430, 453,   // UV-Blue range
  475, 498, 520, 543,   // Blue-Green range  
  565, 588, 610, 633,   // Green-Orange range
  655, 678, 700, 723    // Orange-Red range
];

export default function Case5() {
  const [showMIS, setShowMIS] = useState(false);
  const [selectedBand, setSelectedBand] = useState(7); // Default to ~543nm (green)

  // Get wavelength for band index
  const getWavelength = (bandIndex) => WAVELENGTHS[bandIndex] || 550;
  
  // Get approximate color for wavelength (for visual indicator)
  const getWavelengthColor = (wavelength) => {
    if (wavelength < 450) return '#8B00FF'; // Violet
    if (wavelength < 480) return '#0000FF'; // Blue
    if (wavelength < 510) return '#00FFFF'; // Cyan
    if (wavelength < 560) return '#00FF00'; // Green
    if (wavelength < 590) return '#FFFF00'; // Yellow
    if (wavelength < 620) return '#FF7F00'; // Orange
    return '#FF0000'; // Red
  };

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
              and a professional multispectral imaging system (MIS) capturing <strong>16 spectral channels</strong> from 
              approximately 385-723nm using a 4×4 mosaic filter array.
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
                    src={`${import.meta.env.BASE_URL}case5/data/mis_band_${selectedBand.toString().padStart(2, '0')}.png`}
                    alt={`MIS Spectral band ${selectedBand} (${getWavelength(selectedBand)}nm)`}
                    className="w-full"
                  />
                  <div className="bg-gray-50 p-4">
                    <h3 className="font-medium text-gray-900">
                      Spectral Channel {selectedBand + 1}: {getWavelength(selectedBand)}nm
                      <span 
                        className="inline-block w-4 h-4 ml-2 rounded-full align-middle"
                        style={{ backgroundColor: getWavelengthColor(getWavelength(selectedBand)) }}
                      />
                    </h3>
                    
                    {/* Band Slider */}
                    <div className="mt-4 space-y-2">
                      <div className="flex justify-between text-xs text-gray-500">
                        <span>385nm (UV/Violet)</span>
                        <span>543nm (Green)</span>
                        <span>723nm (Red/NIR)</span>
                      </div>
                      <input
                        type="range"
                        min="0"
                        max={NUM_BANDS - 1}
                        value={selectedBand}
                        onChange={(e) => setSelectedBand(parseInt(e.target.value))}
                        className="w-full h-3 rounded-lg appearance-none cursor-pointer"
                        style={{
                          background: 'linear-gradient(to right, violet, blue, cyan, green, yellow, orange, red)'
                        }}
                      />
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-gray-600">
                          Channel {selectedBand + 1} of {NUM_BANDS}
                        </span>
                        <span className="text-sm font-medium" style={{ color: getWavelengthColor(getWavelength(selectedBand)) }}>
                          {getWavelength(selectedBand)}nm
                        </span>
                      </div>
                    </div>
                    
                    <p className="text-sm text-gray-600 mt-3">
                      Grayscale image showing reflectance at {getWavelength(selectedBand)}nm wavelength.
                      Use the slider to explore all 16 spectral channels.
                    </p>
                    <p className="text-xs text-gray-500 mt-1">
                      Original resolution: 646 × 484 pixels (demosaiced from 4×4 pattern)
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
              <li><strong>MIS:</strong> Multispectral mosaic images (2584×1936, float32) with 4×4 filter array → 16 channels at 646×484</li>
              <li><strong>WT (White Target):</strong> Calibrated images with color checker</li>
              <li><strong>NT (No Target):</strong> Scene images without color checker</li>
            </ul>

            <h3 className="text-lg font-medium text-gray-900 mt-6">Spectral Channels</h3>
            <p>
              The MIS sensor uses a 4×4 mosaic filter array to capture <strong>16 spectral channels</strong> spanning 
              approximately 385-723nm. Each channel has a specific spectral response curve that 
              determines its sensitivity to different wavelengths of light.
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
