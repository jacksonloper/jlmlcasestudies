import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Landing from './pages/Landing';
import Case1 from './pages/Case1';
import Case1Solutions from './pages/Case1Solutions';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/case1" element={<Case1 />} />
        <Route path="/case1/solutions" element={<Case1Solutions />} />
      </Routes>
    </Router>
  );
}

export default App;
