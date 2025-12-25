import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Landing from './pages/Landing';
import Case1 from '@case1/Case1';
import Case1Solutions from '@case1/Case1Solutions';
import Case2 from '@case2/Case2';
import Case2Solutions from '@case2/Case2Solutions';

function App() {
  return (
    <Router basename={import.meta.env.BASE_URL}>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/case1" element={<Case1 />} />
        <Route path="/case1/solutions" element={<Case1Solutions />} />
        <Route path="/case2" element={<Case2 />} />
        <Route path="/case2/solutions" element={<Case2Solutions />} />
      </Routes>
    </Router>
  );
}

export default App;
