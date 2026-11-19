import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Landing from './pages/Landing';
import Case1 from '@case1/Case1';
import Case1Solutions from '@case1/Case1Solutions';
import Case2 from '@case2/Case2';
import Case2Solutions from '@case2/Case2Solutions';
import Case3 from '@case3/Case3';
import Case3Solutions from '@case3/Case3Solutions';
import Case4 from '@case4/Case4';
import Case4Solutions from '@case4/Case4Solutions';
import Case5 from '@case5/Case5';

function App() {
  return (
    <Router basename={import.meta.env.BASE_URL}>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/case1" element={<Case1 />} />
        <Route path="/case1/solutions" element={<Case1Solutions />} />
        <Route path="/case2" element={<Case2 />} />
        <Route path="/case2/solutions" element={<Case2Solutions />} />
        <Route path="/case3" element={<Case3 />} />
        <Route path="/case3/solutions" element={<Case3Solutions />} />
        <Route path="/case4" element={<Case4 />} />
        <Route path="/case4/solutions" element={<Case4Solutions />} />
        <Route path="/case5" element={<Case5 />} />
      </Routes>
    </Router>
  );
}

export default App;
