import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Landing from './pages/Landing';
import Case1 from './pages/Case1';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/case1" element={<Case1 />} />
      </Routes>
    </Router>
  );
}

export default App;
