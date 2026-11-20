import { useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
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
import RiddleLab from '@case5/RiddleLab';
import RiddleLeaderboard from '@case5/RiddleLeaderboard';

const SITE = 'ML Case Studies';
const TITLES = {
  '/case1': 'Case Study 1',
  '/case1/solutions': 'Case Study 1 Solutions',
  '/case2': 'Case Study 2',
  '/case2/solutions': 'Case Study 2 Solutions',
  '/case3': 'Case Study 3',
  '/case3/solutions': 'Case Study 3 Solutions',
  '/case4': 'Case Study 4',
  '/case4/solutions': 'Case Study 4 Solutions',
  '/case5': 'Case Study 5',
  '/case5/lab': 'Riddle Lab',
  '/case5/leaderboard': 'Riddle Lab Leaderboard',
};

// Sets the browser tab title for each page when the route changes.
function PageTitle() {
  const { pathname } = useLocation();
  useEffect(() => {
    const page = TITLES[pathname.replace(/\/+$/, '')];
    document.title = page ? `${page} · ${SITE}` : SITE;
  }, [pathname]);
  return null;
}

function App() {
  return (
    <Router basename={import.meta.env.BASE_URL}>
      <PageTitle />
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
        <Route path="/case5/lab" element={<RiddleLab />} />
        <Route path="/case5/leaderboard" element={<RiddleLeaderboard />} />
      </Routes>
    </Router>
  );
}

export default App;
