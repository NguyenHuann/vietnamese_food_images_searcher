import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Layout } from './components/Layout';
import Home from './pages/Home';
import DiscoveryResults from './pages/DiscoveryResults';
import DishDetail from './pages/DishDetail';
import History from './pages/History';

export default function App() {
  return (
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/results" element={<DiscoveryResults />} />
          <Route path="/detail" element={<DishDetail />} />
          <Route path="/history" element={<History />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}
