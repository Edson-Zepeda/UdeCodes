import { NavLink, Route, Routes } from "react-router-dom";

import DashboardPage from "./pages/DashboardPage";
import HomePage from "./pages/HomePage";
import LotsPage from "./pages/LotsPage";

// Import logo from repository root so Vite bundles it correctly.
import logoSrc from "../../Gategroup_logo.png";

const App = () => {
  return (
    <div className="app-shell">
      <header className="nav-bar">
        <div className="nav-bar__logo">
          <img src={logoSrc} alt="gategroup" style={{ height: 34, objectFit: "contain" }} />
        </div>
        <nav className="nav-bar__links">
          <NavLink to="/" className={({ isActive }) => (isActive ? "nav-bar__link nav-bar__link--active" : "nav-bar__link")}>
            Inicio
          </NavLink>
          <NavLink to="/lots" className={({ isActive }) => (isActive ? "nav-bar__link nav-bar__link--active" : "nav-bar__link")}>
            Lotes
          </NavLink>
          <NavLink
            to="/dashboard"
            className={({ isActive }) => (isActive ? "nav-bar__link nav-bar__link--active" : "nav-bar__link")}
          >
            Dashboard
          </NavLink>
        </nav>
        <a className="primary-button" href="https://github.com/Edson-Zepeda/UdeCodes" target="_blank" rel="noreferrer">
          Repositorio
        </a>
      </header>

      <main className="app-shell__content">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/lots" element={<LotsPage />} />
          <Route path="/dashboard" element={<DashboardPage />} />
        </Routes>
      </main>

      <footer className="footer">© 2025 SPIR · Smart Planning Intelligent Resources.</footer>
    </div>
  );
};

export default App;
