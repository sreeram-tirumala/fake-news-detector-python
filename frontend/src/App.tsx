import { useState } from "react";
import "./App.css";
import PredictTab from "./components/tabs/PredictTab";
import DataVizTab from "./components/tabs/DataVizTab";

const TABS = ["Predict", "Data Viz"] as const;

type Tab = (typeof TABS)[number];

function App() {
  const [active, setActive] = useState<Tab>("Predict");

  return (
    <div className="app">
      <h1>📰 AI-Driven Fake News Detector</h1>
      <nav className="tab-nav">
        {TABS.map((tab) => (
          <button
            key={tab}
            className={tab === active ? "tab active" : "tab"}
            onClick={() => setActive(tab)}
          >
            {tab}
          </button>
        ))}
      </nav>
      <div className="tab-content">
        {active === "Predict" && <PredictTab />}
        {active === "Data Viz" && <DataVizTab />}
      </div>
    </div>
  );
}

export default App;
