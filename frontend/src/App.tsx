import { useState } from "react";
import "./App.css";
import PredictTab from "./components/tabs/PredictTab";
import DataWranglingTab from "./components/tabs/DataWranglingTab";
import TimeSeriesTab from "./components/tabs/TimeSeriesTab";
import PcaClustersTab from "./components/tabs/PcaClustersTab";
import TopicsTab from "./components/tabs/TopicsTab";
import ModelMetricsTab from "./components/tabs/ModelMetricsTab";

const TABS = [
  "Predict",
  "Data Wrangling",
  "Time Series",
  "PCA & Clusters",
  "Topics (NMF)",
  "Model Metrics",
] as const;

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
        {active === "Data Wrangling" && <DataWranglingTab />}
        {active === "Time Series" && <TimeSeriesTab />}
        {active === "PCA & Clusters" && <PcaClustersTab />}
        {active === "Topics (NMF)" && <TopicsTab />}
        {active === "Model Metrics" && <ModelMetricsTab />}
      </div>
    </div>
  );
}

export default App;
