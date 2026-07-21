import { useState, type ComponentType } from "react";
import MiniPredictor from "../MiniPredictor";
import DataWranglingSection from "../dataviz/DataWranglingSection";
import TimeSeriesSection from "../dataviz/TimeSeriesSection";
import PcaClustersSection from "../dataviz/PcaClustersSection";
import TopicsSection from "../dataviz/TopicsSection";
import ModelMetricsSection from "../dataviz/ModelMetricsSection";

const VIEWS = [
  "Data Wrangling",
  "Time Series",
  "PCA & Clusters",
  "Topics (NMF)",
  "Model Metrics",
] as const;

type View = (typeof VIEWS)[number];

const SECTIONS: Record<View, ComponentType> = {
  "Data Wrangling": DataWranglingSection,
  "Time Series": TimeSeriesSection,
  "PCA & Clusters": PcaClustersSection,
  "Topics (NMF)": TopicsSection,
  "Model Metrics": ModelMetricsSection,
};

export default function DataVizTab() {
  const [view, setView] = useState<View>("Data Wrangling");
  const Section = SECTIONS[view];

  return (
    <div className="tab-panel">
      <MiniPredictor />

      <div className="dataviz-select-row">
        <label htmlFor="dataviz-view">Chart</label>
        <select
          id="dataviz-view"
          value={view}
          onChange={(e) => setView(e.target.value as View)}
          className="dataviz-select"
        >
          {VIEWS.map((v) => (
            <option key={v} value={v}>
              {v}
            </option>
          ))}
        </select>
      </div>

      <Section />
    </div>
  );
}
