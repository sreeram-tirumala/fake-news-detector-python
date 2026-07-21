import { useEffect, useState } from "react";
import {
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { getPca } from "../../api/client";
import type { PcaData } from "../../api/types";

const LABEL_COLORS: Record<string, string> = { real: "#2a9d8f", fake: "#e76f51" };
const CLUSTER_PALETTE = ["#2a9d8f", "#e76f51", "#457b9d", "#f4a261", "#8338ec", "#ffb703", "#06d6a0", "#ef476f"];

export default function PcaClustersSection() {
  const [data, setData] = useState<PcaData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getPca()
      .then(setData)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  if (error) return <p className="error-text">{error}</p>;
  if (!data) return <p className="caption">Loading...</p>;

  return (
    <div className="tab-panel">
      <div className="card">
        <h3>PCA (2D) with K-Means</h3>
        <div className="two-col">
          <div>
            <p className="caption">Colored by ground-truth label</p>
            <ResponsiveContainer width="100%" height={360}>
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" dataKey="x" tick={{ fontSize: 11 }} />
                <YAxis type="number" dataKey="y" tick={{ fontSize: 11 }} />
                <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                <Scatter data={data.points}>
                  {data.points.map((p, i) => (
                    <Cell key={i} fill={LABEL_COLORS[p.label] ?? "#888"} />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <div>
            <p className="caption">Colored by KMeans cluster</p>
            <ResponsiveContainer width="100%" height={360}>
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" dataKey="x" tick={{ fontSize: 11 }} />
                <YAxis type="number" dataKey="y" tick={{ fontSize: 11 }} />
                <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                <Scatter data={data.points}>
                  {data.points.map((p, i) => (
                    <Cell key={i} fill={CLUSTER_PALETTE[p.cluster % CLUSTER_PALETTE.length]} />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
