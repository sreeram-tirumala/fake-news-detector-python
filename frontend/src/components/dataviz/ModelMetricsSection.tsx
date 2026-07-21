import { useEffect, useState } from "react";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { getMetrics } from "../../api/client";
import type { ClassMetric, Metrics } from "../../api/types";

const REPORT_LABELS: Record<string, string> = {
  "0": "fake",
  "1": "real",
  "macro avg": "macro avg",
  "weighted avg": "weighted avg",
};

function isClassMetric(v: ClassMetric | number): v is ClassMetric {
  return typeof v === "object";
}

function downsample<T>(arr: T[], maxPoints = 150): T[] {
  if (arr.length <= maxPoints) return arr;
  const step = arr.length / maxPoints;
  return Array.from({ length: maxPoints }, (_, i) => arr[Math.floor(i * step)]);
}

export default function ModelMetricsSection() {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getMetrics()
      .then(setMetrics)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  if (error) return <p className="error-text">{error}</p>;
  if (!metrics) return <p className="caption">Loading...</p>;

  const rocData = downsample(
    metrics.curves.roc.fpr.map((fpr, i) => ({ fpr, tpr: metrics.curves.roc.tpr[i] })),
  );
  const prData = downsample(
    metrics.curves.pr.precision.map((precision, i) => ({
      precision,
      recall: metrics.curves.pr.recall[i],
    })),
  );

  return (
    <div className="tab-panel">
      <div className="card">
        <h3>Confusion matrix</h3>
        <table className="data-table">
          <thead>
            <tr>
              <th />
              <th>Predicted fake</th>
              <th>Predicted real</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th>Actual fake</th>
              <td>{metrics.confusion_matrix[0][0]}</td>
              <td>{metrics.confusion_matrix[0][1]}</td>
            </tr>
            <tr>
              <th>Actual real</th>
              <td>{metrics.confusion_matrix[1][0]}</td>
              <td>{metrics.confusion_matrix[1][1]}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div className="card">
        <h3>Per-class metrics</h3>
        <table className="data-table">
          <thead>
            <tr>
              <th>Class</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1</th>
              <th>Support</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(metrics.report)
              .filter((entry): entry is [string, ClassMetric] => isClassMetric(entry[1]))
              .map(([key, m]) => (
                <tr key={key}>
                  <td className="capitalize">{REPORT_LABELS[key] ?? key}</td>
                  <td>{m.precision.toFixed(3)}</td>
                  <td>{m.recall.toFixed(3)}</td>
                  <td>{m["f1-score"].toFixed(3)}</td>
                  <td>{m.support}</td>
                </tr>
              ))}
          </tbody>
        </table>
        {typeof metrics.report.accuracy === "number" && (
          <p className="caption">Overall accuracy: {(metrics.report.accuracy as number).toFixed(4)}</p>
        )}
      </div>

      <div className="card">
        <h3>ROC curve (AUC {metrics.curves.roc.auc.toFixed(4)})</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={rocData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="fpr" type="number" domain={[0, 1]} label={{ value: "FPR", position: "insideBottom", offset: -5 }} />
            <YAxis dataKey="tpr" type="number" domain={[0, 1]} label={{ value: "TPR", angle: -90, position: "insideLeft" }} />
            <Tooltip />
            <Line type="monotone" dataKey="tpr" stroke="#2a9d8f" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="card">
        <h3>Precision–recall curve</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={prData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="recall" type="number" domain={[0, 1]} label={{ value: "Recall", position: "insideBottom", offset: -5 }} />
            <YAxis dataKey="precision" type="number" domain={[0, 1]} label={{ value: "Precision", angle: -90, position: "insideLeft" }} />
            <Tooltip />
            <Line type="monotone" dataKey="precision" stroke="#457b9d" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
