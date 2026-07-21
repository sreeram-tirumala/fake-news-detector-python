import { useEffect, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { getWeeklyCounts } from "../../api/client";
import type { WeeklyCount } from "../../api/types";

export default function TimeSeriesSection() {
  const [data, setData] = useState<WeeklyCount[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getWeeklyCounts()
      .then(setData)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  if (error) return <p className="error-text">{error}</p>;
  if (!data) return <p className="caption">Loading...</p>;

  const weeks = Array.from(new Set(data.map((d) => d.week))).sort();
  const byWeek = weeks.map((week) => {
    const row: Record<string, number | string> = { week };
    for (const d of data.filter((r) => r.week === week)) {
      row[d.label] = d.count;
    }
    return row;
  });

  return (
    <div className="tab-panel">
      <div className="card">
        <h3>Weekly article counts by class</h3>
        <ResponsiveContainer width="100%" height={340}>
          <LineChart data={byWeek}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="week" tick={{ fontSize: 11 }} />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="real" stroke="#2a9d8f" dot={false} />
            <Line type="monotone" dataKey="fake" stroke="#e76f51" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
