import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { getClassCounts, getNullRates, getTopTerms } from "../../api/client";
import type { ClassCounts, NullRates, TopTermsData } from "../../api/types";

export default function DataWranglingSection() {
  const [classCounts, setClassCounts] = useState<ClassCounts | null>(null);
  const [nullRates, setNullRates] = useState<NullRates | null>(null);
  const [topTerms, setTopTerms] = useState<TopTermsData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([getClassCounts(), getNullRates(), getTopTerms()])
      .then(([cc, nr, tt]) => {
        setClassCounts(cc);
        setNullRates(nr);
        setTopTerms(tt);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  if (error) return <p className="error-text">{error}</p>;
  if (!classCounts || !nullRates || !topTerms) return <p className="caption">Loading...</p>;

  const classData = Object.entries(classCounts).map(([label, count]) => ({ label, count }));
  const nullData = Object.entries(nullRates).map(([column, rate]) => ({ column, rate: rate * 100 }));

  return (
    <div className="tab-panel">
      <div className="card">
        <h3>Class balance</h3>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={classData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="label" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" fill="#2a9d8f" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="card">
        <h3>Missingness by column</h3>
        <ResponsiveContainer width="100%" height={180}>
          <BarChart data={nullData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="column" />
            <YAxis unit="%" />
            <Tooltip formatter={(v: number) => `${v.toFixed(2)}%`} />
            <Bar dataKey="rate" fill="#e76f51" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="card">
        <h3>Top n-grams by class</h3>
        <div className="two-col">
          {(["real", "fake"] as const).map((cls) => (
            <div key={cls}>
              <h4 className="capitalize">{cls}</h4>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Term</th>
                    <th>Count</th>
                  </tr>
                </thead>
                <tbody>
                  {topTerms[cls].slice(0, 15).map((t) => (
                    <tr key={t.term}>
                      <td>{t.term}</td>
                      <td>{t.count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
