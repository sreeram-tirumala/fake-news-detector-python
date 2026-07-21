import { useEffect, useState } from "react";
import { getTopics } from "../../api/client";
import type { Topic } from "../../api/types";

export default function TopicsSection() {
  const [topics, setTopics] = useState<Topic[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getTopics()
      .then(setTopics)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  if (error) return <p className="error-text">{error}</p>;
  if (!topics) return <p className="caption">Loading...</p>;

  return (
    <div className="tab-panel">
      <div className="card">
        <h3>NMF topics</h3>
        <p className="caption">
          Factor-analysis-style topics extracted from TF-IDF via non-negative matrix
          factorization, each shown as its top contributing terms.
        </p>
        <ul className="topic-list">
          {topics.map((t) => (
            <li key={t.topic}>
              <strong>Topic {t.topic}</strong> — {t.terms.join(", ")}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
