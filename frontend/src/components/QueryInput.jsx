import { useState } from "react";

export default function QueryInput({ onSubmit, loading }) {
  const [query, setQuery] = useState("");

  const handleSubmit = () => {
    if (!query.trim()) return;
    onSubmit(query);
    setQuery("");
  };

  return (
    <div style={{ display: "flex", marginBottom: "1rem" }}>
      <input
        style={{ flex: 1, padding: "0.5rem", fontSize: "1rem" }}
        placeholder="Enter your query"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        disabled={loading}
      />
      <button
        style={{ marginLeft: "0.5rem", padding: "0.5rem 1rem" }}
        onClick={handleSubmit}
        disabled={loading}
      >
        Ask
      </button>
    </div>
  );
}
