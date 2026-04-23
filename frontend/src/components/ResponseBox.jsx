export default function ResponseBox({ response }) {
  if (!response) return null;

  return (
    <div
      style={{
        border: "1px solid #ccc",
        padding: "1rem",
        marginBottom: "1rem",
      }}
    >
      <h2>LLM Response:</h2>
      <p>{response}</p>
    </div>
  );
}
