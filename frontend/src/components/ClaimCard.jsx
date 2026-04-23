export default function ClaimCard({ claim }) {
  const colorMap = {
    Supported: "#d4edda",
    Refuted: "#f8d7da",
    "Not Enough Info": "#fff3cd",
  };

  return (
    <div
      style={{
        backgroundColor: colorMap[claim.label] || "#f1f1f1",
        padding: "0.75rem",
        marginBottom: "0.5rem",
        borderRadius: "4px",
      }}
    >
      <p>
        <strong>Claim:</strong> {claim.claim}
      </p>
      <p>
        <strong>Status:</strong> {claim.label}
      </p>
      {claim.evidence && claim.evidence.length > 0 && (
        <div>
          <strong>Evidence:</strong>
          <ul>
            {claim.evidence.map((e, i) => (
              <li key={i}>{e}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
