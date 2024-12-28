// IntegratedGradientsView.tsx
import React, { useEffect, useState } from "react";
import axios from "axios";

interface IGResponse {
  smiles_list: string[];
  attributions: number[][];
  convergence_delta: number;
}

const IntegratedGradientsView: React.FC = () => {
  const [igData, setIgData] = useState<IGResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    axios
      .post<IGResponse>("http://127.0.0.1:5000/api/integrated-gradients", {
        // e.g., pass an array of SMILES or an ID of the sample to explain
        smiles_list: ["CCO", "CC(C)=O"],
      })
      .then((res) => {
        setIgData(res.data);
      })
      .catch((err) => {
        setError("Failed to compute integrated gradients.");
      })
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <div className="text-white">Loading Integrated Gradients...</div>;
  }

  if (error) {
    return <div className="text-red-500">{error}</div>;
  }

  if (!igData) {
    return <div className="text-white">No data returned.</div>;
  }

  return (
    <div className="bg-gray-800 p-6 rounded shadow-md text-white">
      <h2 className="text-xl font-semibold mb-4">Integrated Gradients</h2>
      <p className="mb-4">
        Convergence Delta: {igData.convergence_delta.toFixed(6)}
      </p>
      {igData.smiles_list.map((smiles, i) => (
        <div key={i} className="mb-4">
          <h3 className="font-bold">{`SMILES #${i + 1}: ${smiles}`}</h3>
          <ul className="list-inside list-disc ml-4">
            {igData.attributions[i].map((attr, j) => (
              <li key={j}>
                Token {j}:{" "}
                <span className="text-blue-300">{attr.toFixed(4)}</span>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
};

export default IntegratedGradientsView;
