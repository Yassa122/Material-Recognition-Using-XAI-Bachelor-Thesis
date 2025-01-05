import React, { useEffect, useState } from "react";
import axios from "axios";
// --- import Recharts components ---
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ResponsiveContainer,
} from "recharts";

interface PropertyMetrics {
  property: string;
  mse: number | null;
  mae: number | null;
  r2: number | null;
  accuracy: number | null;
}

const ModelMetricsCard: React.FC = () => {
  const [metricsData, setMetricsData] = useState<PropertyMetrics[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    axios
      .get("http://localhost:5000/api/model-metrics")
      .then((res) => {
        // The backend returns an object, e.g. { logP: {...}, num_atoms: {...}, pIC50: {...} }
        const data = res.data;

        // Transform the object into an array of { property, mse, mae, r2, accuracy }
        if (data && typeof data === "object") {
          const arr: PropertyMetrics[] = Object.entries(data).map(
            ([key, val]: [string, any]) => ({
              property: key,
              mse: val.mse,
              mae: val.mae,
              r2: val.r2,
              accuracy: val.accuracy,
            })
          );
          setMetricsData(arr);
        } else {
          setError("Received data is not the expected object.");
        }
      })
      .catch((err) => {
        setError("Failed to fetch metrics.");
      })
      .finally(() => {
        setLoading(false);
      });
  }, []);

  const formatValue = (val: number | null | undefined) => {
    if (val === null || val === undefined) return "N/A";
    return val.toFixed(4);
  };

  if (loading) return <div className="text-white">Loading...</div>;
  if (error) return <div className="text-red-500">{error}</div>;

  // Prepare data for the bar chart
  // We'll store each metric in separate fields, e.g. { property: 'logP', MSE: 0.12, MAE: 0.05, ... }
  const chartData = metricsData.map((d) => ({
    property: d.property,
    MSE: d.mse ?? 0,
    MAE: d.mae ?? 0,
    R2: d.r2 ?? 0,
    Accuracy: d.accuracy ?? 0,
  }));

  return (
    <div className="bg-[#202020] p-6 rounded-lg shadow-lg">
      <h3 className="text-white text-lg font-semibold mb-4">
        Property Metrics
      </h3>

      {/* ---------------- Table of Metrics ---------------- */}
      <table className="w-full text-left text-gray-300 mb-8">
        <thead>
          <tr className="border-b border-gray-600">
            <th className="py-2">Property</th>
            <th className="py-2">MSE</th>
            <th className="py-2">MAE</th>
            <th className="py-2">R²</th>
            <th className="py-2">Accuracy</th>
          </tr>
        </thead>
        <tbody>
          {metricsData.map((row, index) => (
            <tr key={index} className="border-b border-gray-800">
              <td className="py-3">{row.property}</td>
              <td className="py-3">{formatValue(row.mse)}</td>
              <td className="py-3">{formatValue(row.mae)}</td>
              <td className="py-3">{formatValue(row.r2)}</td>
              <td className="py-3">
                {row.accuracy !== null && row.accuracy !== undefined
                  ? `${(row.accuracy * 100).toFixed(2)}%`
                  : "N/A"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      {/* ---------------- Bar Chart for Metrics ---------------- */}
      <div style={{ width: "100%", height: 400 }}>
        <ResponsiveContainer>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#555" />
            <XAxis dataKey="property" stroke="#ccc" />
            <YAxis stroke="#ccc" />
            <Tooltip
              contentStyle={{ background: "#333", border: "1px solid #ccc" }}
              labelStyle={{ color: "#fff" }}
            />
            <Legend />
            {/* Grouped bars for each metric */}
            <Bar dataKey="MSE" fill="#82ca9d" />
            <Bar dataKey="MAE" fill="#8884d8" />
            <Bar dataKey="R2" fill="#ffc658" />
            <Bar dataKey="Accuracy" fill="#d0ed57" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default ModelMetricsCard;
