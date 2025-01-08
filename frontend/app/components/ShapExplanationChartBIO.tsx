"use client";
import { useState, useEffect } from "react";
import axios from "axios";
import { Bar } from "react-chartjs-2";

const ShapExplanationChart = () => {
  const [shapData, setShapData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  // Replace with your actual Flask SHAP endpoint
  const SHAP_API_URL = "http://localhost:5000/api/explain_biological_activity";

  useEffect(() => {
    const fetchShapData = async () => {
      try {
        const response = await axios.get(SHAP_API_URL);
        setShapData(response.data);
        setLoading(false);
      } catch (err) {
        console.error("Error fetching SHAP data:", err);
        setError(true);
        setLoading(false);
      }
    };

    fetchShapData();
  }, []);

  if (loading) {
    return <div className="text-center">Loading SHAP...</div>;
  }
  if (error || !shapData) {
    return (
      <div className="text-center text-red-500">Error loading SHAP data.</div>
    );
  }

  // Suppose response has structure:
  // {
  //   shap_values: [ [valMolWt, valDonors, valAcceptors, valLogP], ... ],
  //   feature_names: ["MolWt", "NumHDonors", "NumHAcceptors", "Predicted_logP"],
  //   message: "Some message"
  // }
  // If it’s a single summary for the entire dataset, you might have an array of shap_values or an average.

  // For example, let's assume we have the "average absolute shap value" per feature
  // We'll do a simple bar chart with the mean of absolute shap values for each feature

  const featureNames = shapData.feature_names;
  const allShapValues = shapData.shap_values; // shape (N, 4) for example

  // Compute mean absolute SHAP per feature
  const meanAbsShap = featureNames.map((_, fIdx) => {
    let sum = 0;
    for (let i = 0; i < allShapValues.length; i++) {
      sum += Math.abs(allShapValues[i][fIdx]);
    }
    return sum / allShapValues.length;
  });

  const data = {
    labels: featureNames,
    datasets: [
      {
        label: "Mean |SHAP| value",
        data: meanAbsShap,
        backgroundColor: "rgba(75,192,192,0.6)",
      },
    ],
  };

  const options = {
    responsive: true,
    scales: {
      y: { beginAtZero: true },
    },
  };

  return (
    <div className="mt-4">
      <h2 className="text-xl font-semibold mb-2">
        SHAP Explanation (Mean |SHAP|)
      </h2>
      <Bar data={data} options={options} />
    </div>
  );
};

export default ShapExplanationChart;
